import itertools
import math
from typing import Callable, Optional, Tuple

import einops
import torch
from check_shapes import check_shapes


def flatten_grid(
    x: torch.Tensor,
    start_dim: int = 1,
    end_dim: int = -1,
) -> Tuple[torch.Tensor, Callable[[torch.Tensor], torch.Tensor]]:
    grid_shape = x.shape[start_dim:end_dim]
    n_strings = [f"n{i}" for i in range(len(grid_shape))]
    grid_pattern = f"b {' '.join(n_strings)} e"
    flat_pattern = f"b ({' '.join(n_strings)}) e"
    grid_to_flat = grid_pattern + " -> " + flat_pattern
    flat_to_grid = flat_pattern + " -> " + grid_pattern
    reshape_vars = dict(zip(n_strings, grid_shape))

    def grid_to_flat_fn(x: torch.Tensor) -> torch.Tensor:
        return einops.rearrange(x, grid_to_flat)

    def flat_to_grid_fn(x: torch.Tensor) -> torch.Tensor:
        return einops.rearrange(x, flat_to_grid, **reshape_vars)

    return grid_to_flat_fn(x), flat_to_grid_fn


def construct_grid(
    grid_range: Tuple[Tuple[float, float], ...],
    points_per_dim: Tuple[int, ...],
) -> torch.Tensor:
    grid_range_ = torch.as_tensor(grid_range)
    grid = torch.stack(
        torch.meshgrid(
            *[
                torch.linspace(
                    grid_range_[i, 0],
                    grid_range_[i, 1],
                    steps=points_per_dim[i],
                    dtype=torch.float,
                )
                for i in range(len(grid_range))
            ]
        ),
        dim=-1,
    )

    return grid


@check_shapes(
    "x: [m, n, dx]",
    "x_grid: [m, ..., dx]",
    "return[0]: [m, n, k]",
    "return[1]: [m, n, k]",
)
def nearest_gridded_neighbours(
    x: torch.Tensor,
    x_grid: torch.Tensor,
    k: int = 1,
    roll_dims: Optional[Tuple[int, ...]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    grid_shape = torch.as_tensor(x_grid.shape[1:-1], device=x.device)
    x_grid_flat, _ = flatten_grid(x_grid)

    # Get number of neighbors along each dimension.
    dim_x = x.shape[-1]
    num_grid_spacings = math.ceil(k ** (1 / dim_x))

    # Set roll_dims to the actual index if they are specified as (-x, )
    num_dims = len(grid_shape)
    if roll_dims is not None:
        roll_dims = tuple(roll_dim % num_dims for roll_dim in roll_dims)

    # Quick calculation of nearest grid neighbour.
    x_grid_min = x_grid_flat.amin(dim=1)
    x_grid_max = x_grid_flat.amax(dim=1)
    x_grid_spacing = (x_grid_max - x_grid_min) / (grid_shape - 1)

    nearest_multi_idx = (
        x - x_grid_min[:, None, :] + x_grid_spacing[:, None, :] / 2
    ) // x_grid_spacing[:, None, :]

    # Generate a base grid for combinations of grid_spacing offsets from main neighbor.
    base_grid = torch.tensor(
        list(
            itertools.product(
                torch.arange(
                    -num_grid_spacings // 2 + num_grid_spacings % 2,
                    num_grid_spacings // 2 + 1,
                ),
                repeat=dim_x,
            )
        ),
        device=x.device,
    ).float()

    # Reshape and expand the base grid
    base_grid = base_grid.view(1, 1, -1, dim_x).expand(
        *nearest_multi_idx.shape[:-1], -1, -1
    )
    # Expand the indices of nearest neighbors to account for more than 1.
    nearest_multi_idx_expanded = nearest_multi_idx.unsqueeze(2).expand(
        -1, -1, (num_grid_spacings + 1 - num_grid_spacings % 2) ** dim_x, -1
    )
    # Generate all combinations by adding the offsets to the main neighbor.
    nearest_multi_idx = nearest_multi_idx_expanded + base_grid

    # If not rolling_dims, do not allow neighbors to go off-grid.
    # Otherwise, roll the grid along the specified dimension.
    if roll_dims is None:
        nearest_multi_idx = torch.max(
            torch.min(nearest_multi_idx, grid_shape - 1), torch.zeros_like(grid_shape)
        ).squeeze(-2)
    else:
        nearest_multi_idx = torch.cat(
            [
                (
                    torch.max(
                        torch.min(nearest_multi_idx[..., i], grid_shape[i] - 1),
                        torch.tensor(0),
                    ).unsqueeze(-1)
                    if (i not in roll_dims)
                    # else (nearest_multi_idx[..., i] % grid_shape[i]).unsqueeze(-1)
                    else (
                        (nearest_multi_idx[..., i] % grid_shape[i])
                        + (nearest_multi_idx[..., i] // grid_shape[i])
                    ).unsqueeze(-1)
                )
                for i in range(num_dims)
            ],
            dim=-1,
        ).squeeze(-2)

    # Get strides.
    strides = torch.flip(
        torch.cumprod(
            torch.cat(
                (
                    torch.ones((1,), device=grid_shape.device),
                    torch.flip(grid_shape, dims=(0,)),
                ),
                dim=0,
            ),
            dim=0,
        )[:-1],
        dims=(0,),
    )

    # (batch_size, nt, num_neighbors).
    if k == 1:
        nearest_idx = (
            (nearest_multi_idx * strides).sum(dim=-1).type(torch.int).unsqueeze(-1)
        )
    else:
        nearest_idx = (
            (nearest_multi_idx * strides).sum(dim=-1).type(torch.int).unsqueeze(-1)
        ).squeeze(-1)

    if k != 1:
        # Get mask for MHCA.
        mask = torch.ones_like(nearest_idx, dtype=torch.bool)

        # Sort nearest_idx.
        sorted_nearest_idx, indices = torch.sort(nearest_idx, dim=-1, stable=True)

        # Find first occurence where consecutive elements are different.
        first_occurrence = torch.ones_like(sorted_nearest_idx, dtype=torch.bool)
        first_occurrence[..., 1:] = (
            sorted_nearest_idx[..., 1:] != sorted_nearest_idx[..., :-1]
        )

        # Back to the original shape.
        original_indices = torch.argsort(indices, dim=-1)
        mask = torch.gather(first_occurrence, dim=-1, index=original_indices)
    else:
        mask = None

    return nearest_idx, mask


def complex_log(float_input: torch.Tensor, eps=1e-6) -> torch.ComplexType:
    eps = float_input.new_tensor(eps)
    real = float_input.abs().maximum(eps).log()
    imag = (float_input < 0).to(float_input.dtype) * torch.pi

    return torch.complex(real, imag)


def associative_scan(
    values: torch.Tensor, coeffs: torch.Tensor, dim: int
) -> torch.Tensor:
    log_values = complex_log(values.float())
    log_coeffs = complex_log(coeffs.float())
    a_star = torch.cumsum(log_coeffs, dim=dim)
    log_x0_plus_b_star = torch.logcumsumexp(log_values - a_star, dim=dim)
    log_x = a_star + log_x0_plus_b_star

    return torch.exp(log_x).real
