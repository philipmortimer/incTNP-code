from abc import ABC
from typing import Callable, Optional, Tuple

import einops
import torch
from check_shapes import check_shapes
from torch import nn

from ..utils.distances import sq_dist
from ..utils.grids import (
    associative_scan,
    construct_grid,
    flatten_grid,
    nearest_gridded_neighbours,
)


class BaseSetConv(nn.Module, ABC):
    def __init__(
        self,
        *,
        dims: int,
        init_lengthscale: float = 0.1,
        dist_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = sq_dist,
    ):
        super().__init__()

        # Construct lengthscales.
        init_lengthscale = torch.as_tensor(dims * [init_lengthscale], dtype=torch.float)
        self.lengthscale_param = nn.Parameter(
            (torch.tensor(init_lengthscale).exp() - 1).log(),
            requires_grad=True,
        )

        self.dist_fn = dist_fn

    @property
    def lengthscale(self):
        return 1e-5 + nn.functional.softplus(  # pylint: disable=not-callable
            self.lengthscale_param
        )


class SetConvGridEncoder(BaseSetConv):
    def __init__(
        self,
        *,
        grid_range: Tuple[Tuple[float, float], ...],
        grid_shape: Tuple[int, ...],
        use_nn: bool = False,
        **kwargs,
    ):
        super().__init__(
            **kwargs,
        )

        # Construct grid.
        self.register_buffer("grid", construct_grid(grid_range, grid_shape))
        self.use_nn = use_nn

    @check_shapes(
        "x: [m, n, dx]",
        "z: [m, n, dz]",
        "return[0]: [m, ..., dx]",
        "return[1]: [m, ..., dz]",
    )
    def forward(
        self, x: torch.Tensor, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        grid_shape = self.grid.shape[:-1]
        grid_str = " ".join([f"n{i}" for i in range(len(grid_shape))])
        x_grid = einops.repeat(
            self.grid, grid_str + " d -> b " + grid_str + " d", b=x.shape[0]
        )

        if self.use_nn:
            z_grid = setconv_to_grid_nn(
                x, z, x_grid, self.lengthscale, dist_fn=self.dist_fn
            )
        else:
            z_grid = setconv_to_grid(
                x, z, x_grid, self.lengthscale, dist_fn=self.dist_fn
            )

        return x_grid, z_grid


class SetConvGridDecoder(BaseSetConv):
    def __init__(
        self,
        *,
        top_k_ctot: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(
            **kwargs,
        )

        self.top_k_ctot = top_k_ctot

    @check_shapes(
        "xc: [m, ..., dx]",
        "zc: [m, ..., dz]",
        "xt: [m, nt, dx]",
        "zt: [m, nt, dz]",
        "return: [m, nt, d]",
    )
    def forward(
        self,
        xc: torch.Tensor,
        zc: torch.Tensor,
        xt: torch.Tensor,
        zt: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        # Flatten grids
        xc_flat, _ = flatten_grid(xc)  # shape (batch_size, num_grid_points, Dx)
        zc_flat, _ = flatten_grid(zc)  # shape (batch_size, num_grid_points, Dz)

        if self.top_k_ctot is not None:
            num_batches, nt = xt.shape[:2]

            nearest_idx, mask = nearest_gridded_neighbours(
                xt,
                xc,
                k=self.top_k_ctot,
            )
            batch_idx = (
                torch.arange(num_batches)
                .unsqueeze(-1)
                .unsqueeze(-1)
                .repeat(1, nt, nearest_idx.shape[-1])
            )

            nearest_zc = zc_flat[
                batch_idx,
                nearest_idx,
            ]
            nearest_xc = xc_flat[
                batch_idx,
                nearest_idx,
            ]

            # Rearrange tokens.
            nearest_zc = einops.rearrange(nearest_zc, "b n k e -> (b n) k e")
            nearest_xc = einops.rearrange(nearest_xc, "b n k e -> (b n) k e")
            mask = einops.rearrange(mask, "b n e -> (b n) 1 e")

            # Compute kernel weights.
            xt_flat = einops.rearrange(xt, "b n e -> (b n) 1 e")
            weights = compute_weights(
                x1=xt_flat,
                x2=nearest_xc,
                lengthscales=self.lengthscale,
                dist_fn=self.dist_fn,
            )

            # Apply mask to weights.
            weights = weights * mask
            zt_update_flat = weights @ nearest_zc

            # Reshape output to (batch_size, num_trg, e).
            zt_update = einops.rearrange(
                zt_update_flat, "(b n) 1 e -> b n e", b=num_batches
            )

            if zt is not None:
                zt = zt + zt_update
            else:
                zt = zt_update

        else:
            # Compute kernel weights.
            weights = compute_weights(
                x1=xt,
                x2=xc_flat,
                lengthscales=self.lengthscale,
                dist_fn=self.dist_fn,
            )

            # Shape (batch_size, num_trg, num_grid_points).
            zt_update = weights @ zc_flat

            if zt is not None:
                zt = zt + zt_update
            else:
                zt = zt_update

        return zt


@check_shapes(
    "x: [m, n, dx]",
    "z: [m, n, dz]",
    "x_grid: [m, ..., dx]",
    "z_grid: [m, ..., dz]",
    "return: [m, ..., dz]",
)
def setconv_to_grid(
    x: torch.Tensor,
    z: torch.Tensor,
    x_grid: torch.Tensor,
    lengthscale: torch.Tensor,
    z_grid: Optional[torch.Tensor] = None,
    dist_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = sq_dist,
):
    x_grid_flat, flat_to_grid_fn = flatten_grid(x_grid)

    dists2 = dist_fn(x_grid_flat, x)
    pre_exp = torch.sum(dists2 / lengthscale.pow(2), dim=-1)
    weights = torch.exp(-0.5 * pre_exp)

    # Multiply context outputs by weights.
    # (batch_size, num_grid_points, embed_dim).
    z_grid_flat = weights @ z

    # Reshape grid.
    if z_grid is None:
        return flat_to_grid_fn(z_grid_flat)

    return z_grid + flat_to_grid_fn(z_grid_flat)


@check_shapes(
    "x: [m, n, dx]",
    "z: [m, n, dz]",
    "x_grid: [m, ..., dx]",
    "z_grid: [m, ..., dz]",
    "return: [m, ..., dz]",
)
def setconv_to_grid_nn(
    x: torch.Tensor,
    z: torch.Tensor,
    x_grid: torch.Tensor,
    lengthscale: torch.Tensor,
    z_grid: Optional[torch.Tensor] = None,
    dist_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = sq_dist,
):
    num_batches, num_points = x.shape[:2]

    # Flatten grid.
    x_grid_flat, flat_to_grid_fn = flatten_grid(x_grid)
    num_grid_points = x_grid_flat.shape[1]

    # (batch_size, n).
    nearest_idx, _ = nearest_gridded_neighbours(x, x_grid, k=1)
    nearest_idx = nearest_idx[..., 0]

    n_batch_idx = torch.arange(num_batches).unsqueeze(-1).repeat(1, num_points)
    n_range_idx = torch.arange(num_points).repeat(num_batches, 1)

    _, inverse_indices = torch.unique(nearest_idx, return_inverse=True)

    sorted_indices = nearest_idx.argsort(dim=1, stable=True)
    inverse_indices_sorted = inverse_indices.gather(1, sorted_indices).type(torch.long)
    unsorted_indices = sorted_indices.argsort(dim=1, stable=True)

    # Store changes in value.
    inverse_indices_diff = inverse_indices_sorted - inverse_indices_sorted.roll(
        1, dims=1
    )
    inverse_indices_diff = torch.where(
        inverse_indices_diff == 0,
        torch.ones_like(inverse_indices_diff),
        torch.zeros_like(inverse_indices_diff),
    )
    inverse_indices_diff[:, 0] = torch.zeros_like(inverse_indices_diff[:, 0])

    adjusted_cumsum = associative_scan(
        inverse_indices_diff, inverse_indices_diff, dim=1
    )
    adjusted_cumsum = adjusted_cumsum.round().int()
    cumcount_idx = adjusted_cumsum.gather(1, unsorted_indices)

    max_patch = cumcount_idx.amax() + 1

    # Create tensor with for each grid-token all nearest off-grid + itself in third axis.
    joint_grid_z = torch.full(
        (num_batches * num_grid_points, max_patch, z.shape[-1]),
        -torch.inf,
        device=z.device,
    )
    joint_grid_x = torch.full(
        (num_batches * num_grid_points, max_patch, x.shape[-1]),
        -torch.inf,
        device=x.device,
    )

    # Add nearest off the grid points to each on_the_grid point.
    idx_shifter = torch.arange(
        0, num_batches * num_grid_points, num_grid_points, device=z.device
    ).repeat_interleave(num_points)
    joint_grid_z[nearest_idx.flatten() + idx_shifter, cumcount_idx.flatten()] = z[
        n_batch_idx.flatten(), n_range_idx.flatten()
    ]
    joint_grid_x[nearest_idx.flatten() + idx_shifter, cumcount_idx.flatten()] = x[
        n_batch_idx.flatten(), n_range_idx.flatten()
    ]

    # Create a mask to ignore fake tokens.
    att_mask = torch.ones(
        num_batches * num_grid_points, 1, max_patch, device=z.device, dtype=torch.bool
    )
    att_mask[(joint_grid_z.sum(-1) == -float("inf")).unsqueeze(1)] = False
    joint_grid_z = torch.masked_fill(joint_grid_z, joint_grid_z == -float("inf"), 0.0)
    joint_grid_x = torch.masked_fill(joint_grid_x, joint_grid_x == -float("inf"), 0.0)

    # Rearrange x_grid_flat to be of shape (batch_size * num_grid_points, 1, embed_dim).
    x_grid_flat = einops.rearrange(x_grid_flat, "b m e -> (b m) 1 e")

    # Now we can do the setconv.
    weights = compute_weights(x_grid_flat, joint_grid_x, lengthscale, dist_fn)

    # Apply mask to weights.
    weights = weights * att_mask
    z_grid_flat = weights @ joint_grid_z

    # Reshape output.
    z_grid_flat = einops.rearrange(z_grid_flat, "(b m) 1 e -> b m e", b=num_batches)

    # Reshape grid.
    if z_grid is None:
        return flat_to_grid_fn(z_grid_flat)

    return z_grid + flat_to_grid_fn(z_grid_flat)


def compute_weights(
    x1: torch.Tensor,
    x2: torch.Tensor,
    lengthscales: torch.Tensor,
    dist_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = sq_dist,
) -> torch.Tensor:
    """Compute the weights for the kernel weighted sum."""

    # Expand dimensions for broadcasting.
    dists2 = dist_fn(x1, x2)

    pre_exp = torch.sum(dists2 / lengthscales.pow(2), dim=-1)
    return torch.exp(-0.5 * pre_exp)
