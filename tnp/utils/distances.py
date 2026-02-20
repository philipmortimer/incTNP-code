from typing import Callable, Tuple

import torch


def sq_dist(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """Compute the weights for the SetConv layer, mapping from `x1` to `x2`.

    Arguments:
        x1: Tensor of shape (batch_size, num_x1, dim)
        x2: Tensor of shape (batch_size, num_x2, dim)
        lengthscales: Tensor of shape (dim,) or (dim, num_lengthscales)

    Returns:
        Tensor of shape (batch_size, num_x1, num_x2, dim)
    """

    x1_ = x1[:, :, None, :]
    x2_ = x2[:, None, :, :]
    return (x1_ - x2_).pow(2)


def dist_composition(
    x1: torch.Tensor,
    x2: torch.Tensor,
    dist_fns: Tuple[Callable[[torch.Tensor, torch.Tensor], torch.Tensor], ...],
    dist_fn_dims: Tuple[Tuple[int, ...], ...],
) -> torch.Tensor:
    """
    Compute the distance composition of multiple distance functions.

    Arguments:
        x1: Tensor of shape (batch_size, num_x1, dim)
        x2: Tensor of shape (batch_size, num_x2, dim)
        dist_fns: Tuple of distance functions
        dist_fn_dims: Tuple of dimensions for each distance function
    """
    dists = [
        dist_fn(x1[..., dim_idx], x2[..., dim_idx])
        for dist_fn, dim_idx in zip(dist_fns, dist_fn_dims)
    ]
    return torch.cat(dists, dim=-1)
