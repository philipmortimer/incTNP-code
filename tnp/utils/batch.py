from typing import Callable, Tuple

import numpy as np
import torch


def batch(x: torch.Tensor, other_dims: int) -> torch.Size:
    """Get the shape of the batch of a tensor.

    Args:
        x (tensor): Tensor.
        other_dims (int): Number of non-batch dimensions.

    Returns:
        tuple[int]: Shape of batch dimensions.
    """
    return x.size()[:-other_dims]


def compress_batch_dimensions(
    x: torch.Tensor, other_dims: int
) -> Tuple[torch.Tensor, Callable]:
    """Compress multiple batch dimensions of a tensor into a single batch dimension.

    Args:
        x (tensor): Tensor to compress.
        other_dims (int): Number of non-batch dimensions.

    Returns:
        tensor: `x` with batch dimensions compressed.
        function: Function to undo the compression of the batch dimensions.
    """
    b = batch(x, other_dims)
    if len(b) == 1:
        return x, lambda x: x

    def uncompress(x_after):
        return torch.reshape(x_after, (*b, *x_after.size()[1:]))

    return (
        torch.reshape(x, (int(np.prod(b)), *x.size()[len(b) :])),
        uncompress,
    )


def data(x: torch.Tensor, other_dims: int) -> torch.Size:
    """Get the shape of the data of a tensor.

    Args:
        x (tensor): Tensor.
        other_dims (int): Number of batch dimensions.

    Returns:
        tuple[int]: Shape of data dimensions.
    """
    return x.size()[other_dims:]


def compress_data_dimensions(
    x: torch.Tensor, other_dims: int
) -> Tuple[torch.Tensor, Callable]:
    """Compress multiple data dimensions of a tensor into a single data dimension.

    Args:
        x (tensor): Tensor to compress.
        other_dims (int): Number of batch dimensions.

    Returns:
        tensor: `x` with data dimensions compressed.
        function: Function to undo the compression of the data dimensions.
    """
    d = data(x, other_dims)
    if len(d) == 1:
        return x, lambda x: x

    def uncompress(x_after):
        return torch.reshape(x_after, (*x_after.size()[:-1], *d))

    return (torch.reshape(x, (*x.size()[: len(d)], int(np.prod(d)))), uncompress)
