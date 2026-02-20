from collections.abc import Iterable
from typing import List, Optional, Tuple, Union

import torch
from torch import nn

from ..utils.batch import compress_batch_dimensions
from .embeddings import Embedding


class MLP(nn.Module):
    """MLP.

    Args:
        in_dim (int): Input dimensionality.
        out_dim (int): Output dimensionality.
        layers (tuple[int, ...], optional): Width of every hidden layer.
        num_layers (int, optional): Number of hidden layers.
        width (int, optional): Width of the hidden layers
        nonlinearity (function, optional): Nonlinearity.
        dtype (dtype, optional): Data type.

    Attributes:
        net (object): MLP, but which expects a different data format.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        layers: Optional[Tuple[int, ...]] = None,
        num_layers: Optional[int] = None,
        width: Optional[int] = None,
        nonlinearity: Optional[nn.Module] = None,
    ):
        super().__init__()

        if layers is None:
            # Check that one of the two specifications is given.
            assert (
                num_layers is not None and width is not None
            ), "Must specify either `layers` or `num_layers` and `width`."
            layers = (width,) * num_layers

        # Default to ReLUs.
        if nonlinearity is None:
            nonlinearity = nn.ReLU()

        # Build layers.
        if len(layers) == 0:
            self.net = nn.Linear(in_dim, out_dim)
        else:
            net = [nn.Linear(in_dim, layers[0])]
            for i in range(1, len(layers)):
                net.append(nonlinearity)
                net.append(nn.Linear(layers[i - 1], layers[i]))
            net.append(nonlinearity)
            net.append(nn.Linear(layers[-1], out_dim))
            self.net = nn.Sequential(*net)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, uncompress = compress_batch_dimensions(x, 1)
        x = self.net(x)
        x = uncompress(x)
        return x


class MLPWithEmbedding(MLP):
    def __init__(
        self,
        embeddings: Union[Embedding, Tuple[Embedding, ...]],
        **kwargs,
    ):
        super().__init__(**kwargs)

        if not isinstance(embeddings, Iterable):
            embeddings = (embeddings,)

        self.embeddings = nn.ModuleList(embeddings)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # Embed the input.
        embedded_dims: List[int] = []
        embeddings: List[torch.Tensor] = []
        for embedding in self.embeddings:
            embedded_dims.extend(embedding.active_dims)
            embeddings.append(embedding(x[..., embedding.active_dims]))

        embedded_dims = [
            dim if dim >= 0 else x.shape[-1] + dim for dim in embedded_dims
        ]
        non_embedded_dims = [
            dim for dim in range(x.shape[-1]) if dim not in embedded_dims
        ]
        x_non_embedded = x[..., non_embedded_dims]
        x = torch.cat((x_non_embedded, *embeddings), dim=-1)
        out = super().forward(x)
        return out
