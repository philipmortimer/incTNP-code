from typing import Optional

import torch
from check_shapes import check_shapes
from torch import nn


class DeepSet(nn.Module):
    """Deep set.

    Args:
        phi (object): Pre-aggregation function.
        agg (object, optional): Aggregation function. Defaults to summing.

    Attributes:
        phi (object): Pre-aggregation function.
        agg (object): Aggregation function.
    """

    def __init__(
        self,
        z_encoder: nn.Module,
        agg: str = "sum",
    ):
        super().__init__()

        self.agg_strat_str = agg
        self.z_encoder = z_encoder

        if agg == "sum":
            self.agg = lambda x: torch.nansum(x, dim=-2)
        elif agg == "mean":
            self.agg = lambda x: torch.nanmean(x, dim=-2)
        else:
            raise ValueError("agg must be one of 'sum', 'mean'")

    @check_shapes(
        "x: [m, n, dx]",
        "y: [m, n, dy]",
        "mask: [m, n]",
        "return: [m, dz]",
    )
    def forward(
        self, x: torch.Tensor, y: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        z = torch.cat((x, y), dim=-1)
        z = self.z_encoder(z)
        if mask is not None:
            z[mask] = torch.nan
        z = self.agg(z)
        return z
