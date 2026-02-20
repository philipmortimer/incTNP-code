import math
from abc import ABC
from typing import Tuple

import numpy as np
import torch
from torch import nn


class Embedding(nn.Module, ABC):
    def __init__(self, active_dims: Tuple[int, ...]):
        super().__init__()

        # Which dimensions to apply the embedding to.
        self.active_dims = active_dims


class FourierEmbedding(Embedding):
    def __init__(
        self,
        lower: float,
        upper: float,
        active_dim: int,
        assert_range: bool = True,
        num_wavelengths: int = 10,
    ):
        super().__init__((active_dim,))

        self.lower = lower
        self.upper = upper
        self.assert_range = assert_range
        self.num_wavelengths = num_wavelengths

        # We will use half of the dimensionality for `sin` and the other half for `cos`.
        if num_wavelengths % 2 != 0:
            raise ValueError("The dimensionality must be a multiple of two.")

    def forward(self, x: torch.Tensor):
        # If the input is not within the configured range, the embedding might be ambiguous!
        in_range = torch.logical_and(
            self.lower <= x.abs(), torch.all(x.abs() <= self.upper)
        )
        in_range_or_zero = torch.all(
            torch.logical_or(in_range, x == 0)
        )  # Allow zeros to pass through.
        if self.assert_range and not in_range_or_zero:
            raise AssertionError(
                f"The input tensor is not within the configured range"
                f" `[{self.lower}, {self.upper}]`."
            )

        # Always perform the expansion with `float64`s to avoid numerical accuracy shenanigans.
        x = x.double()

        wavelengths = torch.logspace(
            math.log10(self.lower),
            math.log10(self.upper),
            self.num_wavelengths // 2,
            base=10,
            device=x.device,
            dtype=x.dtype,
        )
        prod = x * 2 * np.pi / wavelengths
        encoding = torch.cat((torch.sin(prod), torch.cos(prod)), dim=-1)

        return encoding.float()  # Cast to `float32` to avoid incompatibilities.
