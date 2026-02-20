from typing import Tuple

import torch

from .synthetic import (
    SyntheticGenerator,
    SyntheticGeneratorBimodalInput,
    SyntheticGeneratorUniformInput,
)


class SawtoothGeneratorBase(SyntheticGenerator):
    def __init__(self, *, min_freq: float, max_freq: float, noise_std: float, **kwargs):
        super().__init__(**kwargs)

        self.min_freq = min_freq
        self.max_freq = max_freq
        self.noise_std = noise_std

    def sample_outputs(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, None]:
        # Sample a frequency.
        freq = self.sample_freq()

        # Sample a direction.
        direction = torch.randn((self.batch_size, x.shape[-1]))
        direction = direction / (direction.abs())

        # Sample a uniformly distributed (conditional on frequency) offset.
        sample = torch.rand((self.batch_size,))
        offset = sample / freq

        # Construct the sawtooth and add noise.
        f = (
            freq[:, None, None] * (x @ direction[:, :, None] - offset[:, None, None])
        ) % 1
        y = f + self.noise_std * torch.randn_like(f)

        return y, None

    def sample_freq(self) -> torch.Tensor:
        # Sample frequency.
        freq = (
            torch.rand((self.batch_size,)) * (self.max_freq - self.min_freq)
            + self.min_freq
        )
        return freq


class SawtoothGenerator(SawtoothGeneratorBase, SyntheticGeneratorUniformInput):
    pass


class SawtoothGeneratorBimodalInput(
    SawtoothGeneratorBase, SyntheticGeneratorBimodalInput
):
    pass
