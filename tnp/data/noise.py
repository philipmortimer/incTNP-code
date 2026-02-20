from typing import Optional, Tuple

import torch

from .synthetic import SyntheticGenerator


class NoiseGenerator(SyntheticGenerator):

    def sample_inputs(
        self,
        nc: int,
        batch_shape: torch.Size,
        nt: Optional[int] = None,
    ) -> torch.Tensor:
        if nt is not None:
            return torch.randn((*batch_shape, nc + nt, self.dim))

        return torch.randn((*batch_shape, nc, self.dim))

    def sample_outputs(self, x: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return torch.randn((*x.shape[:-1], 1)), None
