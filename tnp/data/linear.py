"""
Implements linear model generation from https://arxiv.org/pdf/2306.15063.pdf
"""

from typing import Tuple

import gpytorch
import torch

from .gp import GPGroundTruthPredictor
from .synthetic import SyntheticGeneratorBimodalInput, SyntheticGeneratorUniformInput


class LinearGeneratorUniformInput(SyntheticGeneratorUniformInput):
    def __init__(self, *, noise_std: float, **kwargs):
        super().__init__(**kwargs)

        self.noise_std = noise_std

    def sample_outputs(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, GPGroundTruthPredictor]:
        w = self.sample_weight()
        gt_pred = LinearGroundTruthPredictor(prior_std=1.0, noise_std=self.noise_std)

        # Generate observations.
        f = x @ w[..., None]
        y = f + self.noise_std * torch.randn_like(f)

        return y, gt_pred

    def sample_weight(self) -> torch.Tensor:
        # Sample weight vector.
        w = torch.randn((self.batch_size, self.dim))

        return w


class LinearGeneratorBimodalInput(SyntheticGeneratorBimodalInput):
    def __init__(self, *, noise_std: float, **kwargs):
        super().__init__(**kwargs)

        self.noise_std = noise_std

    def sample_outputs(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, GPGroundTruthPredictor]:
        w = self.sample_weight()
        gt_pred = LinearGroundTruthPredictor(prior_std=1.0, noise_std=self.noise_std)

        # Generate observations.
        f = x @ w[..., None]
        y = f + self.noise_std * torch.randn_like(f)

        return y, gt_pred

    def sample_weight(self) -> torch.Tensor:
        # Sample weight vector.
        w = torch.randn((self.batch_size, self.dim))

        return w


class LinearGroundTruthPredictor(GPGroundTruthPredictor):
    def __init__(self, prior_std: float, noise_std: float):
        kernel = gpytorch.kernels.LinearKernel()
        kernel.variance = prior_std**2
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        likelihood.noise = noise_std**2

        super().__init__(kernel, likelihood)
