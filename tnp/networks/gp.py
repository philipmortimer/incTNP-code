import random
from abc import ABC
from functools import partial
from typing import Tuple

import gpytorch
import torch

from tnp.networks.kernels import GibbsKernel, gibbs_switching_lengthscale_fn


class RandomHyperparameterKernel(ABC, gpytorch.kernels.Kernel):
    def sample_hyperparameters(self):
        pass


class ScaleKernel(gpytorch.kernels.ScaleKernel, RandomHyperparameterKernel):
    def __init__(
        self, min_log10_outputscale: float, max_log10_outputscale: float, **kwargs
    ):
        super().__init__(**kwargs)
        self.min_log10_outputscale = min_log10_outputscale
        self.max_log10_outputscale = max_log10_outputscale

    def sample_hyperparameters(self):
        # Sample outputscale.
        log10_outputscale = (
            torch.rand(()) * (self.max_log10_outputscale - self.min_log10_outputscale)
            + self.min_log10_outputscale
        )

        outputscale = 10.0**log10_outputscale
        self.outputscale = outputscale

        # Sample base kernel hyperparameters.
        self.base_kernel.sample_hyperparameters()


class RBFKernel(gpytorch.kernels.RBFKernel, RandomHyperparameterKernel):
    def __init__(
        self, min_log10_lengthscale: float, max_log10_lengthscale: float, **kwargs
    ):
        super().__init__(**kwargs)
        self.min_log10_lengthscale = min_log10_lengthscale
        self.max_log10_lengthscale = max_log10_lengthscale

    def sample_hyperparameters(self):
        # Sample lengthscale.
        shape = self.ard_num_dims if self.ard_num_dims is not None else ()
        log10_lengthscale = (
            torch.rand(shape)
            * (self.max_log10_lengthscale - self.min_log10_lengthscale)
            + self.min_log10_lengthscale
        )

        lengthscale = 10.0**log10_lengthscale
        self.lengthscale = lengthscale


class MaternKernel(gpytorch.kernels.MaternKernel, RandomHyperparameterKernel):
    def __init__(
        self, min_log10_lengthscale: float, max_log10_lengthscale: float, **kwargs
    ):
        super().__init__(**kwargs)
        self.min_log10_lengthscale = min_log10_lengthscale
        self.max_log10_lengthscale = max_log10_lengthscale

    def sample_hyperparameters(self):
        # Sample lengthscale.
        shape = self.ard_num_dims if self.ard_num_dims is not None else ()
        log10_lengthscale = (
            torch.rand(shape)
            * (self.max_log10_lengthscale - self.min_log10_lengthscale)
            + self.min_log10_lengthscale
        )

        lengthscale = 10.0**log10_lengthscale
        self.lengthscale = lengthscale


class PeriodicKernel(gpytorch.kernels.PeriodicKernel, RandomHyperparameterKernel):
    def __init__(
        self,
        min_log10_lengthscale: float,
        max_log10_lengthscale: float,
        min_log10_period: float,
        max_log10_period: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.min_log10_lengthscale = min_log10_lengthscale
        self.max_log10_lengthscale = max_log10_lengthscale
        self.min_log10_period = min_log10_period
        self.max_log10_period = max_log10_period

    def sample_hyperparameters(self):
        # Sample lengthscale.
        shape = self.ard_num_dims if self.ard_num_dims is not None else ()
        log10_lengthscale = (
            torch.rand(shape)
            * (self.max_log10_lengthscale - self.min_log10_lengthscale)
            + self.min_log10_lengthscale
        )

        lengthscale = 10.0**log10_lengthscale
        self.lengthscale = lengthscale

        # Sample period.
        log10_period = (
            torch.rand(shape) * (self.max_log10_period - self.min_log10_period)
            + self.min_log10_period
        )

        period = 10.0**log10_period
        self.period_length = period


class CosineKernel(gpytorch.kernels.CosineKernel, RandomHyperparameterKernel):
    def __init__(self, min_log10_period: float, max_log10_period: float, **kwargs):
        super().__init__(**kwargs)
        self.min_log10_period = min_log10_period
        self.max_log10_period = max_log10_period

    def sample_hyperparameters(self):
        # Sample period.
        log10_period = (
            torch.rand(()) * (self.max_log10_period - self.min_log10_period)
            + self.min_log10_period
        )

        period = 10.0**log10_period
        self.period_length = period


class RandomGibbsKernel(GibbsKernel, RandomHyperparameterKernel):
    def __init__(
        self,
        changepoints: Tuple[float, ...],
        directions: Tuple[bool, ...] = (True, False),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.changepoints = tuple(changepoints)
        self.directions = tuple(directions)

    def sample_hyperparameters(self):
        # Sample changepoint.
        direction = random.choice(self.directions)
        changepoint = random.choice(self.changepoints)

        self.lengthscale_fn = partial(
            gibbs_switching_lengthscale_fn,
            changepoint=changepoint,
            direction=direction,
        )

# Change surface kernel for two kernels -> blurs between both (see https://proceedings.mlr.press/v51/herlands16.pdf [this is two case instance])
class ChangeSurfaceKernel(gpytorch.kernels.Kernel):
    def __init__(
        self,
        k1: gpytorch.kernels.Kernel, # Start Kernel
        k2: gpytorch.kernels.Kernel, # End Kernel
        t0: float, # Change point centre (number of context points at which two kernels are blurred exactly 50-50)
        tau: float, # Scale of change tau -> 0 means instant switch between two (no smoothing)
        eps: float = 1e-6, # Minimum allowed value of tau - prevents div by zero and numerical badness
    ):
        super().__init__()
        self.tau = max(tau, eps)
        self.sigmoid = torch.nn.Sigmoid()
        self.k1 = k1
        self.k2 = k2
        self.t0 = t0

    # Last coordinate of xs is now the context index (time stamp) -> allows it to get correct sample
    def forward(self, x1, x2, diag=False, **params):
        # Unpacks time step (i.e. no context points from actual x data)
        x1_space, t1 = x1[..., :-1], x1[..., -1:]
        x2_space, t2 = x2[..., :-1], x2[..., -1:]

        w1 = self.sigmoid((t1 - self.t0) / self.tau) # [n1, 1]
        w2 = self.sigmoid((t2 - self.t0) / self.tau) # [n2, 2]

        k1_block = self.k1(x1_space, x2_space, diag=diag, **params)
        k2_block = self.k2(x1_space, x2_space, diag=diag, **params)

        # Handles differences for diag vs not - core idea is same
        if diag:
            w1 = w1.squeeze(-1)
            w2 = w2.squeeze(-1)
            mix1 = (1 - w1) * (1 - w2)
            mix2 = w1 * w2
            cov = mix1 * k1_block + mix2 * k2_block
        else:
            mix1 = (1 - w1) * (1 - w2.transpose(-2, -1))
            mix2 = w1 * w2.transpose(-2, -1)
            cov = mix1 * k1_block + mix2 * k2_block 
        return cov

    def sample_hyperparameters(self):
        if hasattr(self.k1, "sample_hyperparameters"):
            self.k1.sample_hyperparameters()
        if hasattr(self.k2, "sample_hyperparameters"):
            self.k2.sample_hyperparameters()
