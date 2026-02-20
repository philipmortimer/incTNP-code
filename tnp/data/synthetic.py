import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch

from .base import Batch, DataGenerator, GroundTruthPredictor


@dataclass
class SyntheticBatch(Batch):
    gt_mean: Optional[torch.Tensor] = None
    gt_std: Optional[torch.Tensor] = None
    gt_loglik: Optional[torch.Tensor] = None
    gt_pred: Optional[GroundTruthPredictor] = None


class SyntheticGenerator(DataGenerator, ABC):
    def __init__(
        self,
        *,
        dim: int,
        min_nc: int,
        max_nc: int,
        min_nt: int,
        max_nt: int,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Set synthetic generator parameters
        self.dim = dim
        self.min_nc = torch.as_tensor(min_nc, dtype=torch.int)
        self.max_nc = torch.as_tensor(max_nc, dtype=torch.int)
        self.min_nt = torch.as_tensor(min_nt, dtype=torch.int)
        self.max_nt = torch.as_tensor(max_nt, dtype=torch.int)

    def generate_batch(self) -> Batch:
        """Generate batch of data.

        Returns:
            batch: Tuple of tensors containing the context and target data.
        """

        # Sample number of context and target points.
        nc = torch.randint(low=self.min_nc, high=self.max_nc + 1, size=())
        nt = torch.randint(low=self.min_nt, high=self.max_nt + 1, size=())

        # Sample entire batch (context and target points).
        batch = self.sample_batch(
            nc=nc,
            nt=nt,
            batch_shape=torch.Size((self.batch_size,)),
        )

        return batch

    def sample_batch(
        self,
        nc: int,
        nt: int,
        batch_shape: torch.Size,
    ) -> SyntheticBatch:
        # Sample inputs, then outputs given inputs
        x = self.sample_inputs(nc=nc, nt=nt, batch_shape=batch_shape)
        y, gt_pred = self.sample_outputs(x=x)

        xc = x[:, :nc, :]
        yc = y[:, :nc, :]
        xt = x[:, nc:, :]
        yt = y[:, nc:, :]

        return SyntheticBatch(
            x=x,
            y=y,
            xc=xc,
            yc=yc,
            xt=xt,
            yt=yt,
            gt_pred=gt_pred,
        )

    @abstractmethod
    def sample_inputs(
        self,
        nc: int,
        batch_shape: torch.Size,
        nt: Optional[int] = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def sample_outputs(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[GroundTruthPredictor]]:
        """Sample context and target outputs, given the inputs `x`.

        Arguments:
            x: Tensor of shape (batch_size, nc + nt, dim) containing
                the context and target inputs.

        Returns:
            y: Tensor of shape (batch_size, nc + nt, 1) containing
                the context and target outputs.
        """


class SyntheticGeneratorUniformInput(SyntheticGenerator):
    def __init__(
        self,
        *,
        context_range: Tuple[Tuple[float, float], ...],
        target_range: Tuple[Tuple[float, float], ...],
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.context_range = torch.as_tensor(context_range, dtype=torch.float)
        self.target_range = torch.as_tensor(target_range, dtype=torch.float)

    def sample_inputs(
        self,
        nc: int,
        batch_shape: torch.Size,
        nt: Optional[int] = None,
    ) -> torch.Tensor:
        xc = (
            torch.rand((*batch_shape, nc, self.dim))
            * (self.context_range[:, 1] - self.context_range[:, 0])
            + self.context_range[:, 0]
        )

        if nt is not None:
            xt = (
                torch.rand((*batch_shape, nt, self.dim))
                * (self.target_range[:, 1] - self.target_range[:, 0])
                + self.target_range[:, 0]
            )

            return torch.concat([xc, xt], axis=1)

        return xc


class SyntheticGeneratorUniformInputRandomOffset(SyntheticGeneratorUniformInput):
    def __init__(
        self,
        *,
        offset_range: Tuple[Tuple[float, float], ...],
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.offset_range = torch.as_tensor(offset_range, dtype=torch.float)

    def sample_inputs(
        self,
        nc: int,
        batch_shape: torch.Size,
        nt: Optional[int] = None,
    ) -> torch.Tensor:
        x = super().sample_inputs(nc, batch_shape, nt)

        # Apply offset.
        offset = (
            torch.rand(*batch_shape, 1, self.dim)
            * (self.offset_range[:, 1] - self.offset_range[:, 0])
            + self.offset_range[:, 0]
        )
        x = x + offset
        return x


class SyntheticGeneratorBimodalInput(SyntheticGenerator):
    def __init__(
        self,
        *,
        context_range: Tuple[Tuple[Tuple[float, float], Tuple[float, float]], ...],
        target_range: Tuple[Tuple[Tuple[float, float], Tuple[float, float]], ...],
        mode_offset_range: Tuple[Tuple[float, float], ...] = ((0.0, 0.0),),
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.context_range = torch.as_tensor(context_range, dtype=torch.float)
        self.target_range = torch.as_tensor(target_range, dtype=torch.float)
        self.mode_offset_range = torch.as_tensor(mode_offset_range, dtype=torch.float)

    def sample_inputs(
        self,
        nc: int,
        batch_shape: torch.Size,
        nt: Optional[int] = None,
    ) -> torch.Tensor:

        # Sample the mode.
        ctx_bernoulli_probs = torch.empty(*batch_shape, nc).fill_(0.5)
        ctx_modes = torch.bernoulli(ctx_bernoulli_probs).int()

        # Apply offset to the range of each mode.
        mode_offset = (
            torch.rand((self.dim, 2))
            * (self.mode_offset_range[..., 1] - self.mode_offset_range[..., 0])
            + self.mode_offset_range[..., 0]
        )
        context_range = self.context_range + mode_offset[..., None]
        target_range = self.target_range + mode_offset[..., None]

        xc = torch.rand((*batch_shape, nc, self.dim)) * (
            context_range[..., ctx_modes, 1].permute(1, 2, 0)
            - context_range[..., ctx_modes, 0].permute(1, 2, 0)
        ) + context_range[..., ctx_modes, 0].permute(1, 2, 0)

        if nt is not None:
            # Sample the mode.
            trg_bernoulli_probs = torch.empty(*batch_shape, nt).fill_(0.5)
            trg_modes = torch.bernoulli(trg_bernoulli_probs).int()
            xt = torch.rand((*batch_shape, nt, self.dim)) * (
                target_range[..., trg_modes, 1].permute(1, 2, 0)
                - target_range[..., trg_modes, 0].permute(1, 2, 0)
            ) + target_range[..., trg_modes, 0].permute(1, 2, 0)

            return torch.concat([xc, xt], axis=1)

        return xc


class SyntheticGeneratorMixture(SyntheticGenerator):
    def __init__(
        self,
        *,
        generators: Tuple[SyntheticGenerator, ...],
        mixture_probs: Tuple[float, ...],
        mix_samples: bool = False,
        **kwargs,
    ):
        assert len(generators) == len(
            mixture_probs
        ), "Must be a mixture prob for each generator."
        assert sum(mixture_probs) == 1, "Sum of mixture_probs must be 1."
        assert all(
            prob > 0 for prob in mixture_probs
        ), "All elements of mixture_probs must be positive."

        super().__init__(**kwargs)

        # Whether or not to sample mixture for each sample in batch.
        self.mix_samples = mix_samples
        self.generators = generators
        self.mixture_probs = mixture_probs

        # Ensure samples per epoch of generators are infinite, so does not stop sampling.
        for generator in self.generators:
            generator.num_batches = np.inf

    def generate_batch(self) -> Batch:
        # Sample generator.
        gen = random.choices(self.generators, weights=self.mixture_probs, k=1)[0]

        # Sample number of context and target points.
        nc = torch.randint(low=self.min_nc, high=self.max_nc + 1, size=())
        nt = torch.randint(low=self.min_nt, high=self.max_nt + 1, size=())

        return gen.sample_batch(
            nc=nc, nt=nt, batch_shape=torch.Size((self.batch_size,))
        )

    def sample_inputs(
        self,
        nc: int,
        batch_shape: torch.Size,
        nt: Optional[int] = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    def sample_outputs(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[GroundTruthPredictor]]:
        raise NotImplementedError
