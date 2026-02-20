import random
from abc import ABC
from typing import Dict, Iterable, Optional, Tuple, Union, List, Callable

import einops
import gpytorch
import torch

from ..networks.gp import RandomHyperparameterKernel, ChangeSurfaceKernel
from .base import GroundTruthPredictor
from .synthetic import SyntheticGeneratorUniformInput


class GPRegressionModel(gpytorch.models.ExactGP):
    def __init__(
        self,
        likelihood: gpytorch.likelihoods.GaussianLikelihood,
        kernel: gpytorch.kernels.Kernel,
        train_inputs: Optional[torch.Tensor] = None,
        train_targets: Optional[torch.Tensor] = None,
    ):
        super().__init__(
            train_inputs=train_inputs,
            train_targets=train_targets,
            likelihood=likelihood,
        )
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = kernel

    def forward(  # pylint: disable=arguments-differ
        self, x: torch.Tensor
    ) -> gpytorch.distributions.MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GPGroundTruthPredictor(GroundTruthPredictor):
    def __init__(
        self,
        kernel: gpytorch.kernels.Kernel,
        likelihood: gpytorch.likelihoods.GaussianLikelihood,
    ):
        self.kernel = kernel
        self.likelihood = likelihood

        self._result_cache: Optional[Dict[str, torch.Tensor]] = None
        self._joint_result_cache: Optional[Dict[str, torch.Tensor]] = None

    def __call__(
        self,
        xc: torch.Tensor,
        yc: torch.Tensor,
        xt: torch.Tensor,
        yt: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:

        # Move devices.
        old_device = xc.device
        device = self._get_kernel_device() # Gets kernel device in a way that also works with gpytorch 1.8.1 which doesnt support kernel.device
        #device = self.kernel.device
        xc = xc.to(device)
        yc = yc.to(device)
        xt = xt.to(device)
        if yt is not None:
            yt = yt.to(device)

        if yt is not None and self._result_cache is not None:
            # Return cached results.
            return (
                self._result_cache["mean"],
                self._result_cache["std"],
                self._result_cache["gt_loglik"],
            )

        mean_list = []
        std_list = []
        gt_loglik_list = []

        # Compute posterior.
        for i, (xc_, yc_, xt_) in enumerate(zip(xc, yc, xt)):
            gp_model = GPRegressionModel(
                likelihood=self.likelihood,
                kernel=self.kernel,
                train_inputs=xc_,
                train_targets=yc_[..., 0],
            )
            gp_model = gp_model.to(xc_.device)
            gp_model.eval()
            gp_model.likelihood.eval()
            with torch.no_grad():

                dist = gp_model(xt_)
                pred_dist = gp_model.likelihood.marginal(dist)
                if yt is not None:
                    gt_loglik = pred_dist.to_data_independent_dist().log_prob(
                        yt[i, ..., 0]
                    )
                    gt_loglik_list.append(gt_loglik)

                mean_list.append(pred_dist.mean)
                try:
                    std_list.append(pred_dist.stddev)
                except RuntimeError:
                    std_list.append(pred_dist.covariance_matrix.diagonal() ** 0.5)

        mean = torch.stack(mean_list, dim=0)
        std = torch.stack(std_list, dim=0)
        gt_loglik = torch.stack(gt_loglik_list, dim=0) if gt_loglik_list else None

        # Cache for deterministic validation batches.
        # Note yt is not specified when passing x_plot.
        if yt is not None:
            self._result_cache = {
                "mean": mean,
                "std": std,
                "gt_loglik": gt_loglik,
            }

        # Move back.
        xc = xc.to(old_device)
        yc = yc.to(old_device)
        xt = xt.to(old_device)
        if yt is not None:
            yt = yt.to(old_device)

        mean = mean.to(old_device)
        std = std.to(old_device)
        if gt_loglik is not None:
            gt_loglik = gt_loglik.to(old_device)

        return mean, std, gt_loglik
    
    # Gets the joint ll conditioing on fix set
    def get_joint_loglik_fixed_context(
        self,
        x_fix: torch.Tensor, # [m, n_fix, dx]
        y_fix: torch.Tensor, # [m, n_fx, dy]
        x_eval: torch.Tensor, # [m, n_eval, dx]
        y_eval: torch.Tensor, # [m, n_eval, dy]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        old_device = x_eval.device
        device = self._get_kernel_device() # Gets kernel device in a way that also works with gpytorch 1.8.1 which doesnt support kernel.device
        x_fix = x_fix.to(device)
        y_fix = y_fix.to(device)
        x_eval = x_eval.to(device)
        y_eval = y_eval.to(device)

        if self._joint_result_cache is not None:
            return (
                self._joint_result_cache["mean"],
                self._joint_result_cache["std"],
                self._joint_result_cache["gt_loglik"],
            )

        mean_list = []
        std_list = []
        gt_loglik_list = []
        for i, (xc_, yc_, xt_, yt_) in enumerate(zip(x_fix, y_fix, x_eval, y_eval)):
            gp_model = GPRegressionModel(
                likelihood=self.likelihood,
                kernel=self.kernel,
                train_inputs=xc_,
                train_targets=yc_[..., 0],
            )
            gp_model = gp_model.to(xc_.device)
            gp_model.eval()
            gp_model.likelihood.eval()
            with torch.no_grad():

                dist = gp_model(xt_)
                pred_dist = gp_model.likelihood.marginal(dist)
                gt_loglik = pred_dist.log_prob(yt_[..., 0])
                gt_loglik_list.append(gt_loglik)

                mean_list.append(pred_dist.mean)
                try:
                    std_list.append(pred_dist.stddev)
                except RuntimeError:
                    std_list.append(pred_dist.covariance_matrix.diagonal() ** 0.5)
        mean = torch.stack(mean_list, dim=0)
        std = torch.stack(std_list, dim=0)
        gt_loglik = torch.stack(gt_loglik_list, dim=0)

        self._joint_result_cache = {
            "mean": mean,
            "std": std,
            "gt_loglik": gt_loglik,
        }
        x_fix=x_fix.to(old_device)
        y_fix=y_fix.to(old_device)
        x_eval=x_eval.to(old_device)
        y_eval=y_eval.to(old_device)
        mean = mean.to(old_device)
        std = std.to(old_device)
        gt_loglik = gt_loglik.to(old_device)

        return mean, std, gt_loglik

            
    
    # Gets kernel device in a way that also works with gpytorch 1.8.1 which doesnt support kernel.device
    def _get_kernel_device(self):
        # Works with new Gpytroch versions (i.e. what is used in the tnp codebase by default)
        if hasattr(self.kernel, "device"): return self.kernel.device

        # Looks at params to infer device
        for p in self.kernel.parameters():
            return p.device
        
        raise ValueError("Could not infer kernel type check gpytorch versions")

    def sample_outputs(
        self, x: torch.Tensor, sample_shape: torch.Size = torch.Size()
    ) -> torch.Tensor:

        gp_model = GPRegressionModel(
            likelihood=self.likelihood,
            kernel=self.kernel,
        )
        gp_model.eval()
        gp_model.likelihood.eval()

        # Sample from prior.
        with torch.no_grad():
            dist = gp_model.forward(x)
            f = dist.sample(sample_shape=sample_shape)
            dist = gp_model.likelihood(f)
            y = dist.sample()
            return y[..., None]


class GPGenerator(ABC):
    def __init__(
        self,
        *,
        kernel: Union[
            RandomHyperparameterKernel,
            Tuple[RandomHyperparameterKernel, ...],
        ],
        noise_std: float,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.kernel = kernel
        if isinstance(self.kernel, Iterable):
            self.kernel = tuple(self.kernel)

        self.noise_std = noise_std

    def set_up_gp(self) -> GPGroundTruthPredictor:
        if isinstance(self.kernel, tuple):
            kernel = random.choice(self.kernel)
        else:
            kernel = self.kernel

        kernel = kernel()
        kernel.sample_hyperparameters()

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        likelihood.noise = self.noise_std**2.0

        return GPGroundTruthPredictor(kernel=kernel, likelihood=likelihood)

    def sample_outputs(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, GroundTruthPredictor]:
        gt_pred = self.set_up_gp()
        y = gt_pred.sample_outputs(x)
        return y, gt_pred


class RandomScaleGPGenerator(GPGenerator, SyntheticGeneratorUniformInput):
    pass


# Takes a list of GT preds for within a batch and combines them
class MixedBatchGroundTruthPredictor(GroundTruthPredictor):
    def __init__(
        self,
        gt_predictors: List[GroundTruthPredictor],
    ):
        self.gt_predictors = gt_predictors

    def __call__(
        self,
        xc: torch.Tensor,
        yc: torch.Tensor,
        xt: torch.Tensor,
        yt: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        m = xc.shape[0]
        assert m <= len(self.gt_predictors), "Invalid predicted batch shape for number of predictors"
        # Iterates over the first m tasks and assume that they are ordered to correspond to the correct GT predictor - plot code is a bit hacky so we are inheriting tech debt here a little bit. check_shapes cleaner here
        mean_list = []
        std_list = []
        gt_loglik_list = []
        for i in range(m):
            mean, std, ll = self.gt_predictors[i](
                xc=xc[i : i + 1],
                yc=yc[i : i + 1],
                xt=xt[i : i + 1],
                yt=None if yt is None else yt[i : i + 1],
            )
            mean_list.append(mean)
            std_list.append(std)
            if ll is not None: gt_loglik_list.append(ll)
        mean = torch.cat(mean_list, 0)
        std = torch.cat(std_list, 0)
        ll = torch.cat(gt_loglik_list, 0) if gt_loglik_list != [] else None
        return mean, std, ll


# Used for combined kernels where each batch randomly samples kernel per point (so a batch may have samples from multiple kernels)
class MixedBatchKernelGPGenerator(GPGenerator, SyntheticGeneratorUniformInput):
    def sample_outputs(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, GroundTruthPredictor]:
        # x shape is [m, n, dx]
        m, n, _ = x.shape
        y = None
        gt_preds = [None] * m
        for i in range(m):
            gt_pred = self.set_up_gp() # Sets up a new GP with a randomly chosen kernel of the available ones
            x_i = x[i, :, :].unsqueeze(0) # Equivalent to an input of batchsize 1 [1, n, dx]
            y_i = gt_pred.sample_outputs(x_i) # [1, n, dy]
            gt_preds[i] = gt_pred
            
            # Allocates y
            if y is None:
                _, _, dy = y_i.shape
                y = torch.empty((m, n, dy), device=y_i.device, dtype=y_i.dtype)
            y[i, :, :] = y_i.squeeze(0)
        
        return y, MixedBatchGroundTruthPredictor(gt_preds)


class RandomScaleGPGeneratorSameInputs(RandomScaleGPGenerator):

    def sample_inputs(
        self,
        nc: int,
        batch_shape: torch.Size,
        nt: Optional[int] = None,
    ) -> torch.Tensor:
        x = super().sample_inputs(nc=nc, batch_shape=torch.Size(), nt=nt)
        x = einops.repeat(x, "n d -> b n d", b=batch_shape[0])
        return x

    def sample_outputs(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        gt_pred = self.set_up_gp()
        sample_shape = x.shape[:-2]
        return gt_pred.sample_outputs(x[0], sample_shape=sample_shape), gt_pred


# GP Generator for a change surface kernel - to see the distribution shift over time
class ChangeKernelGPGenerator(GPGenerator, SyntheticGeneratorUniformInput):
    def __init__(
        self,
        *,
        kernels: Tuple[RandomHyperparameterKernel, ...], # Kernel factories
        noise_std: float,
        max_nc: int,
        t0: int,
        tau: float,
        **kwargs,
    ):
        super().__init__(kernel=kernels, noise_std=noise_std, max_nc=max_nc, **kwargs)
        self.max_nc = max_nc
        self.t0 = t0
        self.tau = tau

    # Helper to randomly sample kernels. In case of just rbf will sample two rbf factories.
    # In case of combined may sample any two combos of the five.
    @staticmethod
    def _draw_two_kernel_factories(
        factories: Tuple[Callable[[], gpytorch.kernels.Kernel], ...]
    ) -> Tuple[Callable[[], gpytorch.kernels.Kernel],
            Callable[[], gpytorch.kernels.Kernel]]:
        if len(factories) == 1:
            return factories[0], factories[0]
        return random.sample(factories, 2)

    # Intialises the GP with the change kernel
    def set_up_gp(self) -> GPGroundTruthPredictor:
        f1, f2 = self._draw_two_kernel_factories(self.kernel)
        k1 = f1()
        k1.sample_hyperparameters()
        k2 = f2()
        k2.sample_hyperparameters()

        change_kern = ChangeSurfaceKernel(k1=k1, k2=k2, t0=self.t0, tau=self.tau)

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        likelihood.noise = self.noise_std ** 2

        return GPGroundTruthPredictor(kernel=change_kern, likelihood=likelihood)

    # Samples inputs and appends time stamp (number of context points) to control blurring of two kernels
    def sample_inputs(
        self,
        nc: int,
        batch_shape: torch.Size,
        nt: int,
    ) -> torch.Tensor:
        x_space = super().sample_inputs(nc=nc, batch_shape=batch_shape, nt=nt) # The actual x
        m, n, dx = x_space.shape
        t_index = torch.arange(n, dtype=x_space.dtype, device=x_space.device)
        t_index = einops.repeat(t_index, "n -> m n 1", m=m)
        x = torch.cat([x_space, t_index], dim=-1) # [m, n, dx + 1]
        return x