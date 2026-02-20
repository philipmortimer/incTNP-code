# WISKI GP in NP land - https://arxiv.org/pdf/2103.01454 https://github.com/wjmaddox/online_gp 
import sys
from copy import deepcopy
import math
import numpy as np
from online_gp.models.stems import LinearStem
import torch
import pandas as pd
import gpytorch
from gpytorch import mlls
from online_gp.models import OnlineSKIRegression
from online_gp.models.stems import Identity
from online_gp.utils.cuda import try_cuda
from torch import nn
from check_shapes import check_shapes
from typing import Callable, Optional
import torch.distributions as td
from .incUpdateBase import IncUpdateEff


class WiskiGP(nn.Module):
    def __init__(
        self,
        stem_factory: Optional[Callable[[int], nn.Module]],
        pretrain_lr,
        stream_lr,
        grid_size,
        grid_bound, 
        pretrain,
        update_gp_and_stem,
        covar_module_factory: Optional[Callable[[], gpytorch.kernels.Kernel]],
        init_chunk_size, # Number of points in initial pre training context set
        num_epochs,
        device,
        jitter,
        x_min,
        x_max,
        noise_std,
        no_pretrain_init_prop: float = 0.05, # Percentage of points to include in init when not doing pretraining
    ):
        super().__init__()

        self.stem_factory = stem_factory
        self.pretrain_lr = pretrain_lr
        self.stream_lr = stream_lr
        self.grid_size = grid_size
        self.grid_bound = grid_bound
        self.covar_module_factory = covar_module_factory
        self.init_chunk_size = init_chunk_size
        self.num_epochs = num_epochs
        self.device = device
        self.jitter = jitter
        self.x_min = x_min
        self.x_max = x_max
        self.noise_std = noise_std
        self.pretrain = pretrain
        self.update_gp_and_stem = update_gp_and_stem
        self.gt_pred = None
        self.no_pretrain_init_prop=no_pretrain_init_prop
    
    @check_shapes(
        "xc: [m, nc, dx]", "yc: [m, nc, dy]", "xt: [m, nt, dx]"
    )
    def forward(self, xc, yc, xt) -> torch.distributions.Distribution:
        m, nc, dx = xc.shape
        _, _, dy = yc.shape
        _, nt, _ = xt.shape

        means = []
        vars = []
        # Creates a new WISKI model for each batch seperately - this is expensive and slow
        for batch_i in range(m):
            xc_b, yc_b, xt_b = xc[batch_i], yc[batch_i], xt[batch_i]
            mean_b, var_b = self._data_forward(xc_b, yc_b, xt_b)
            means.append(mean_b)
            vars.append(var_b)
        means_stacked = torch.stack(means, dim=0)
        vars_stacked = torch.stack(vars, dim=0)

        pred_dist = torch.distributions.Normal(means_stacked, (vars_stacked + self.jitter).sqrt()) # May need to watch for numerical stability with the .sqrt()
        return pred_dist

    # Updates the covariance constructor to match the true groundtruth
    def update_covar_params(self, gt_pred):
        assert(isinstance(gt_pred.kernel, gpytorch.kernels.RBFKernel)), "Fixed init only supported for rbf currently"
        self.gt_pred = gt_pred


    @check_shapes(
        "xc: [nc, dx]", "yc: [nc, dy]", "xt: [nt, dx]"
    )
    def _data_forward(self, xc, yc, xt):
        # Shapes
        nc, dx= xc.shape
        _, dy = yc.shape
        nt, _ = xt.shape

        if not self.pretrain: self.init_chunk_size = min(max(int(round(self.no_pretrain_init_prop * nc)), 1), nc - 1)

        # Train for a single example using WISKI
        init_x, init_y, xc_stream, yc_stream, tmean, tstd, xt_norm, x_range = preprocess_data(xc, yc, self.init_chunk_size, xt, self.x_min, self.x_max)
        noise_var_norm = None

        if not self.pretrain:
            # Fixes GP params for rbf kernel
            assert self.gt_pred is not None, "With no pretraining, need to set gt values explicity"
            gt_noise_var = self.gt_pred.likelihood.noise.detach().item()
            gt_ls = self.gt_pred.kernel.lengthscale.detach().item()
            gt_os = 1.0 # base output scale of 1
            noise_var_norm = gt_noise_var / (tstd.item() ** 2) # Sets new noise, updated later down
            #noise_var_norm = None
            new_ls = gt_ls * (2.0 / x_range)
            new_os = gt_os * (1.0 / (tstd**2))
            self.covar_module_factory = lambda: _covar_module_factory_method(lengthscale=new_ls, outputscale=new_os)

        
        if self.noise_std is not None and noise_var_norm is None: noise_var_norm = (self.noise_std / tstd) ** 2

        # Makes WISKI model
        stem = self.stem_factory(input_dim=dx)
        covar_module = self.covar_module_factory()
        wiski_model = OnlineSKIRegression(stem, init_x, init_y, lr=self.pretrain_lr, grid_size=self.grid_size, grid_bound=self.grid_bound, covar_module=covar_module, base_noise_var=noise_var_norm, learn_additional_noise=self.pretrain)
        wiski_model = try_cuda(wiski_model)

        if self.pretrain:
            wiski_model.fit(init_x, init_y, self.num_epochs)  # pretrain model

        self.print_gp_params(wiski_model, self.gt_pred, tstd, x_range, stage="Post-Init")

        # Streamed updates
        wiski_model.set_lr(self.stream_lr)
        for t, (x, y) in enumerate(zip(xc_stream, yc_stream)):
            wiski_model.update(x, y, update_stem=self.update_gp_and_stem, update_gp=self.update_gp_and_stem)
        self.print_gp_params(wiski_model, self.gt_pred, tstd, x_range, stage="Post-Stream")
        # Distribution over targets
        with torch.no_grad():
            pred_mean, pred_var = wiski_model.predict(xt_norm)

        # Unnormalises target distribution
        mean_un = tmean + tstd * pred_mean
        var_un = (tstd ** 2) * pred_var
        
        return mean_un, var_un
    
    # Updates used for exchangeability calc to save doing pretraining everytime
    # Preconditions on the fixed distribution, only uses xt and yt for norm stats as in wiski paper
    def pretrain_on_fixed(self, x_fixed, y_fixed, x_eval, y_eval):
        m, _, dx = x_fixed.shape
        self.pretrained_base_models = []
        self.norm_stats = []
        for i in range(m):
            # Data norm - may want to check this
            xc, yc, xt, yt = x_fixed[i], y_fixed[i], x_eval[i], y_eval[i]
            dataset_size = xc.size(0)
            x_all = torch.cat([xc, xt], dim=0) # Normalises across target and context inputs to ensure valid range - as is done in wiski paper imp
            y_all = torch.cat([yc, yt], dim=0)
            y_all = yc # Normalise just on ctx ys
            x_min, _ = x_all.min(0)
            x_max, _ = x_all.max(0)
            x_range = x_max - x_min
            x_range[x_range == 0] = 1.0
            xc = 2 * ((xc - x_min) / x_range - 0.5)
            xt_norm = 2 * ((xt - x_min) / x_range - 0.5)
            tmean = y_all.mean(0)
            tstd = y_all.std(0)
            if dataset_size <= 1: tstd = 1
            else: tstd = y_all.std(0, unbiased=False)
            tstd[tstd == 0] = 1.0
            yc = (yc - tmean) / tstd
            norm_i = {"tmean": tmean, "tstd": tstd, "x_min": x_min, "x_range": x_range}
            self.norm_stats.append(norm_i)

            # Makes model and pretrains it
            stem = self.stem_factory(input_dim=dx)
            covar_module = self.covar_module_factory()
            wiski_model = OnlineSKIRegression(stem, xc, yc, lr=self.pretrain_lr, grid_size=self.grid_size, grid_bound=self.grid_bound, covar_module=covar_module)
            wiski_model = try_cuda(wiski_model)
            wiski_model.fit(xc, yc, self.num_epochs)
            wiski_model.set_lr(self.stream_lr)
            self.pretrained_base_models.append(wiski_model)

    # Takes in a list of pretrained models (and corresponding normalisation stats) and predicts on the targets at current model state. It then updates the given model in a streaming manner with the target
    def stream_batch_step(self, models_list, norm_statistics, xt, yt):
        m, _, _ = xt.shape
        assert m ==len(models_list), "Model list does not match batching"
        means, vars = [], []
        for i in range(m):
            xt_i, yt_i = xt[i], yt[i]
            stats = norm_statistics[i]
            model = models_list[i]
            xt_norm = 2 * ((xt_i - stats["x_min"]) / stats["x_range"] - 0.5)
            yt_norm = (yt_i - stats["tmean"]) / stats["tstd"]
            with torch.no_grad():
                pred_mean, pred_var = model.predict(xt_norm)
            mean_un = stats["tmean"] + stats["tstd"] * pred_mean
            var_un = (stats["tstd"] ** 2) * pred_var
            means.append(mean_un)
            vars.append(var_un)
            for t, (x, y) in enumerate(zip(xt_norm, yt_norm)):
                model.update(x, y, update_stem=self.update_gp_and_stem, update_gp=self.update_gp_and_stem)
        means_stacked = torch.stack(means, dim=0)
        vars_stacked = torch.stack(vars, dim=0)
        pred_dist = torch.distributions.Normal(means_stacked, (vars_stacked + self.jitter).sqrt())
        return pred_dist
    
    # Incremental updates -  used for tabular streaming
    def init_inc_structs(self, m: int, max_nc: int, device: str, use_flash: bool=False, cache_mhca: bool=False):
        self.pretrain_inc_happened_yet = False
        self.wiski_model_inc = None
        assert self.normalise_data == False, "Data normalisation must be handled outside of wiski when streaming"

    def repeat_ctx(self, repeat_times: int):
        assert False, "Not implemented repeat ctx for wiski and shouldnt need it anyway"

    def update_ctx(self, xc: torch.Tensor, yc: torch.Tensor, use_flash: bool=False, cache_mhca: bool=False):
        if self.pre_pad_dx != None: 
            xc = xc[:, :, :self.pre_pad_dx]
        m, nc, dx= xc.shape
        assert m == 1, "WISKI only supports one batch at a time streamed currently, changes as needed later"
        if self.pretrain_inc_happened_yet == False:
            # Does pretraining step with all provided ctx points
            init_x, init_y = xc[0], yc[0]
            stem = self.stem_factory(input_dim=dx)
            covar_module = self.covar_module_factory()
            self.wiski_model_inc = OnlineSKIRegression(stem, init_x, init_y, lr=self.pretrain_lr, grid_size=self.grid_size, grid_bound=self.grid_bound, covar_module=covar_module)
            self.wiski_model_inc = try_cuda(self.wiski_model_inc)
            self.wiski_model_inc.set_lr(gp_lr=self.pretrain_lr, stem_lr=self.pretrain_lr/10)
            self.wiski_model_inc.fit(init_x, init_y, self.num_epochs)  # pretrain model
            self.wiski_model_inc.set_lr(gp_lr=self.stream_lr, stem_lr=self.stream_lr/10) # Sets new lr
            self.pretrain_inc_happened_yet = True
        else: # Streamed updates
            xc_stream, yc_stream = xc[0], yc[0]
            for t, (x, y) in enumerate(zip(xc_stream, yc_stream)):
                self.wiski_model_inc.update(x, y)
    
    def query(self, xt: torch.Tensor, dy: int, use_flash: bool=False, cache_mhca: bool=False) -> td.Normal:
        assert xt.shape[0] == 1, "WISKI streamed only suppotrs a single batch currently"
        if self.pre_pad_dx != None: 
            xt = xt[:, :, :self.pre_pad_dx]
        xt_targ = xt[0]
        with torch.no_grad():
            pred_mean, pred_var = self.wiski_model_inc.predict(xt_targ)
        pred_std = (pred_var + self.jitter).sqrt()
        return td.Normal(loc=torch.unsqueeze(pred_mean, dim=0), scale=torch.unsqueeze(pred_std, dim=0))   

# Returns the WISKI model for the powerplant dataset based on the configs https://github.com/wjmaddox/online_gp/tree/main as closely as possible. This should only be used on UCI in incremental mode
class PowerplantWiski(WiskiGP):
    def __init__(self):
        super().__init__(
            stem_factory=lambda input_dim: LinearStem(input_dim=input_dim, feature_dim=2),
            pretrain_lr=0.05,
            stream_lr=0.005,
            grid_size=16,
            grid_bound=1, 
            covar_module_factory=lambda: None,# Lets data loader create RBF ARD kernel internally
            init_chunk_size=None,
            num_epochs=200,
            device="cuda",
            jitter=1e-6,
            normalise_data=False,)
        
class SkillcraftWiski(WiskiGP):
    def __init__(self):
        super().__init__(
            stem_factory=lambda input_dim: LinearStem(input_dim=input_dim, feature_dim=2),
            pretrain_lr=0.05,
            stream_lr=0.005,
            grid_size=16,
            grid_bound=1, 
            covar_module_factory=lambda: None,# Lets data loader create RBF ARD kernel internally
            init_chunk_size=None,
            num_epochs=200,
            device="cuda",
            jitter=1e-6,
            normalise_data=False,)
        
class ProteinWiski(WiskiGP):
    def __init__(self):
        super().__init__(
            stem_factory=lambda input_dim: LinearStem(input_dim=input_dim, feature_dim=2),
            pretrain_lr=0.01,
            stream_lr=0.001,
            grid_size=16,
            grid_bound=1, 
            covar_module_factory=lambda: None,# Lets data loader create RBF ARD kernel internally
            init_chunk_size=None,
            num_epochs=200,
            device="cuda",
            jitter=1e-6,
            normalise_data=False,)
        
class ElevatorsWiski(WiskiGP):
    def __init__(self):
        super().__init__(
            stem_factory=lambda input_dim: LinearStem(input_dim=input_dim, feature_dim=2),
            pretrain_lr=0.01,
            stream_lr=0.001,
            grid_size=16,
            grid_bound=1, 
            covar_module_factory=lambda: None,# Lets data loader create RBF ARD kernel internally
            init_chunk_size=None,
            num_epochs=200,
            device="cuda",
            jitter=1e-6,
            normalise_data=False,)
        
class GenericWiski(WiskiGP):
    def __init__(self):
        super().__init__(
            stem_factory=lambda input_dim: LinearStem(input_dim=input_dim, feature_dim=2),
            pretrain_lr=0.01,
            stream_lr=0.001,
            grid_size=16,
            grid_bound=1, 
            covar_module_factory=lambda: None,# Lets data loader create RBF ARD kernel internally
            init_chunk_size=None,
            num_epochs=200,
            device="cuda",
            jitter=1e-6,
            normalise_data=False,)
    

    
@check_shapes(
    "xc: [nc, dx]", "yc: [nc, dy]"
)
def preprocess_data(xc, yc, num_init, xt, x_min, x_max):
    dataset_size = xc.size(0)
    x_all = torch.cat([xc, xt], dim=0) # Normalises across target and context inputs to ensure valid range - as is done in wiski paper imp
    
    x_min, _ = x_all.min(0)
    x_max, _ = x_all.max(0)
    x_range = x_max - x_min
    #x_range[x_range == 0] = 1.0
    xc = 2 * ((xc - x_min) / x_range - 0.5)
    xt_norm = 2 * ((xt - x_min) / x_range - 0.5)

    tmean = yc.mean(0)
    tstd = yc.std(0)
    if dataset_size <= 1:
        tstd = 1
    tstd = yc.std(0, unbiased=False)
    tstd[tstd == 0] = 1.0

    yc = (yc - tmean) / tstd

    init_x, xc = xc[:num_init], xc[num_init:]
    init_y, yc = yc[:num_init], yc[num_init:]
    
    return init_x, init_y, xc, yc, tmean, tstd, xt_norm, x_range

def _covar_module_factory_method(lengthscale=0.25, outputscale=1.0): 
    base = gpytorch.kernels.RBFKernel( ard_num_dims=1, )
    base.lengthscale = lengthscale
    cov = gpytorch.kernels.ScaleKernel(base,) 
    cov.outputscale = outputscale
    return cov
# WISKI GP model with sensible defaults to allow us to focus on the params we care for really. 
# Only designed for exchangeability calculations with RBF kernel.
class ExchangeCalcWiskiRBF(WiskiGP):
    def __init__(
        self,
        grid_size: int, # Number of inducing points
        init_chunk_size: int, # Number of points in initial pre training context set
        stem_factory: Optional[Callable[[int], nn.Module]] = lambda input_dim: Identity(input_dim=input_dim), # Identity stem
        pretrain: bool = True,
        update_gp_and_stem: bool = True,
        pretrain_lr: float = 5e-2,
        stream_lr: float = 5e-3,
        grid_bound: float = 1.0, # Set to 1 as in WiskiGP x inputs scaled to [-1, 1]
        covar_module_factory: Optional[Callable[[], gpytorch.kernels.Kernel]] = _covar_module_factory_method,
        num_epochs: int = 200,
        device: str = "cuda",
        jitter: float = 1e-6, # Small jitter (added for safe sqrt)
        x_min: float = -2.0, # Min and max to make specifying priors easier - note this assume context and target range are the same and known. Can replace easily with per episode calcs but just neater
        x_max: float = 2.0,
        noise_std: Optional[float] = None,
    ):
        super().__init__(stem_factory=stem_factory, pretrain_lr=pretrain_lr, stream_lr=stream_lr,
            grid_size=grid_size, grid_bound=grid_bound, covar_module_factory=covar_module_factory,
        init_chunk_size=init_chunk_size, num_epochs=num_epochs, device=device, jitter=jitter,
        x_min=x_min, x_max=x_max, noise_std=noise_std, pretrain=pretrain, update_gp_and_stem=update_gp_and_stem)


if __name__ == "__main__":
    print("testing")
    grid_size = 16
    init_chunk_size = 10
    wisk_mod = ExchangeCalcWiskiRBF(grid_size=grid_size, init_chunk_size=init_chunk_size)