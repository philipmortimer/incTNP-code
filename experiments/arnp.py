# Autoregressive neural process - test time only.
# Based on https://arxiv.org/pdf/2303.14468 - takes a normal NP model and treats predicted target points as context points
# Inspired by https://github.com/wesselb/neuralprocesses/blob/main/neuralprocesses
from data_temp.data_processing.elevations import get_cached_elevation_grid
from tnp.data.hadISDTemporal import TemporalHadISDDataGenerator
from tnp.models.tnpa import TNPA
import torch
from check_shapes import check_shapes
from torch import nn
from typing import Optional, Union, Literal, Callable, Tuple
from tnp.utils.np_functions import np_pred_fn
from tnp.data.base import Batch
from tnp.models.incUpdateBase import IncUpdateEff, IncUpdateEffFixed
from plot_adversarial_perms import get_model
from tnp.data.gp import RandomScaleGPGenerator
from tnp.networks.gp import MaternKernel, PeriodicKernel, RBFKernel
from torch.nn.attention import SDPBackend, sdpa_kernel
from functools import partial
from tqdm import tqdm
import numpy as np
import torch.distributions as td
from plot import plot
import os
import matplotlib.pyplot as plt
from tnp.utils.data_loading import adjust_num_batches
import matplotlib
import gc
import random

matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"
matplotlib.rcParams["axes.titlesize"]= 14


@check_shapes(
    "xc: [m, nc, dx]", "yc: [m, nc, dy]", "xt: [m, nt, dx]", "yt: [m, nt, dy]",
)
@torch.no_grad
def _shuffle_targets(np_model: nn.Module, xc: torch.Tensor, yc: torch.Tensor, xt: torch.Tensor, yt: Optional[torch.Tensor],
    order: Literal["random", "given", "left-to-right", "variance", "spatiotemporal"]):
    m, nt, dx = xt.shape
    _, _, dy = yc.shape
    device = xt.device
    if order == "given":
        perm = torch.arange(nt, device=device).repeat(m, 1)
        return xt, yt, perm
    elif order == "random":
        perm = torch.rand(m, nt, device=device).argsort(dim=1)
        perm_x = perm.unsqueeze(-1).expand(-1, -1, dx)
        xt_shuffled = torch.gather(xt, 1, perm_x)
        if yt is not None:
            perm_y = perm.unsqueeze(-1).expand(-1, -1, dy)
            yt_shuffled = torch.gather(yt, 1, perm_y)
        else: yt_shuffled = None
        return xt_shuffled, yt_shuffled, perm
    elif order == "left-to-right":
        assert dx == 1, "left-to-right ordering only supported for one dimensional dx"
        perm = torch.argsort(xt.squeeze(-1), dim=1)
        perm_x = perm.unsqueeze(-1).expand(-1, -1, dx)
        xt_sorted = torch.gather(xt, 1, perm_x)
        if yt is not None:
            perm_y = perm.unsqueeze(-1).expand(-1, -1, dy)
            yt_sorted = torch.gather(yt, 1, perm_y)
        else: yt_sorted = None
        return xt_sorted, yt_sorted, perm
    elif order == "variance":
        # Predicts all target points conditioned on context points and orders (highest variance first) - this is obviously much more expensive
        batch = Batch(xc=xc, yc=yc, xt=xt, yt=None, x=None, y=None)
        pred_dist = np_pred_fn(np_model, batch)
        var = pred_dist.variance.mean(-1) # Gets variance (averaged over dy) [m, nt]
        perm = torch.argsort(var, dim=1, descending=True)
        perm_x = perm.unsqueeze(-1).expand(-1, -1, dx)
        xt_sorted = torch.gather(xt, 1, perm_x)
        if yt is not None:
            perm_y = perm.unsqueeze(-1).expand(-1, -1, dy)
            yt_sorted = torch.gather(yt, 1, perm_y)
        else: yt_sorted = None
        return xt_sorted, yt_sorted, perm
    elif order == "time":# For hadISD time ordering
        times = xt[:, :, 3]
        perm = torch.argsort(times, dim=1)
        perm_x = perm.unsqueeze(-1).expand(-1, -1, dx)
        xt_sorted = torch.gather(xt, 1, perm_x)
        if yt is not None:
            perm_y = perm.unsqueeze(-1).expand(-1, -1, dy)
            yt_sorted = torch.gather(yt, 1, perm_y)
        else:
            yt_sorted = None
        return xt_sorted, yt_sorted, perm
    elif order == "spatiotemporal": # Sorts by lat lon and time
        t_coords = xt[:, :, 3]
        lat_coords = xt[:, :, 0]
        lon_coords = xt[:, :, 1]
        perm = torch.argsort(lon_coords, dim=1, stable=True)
        lat_sorted = torch.gather(lat_coords, 1, perm)
        perm = torch.gather(perm, 1, torch.argsort(lat_sorted, dim=1, stable=True))
        t_sorted = torch.gather(t_coords, 1, perm)
        perm = torch.gather(perm, 1, torch.argsort(t_sorted, dim=1, stable=True))
        perm_x = perm.unsqueeze(-1).expand(-1, -1, dx)
        xt_sorted = torch.gather(xt, 1, perm_x)
        if yt is not None:
            perm_y = perm.unsqueeze(-1).expand(-1, -1, dy)
            yt_sorted = torch.gather(yt, 1, perm_y)
        else:
            yt_sorted = None  
        return xt_sorted, yt_sorted, perm



@check_shapes(
    "xc: [m, nc, dx]", "yc: [m, nc, dy]", "xt: [m, nt, dx]", "yt: [m, nt, dy]"
)
@torch.no_grad()
def ar_loglik(np_model: nn.Module, xc: torch.Tensor, yc: torch.Tensor, xt: torch.Tensor, yt: torch.Tensor,
    normalise: bool = True, order: Literal["random", "given", "left-to-right", "variance"] = "random") -> torch.Tensor:
    xt, yt, _ = _shuffle_targets(np_model, xc, yc, xt, yt, order)
    np_model.eval()
    m, nt, dx = xt.shape
    _, nc, dy = yc.shape
    log_probs = torch.zeros((m), device=xt.device)
    squared_errors = torch.zeros((m), device=xt.device, dtype=torch.float64)
    for i in range(nt):
        # Sets context and target
        xt_sel = xt[:,i:i+1,:]
        yt_sel = yt[:,i:i+1,:]
        xc_it = torch.cat((xc, xt[:, :i, :]), dim=1)
        yc_it = torch.cat((yc, yt[:, :i, :]), dim=1)
        batch = Batch(xc=xc_it, yc=yc_it, xt=xt_sel, yt=yt_sel, x=torch.cat((xc_it, xt_sel), dim=1), y=torch.cat((yc_it, yt_sel), dim=1))

        # Prediction + log prob
        pred_dist = np_pred_fn(np_model, batch)
        log_probs += pred_dist.log_prob(yt_sel).sum(dim=(-1, -2))

        squared_errors += (pred_dist.mean - yt_sel).to(squared_errors.dtype).pow(2).sum(dim=(-1, -2))
    if normalise:
        log_probs /= (nt * dy)
    rmse = torch.sqrt(squared_errors / (nt * dy)).to(xt.dtype)
    return log_probs, rmse



@check_shapes(
    "xc: [m, nc, dx]", "yc: [m, nc, dy]", "xt: [m, nt, dx]"
)
@torch.no_grad()
def ar_predict(model, xc: torch.Tensor, yc: torch.Tensor, xt: torch.Tensor,
    order: Literal["random", "given", "left-to-right", "variance"] = "random",
    num_samples: int = 10,
    prioritise_fixed: bool = False, # If incremental updates are available prioritise fixed or true dynamic algorithm
    device: str = "cuda", # Device for computing
    device_ret: str = "cpu", # Return device
    use_flash: bool = False, # Use flash attention if possible? - experimental
    run_mode: Literal["normal", "flash", "normal_cast"] = "normal", # Mode to run call in.
    cast_model: bool = False, # Casts model to correct type (likely want this false)
    cache_mhca: bool = False,
    persist_small: bool = False, # Used to flag a streamed ctx optimisation and means dont condition on initial ctx as that has already happened
    ):
    # Determines context managers
    assert run_mode in ("normal", "flash", "normal_cast"), "run mode must be specified"
    if run_mode == "flash":
        if cast_model: model.to(dtype=torch.bfloat16) # converts models
        with torch.no_grad(), torch.autocast(device_type=device, dtype=torch.bfloat16), torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
            return _ar_predict_internal(model=model, xc=xc, yc=yc, xt=xt,order=order,num_samples=num_samples,prioritise_fixed=prioritise_fixed,device=device,device_ret=device_ret,use_flash=use_flash, cache_mhca=cache_mhca, persist_small=persist_small)
    elif run_mode == "normal":
        if cast_model: model.to(dtype=torch.float32) # Careful not to benchmark this cost by either having the model in this mode 
        with torch.no_grad(): # autocasts here too
            return _ar_predict_internal(model=model, xc=xc, yc=yc, xt=xt,order=order,num_samples=num_samples,prioritise_fixed=prioritise_fixed,device=device,device_ret=device_ret,use_flash=use_flash, cache_mhca=cache_mhca, persist_small=persist_small)
    elif run_mode == "normal_cast":
        if cast_model: model.to(dtype=torch.bfloat16) # converts models
        with torch.no_grad(), torch.autocast(device_type=device, dtype=torch.bfloat16):
            return _ar_predict_internal(model=model, xc=xc, yc=yc, xt=xt,order=order,num_samples=num_samples,prioritise_fixed=prioritise_fixed,device=device,device_ret=device_ret,use_flash=use_flash, cache_mhca=cache_mhca, persist_small=persist_small)

@check_shapes(
    "xc: [m, nc, dx]", "yc: [m, nc, dy]", "xt: [m, nt, dx]"
)
@torch.no_grad()
def _ar_predict_internal(model, xc: torch.Tensor, yc: torch.Tensor, xt: torch.Tensor,
    order: Literal["random", "given", "left-to-right", "variance"] = "random",
    num_samples: int = 10,
    prioritise_fixed: bool = False, # If incremental updates are available prioritise fixed or true dynamic algorithm
    device: str = "cuda", # Device for computing
    device_ret: str = "cpu", # Return device
    use_flash: bool = False, # Use flash attention if possible? - experimental
    cache_mhca: bool = False,
    persist_small: bool = False,
    ):
    model.eval()
    m, nt, dx = xt.shape
    _, nc, dy = yc.shape
    xc, yc, xt = xc.to(device), yc.to(device), xt.to(device)

    xc_stacked = xc.repeat_interleave(num_samples, dim=0)
    yc_stacked = yc.repeat_interleave(num_samples, dim=0)
    xt_stacked = xt.repeat_interleave(num_samples, dim=0)

    xt_stacked, _, perm = _shuffle_targets(model, xc_stacked, yc_stacked, xt_stacked, None, order) # Should I shuffle before or after stacking?

    yt_preds_mean, yt_preds_std = torch.empty((m * num_samples, nt, dy), device=device), torch.empty((m * num_samples, nt, dy), device=device)

    is_fixed_inc_update = isinstance(model, IncUpdateEffFixed)
    is_inc_gen_update = isinstance(model, IncUpdateEff)
    is_fixed_inc_update = (is_fixed_inc_update and prioritise_fixed) or (is_fixed_inc_update and not is_inc_gen_update)
    assert not is_fixed_inc_update, "Removed fixed update support to simplyif code - still works just fine"
    is_inc_gen_update = (is_inc_gen_update and not prioritise_fixed) or (is_inc_gen_update and not is_fixed_inc_update)
    assert is_fixed_inc_update != is_inc_gen_update or (not is_fixed_inc_update and not is_inc_gen_update), "Xor onf fixed vs inc update"
    if is_inc_gen_update:
        if persist_small:
            if nc != 0: model.update_ctx(xc=xc, yc=yc,use_flash=use_flash, cache_mhca=cache_mhca, persist_small=True) # If model fully conditioned, no need to do one more
            model.repeat_ctx(repeat_times=num_samples, persist_small=True)
        else:
            model.init_inc_structs(m=xc.shape[0], max_nc=nc+nt, device=device, use_flash=use_flash, cache_mhca=cache_mhca)
            model.update_ctx(xc=xc, yc=yc,use_flash=use_flash, cache_mhca=cache_mhca)
            model.repeat_ctx(repeat_times=num_samples) # Copies the response across samples
    elif is_fixed_inc_update:
        model.init_inc_structs_fixed(m=xc_stacked.shape[0], max_nc=nc+nt, xt=xt_stacked, dy=dy, device=device,use_flash=use_flash)

    for i in range(nt):
        xt_tmp = xt_stacked[:, i:i+1,:]
        if is_inc_gen_update:
            pred_dist = model.query(xt=xt_tmp, dy=dy,use_flash=use_flash, cache_mhca=cache_mhca)
        elif is_fixed_inc_update:
            pred_dist = model.query_fixed(tgt_start_ind=i, tgt_end_ind=i+1, use_flash=use_flash)
        else:
            if i == 0 and order in ("given" ,"left-to-right"): # Does first prediction with non stacked ctx and simple exapnds it for deterministic orderings
                batch = Batch(xc=xc, yc=yc, xt=xt_stacked[::num_samples, i:i+1, :], yt=None, x=None, y=None)
                pred_dist = np_pred_fn(model, batch)
                pred_mean = pred_dist.mean.repeat_interleave(num_samples, dim=0)
                pred_std = pred_dist.stddev.repeat_interleave(num_samples, dim=0)
                pred_dist = td.Normal(pred_mean, pred_std)
            else:
                batch = Batch(xc=xc_stacked, yc=yc_stacked, xt=xt_tmp, yt=None, x=None, y=None)
                pred_dist = np_pred_fn(model, batch)
        assert isinstance(pred_dist, td.Normal), "Must predict a gaussian"
        pred_mean, pred_std = pred_dist.mean, pred_dist.stddev
        yt_preds_mean[:,i:i+1,:] = pred_mean
        yt_preds_std[:,i:i+1,:] = pred_std
        # Samples from the predictive distribution and updates the context
        if i < nt - 1:
            yt_sampled = pred_dist.sample() # [m * num_samples, 1, dy]
            if is_inc_gen_update:
                model.update_ctx(xc=xt_tmp, yc=yt_sampled, use_flash=use_flash, cache_mhca=cache_mhca)
            elif is_fixed_inc_update:
                model.update_ctx_fixed(xc=xt_tmp, yc=yt_sampled, use_flash=use_flash)
            else:
                xc_stacked = torch.cat((xc_stacked, xt_tmp), dim=1)
                yc_stacked = torch.cat((yc_stacked, yt_sampled), dim=1)
                
    # Unshuffles the target ordering to be in line with what was passed in
    inv_perm = perm.argsort(dim=1)
    idx = inv_perm.unsqueeze(-1).expand(-1, -1, dy)
    yt_preds_mean = yt_preds_mean.gather(dim=1, index=idx)
    yt_preds_std = yt_preds_std.gather(dim=1, index=idx)

    yt_preds_mean = yt_preds_mean.view(num_samples, m, nt, dy)
    yt_preds_std = yt_preds_std.view(num_samples, m, nt, dy)
    # Permutes to [m, nt, dy, num_samples]
    yt_preds_mean = yt_preds_mean.permute(1,2,3,0)
    yt_preds_std = yt_preds_std.permute(1,2,3,0)
    mix = td.Categorical(torch.full((m, nt, dy, num_samples), 1.0 / num_samples, device=device_ret))
    comp = td.Normal(yt_preds_mean.to(device_ret), yt_preds_std.to(device_ret))
    approx_dist = td.MixtureSameFamily(mix, comp)

    # For sample draws return raw samples and run through model again for smooth samples (see paper / code)
    return approx_dist



# -------------------------------------------------------------------------------------------------------




# Plots a handful of kernels
def plot_ar_unrolls():
    # Hypers
    order="given"
    #no_samples = [1, 2, 5, 10, 50, 100, 500, 1000]
    no_samples = [1, 2, 10, 50]
    folder_name = "experiments/plot_results/ar/plots/"
    no_kernels = 5#20
    device="cuda"
    use_flash = True
    # End of hypers
    models = model_list_rbf_nc_64()
    models.append(("", "", "ConvCNP", ""))
    models.append(("", "", "LBANP", ""))
    models.append(("", "", "MCNP", ""))
    data = get_rbf_rangesame_testset()
    for (model_yml, model_wab, model_name, is_local) in models:
        print(model_name)
        model = get_model(model_yml, model_wab, seed=False, device=device, local_weights=is_local)
        #model = get_model(model_yml, model_wab, seed=False, device=device, local_weights=is_local)
        model.eval()
        model_folder = f"{folder_name}/{model_name}"
        if not os.path.isdir(model_folder):
            os.makedirs(model_folder)
        seed = 22
        for sample in no_samples:
            def pred_fn_pred(model, batch, predict_without_yt_tnpa=True):
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                np.random.seed(seed)
                random.seed(seed)
                return ar_predict(model, batch.xc, batch.yc, batch.xt, order, sample, device=device, use_flash=True, run_mode="flash", cast_model=True, opt_flag=True)

            plot(model=model, batches=data, num_fig=min(no_kernels, len(data)), name=model_folder+f"/ns_{sample}_od_{order}_CACHED",
                savefig=True, logging=False, pred_fn=pred_fn_pred, x_range = (-2.0, 2.0),
                model_lbl=f"Flash AR {model_name} (S={sample})")
            
            def pred_fn_pred(model, batch, predict_without_yt_tnpa=True):
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                np.random.seed(seed)
                random.seed(seed)
                return ar_predict(model, batch.xc, batch.yc, batch.xt, order, sample, device=device, use_flash=True, run_mode="flash", cast_model=True, opt_flag=False)

            plot(model=model, batches=data, num_fig=min(no_kernels, len(data)), name=model_folder+f"/ns_{sample}_od_{order}_DEFA",
                savefig=True, logging=False, pred_fn=pred_fn_pred, x_range = (-2.0, 2.0),
                model_lbl=f"Flash AR {model_name} (S={sample})")



def get_rbf_rangesame_testset():
    # RBF Dataset
    min_nc = 1
    max_nc = 64
    nt= 128
    context_range = [[-2.0, 2.0]]
    target_range = [[-2.0, 2.0]]
    samples_per_epoch = 4_096
    batch_size = 16
    noise_std = 0.1
    deterministic = True
    ard_num_dims = 1
    min_log10_lengthscale = -0.602
    max_log10_lengthscale = 0.0
    rbf_kernel_factory = partial(RBFKernel, ard_num_dims=ard_num_dims, min_log10_lengthscale=min_log10_lengthscale,
                         max_log10_lengthscale=max_log10_lengthscale)
    kernels = [rbf_kernel_factory]
    gen_test = RandomScaleGPGenerator(dim=1, min_nc=min_nc, max_nc=max_nc, min_nt=nt, max_nt=nt, batch_size=batch_size,
        context_range=context_range, target_range=target_range, samples_per_epoch=samples_per_epoch, noise_std=noise_std,
        deterministic=deterministic, kernel=kernels)
    data = list(gen_test)
    return data

def get_model_list_rbf_old():
    # List of models to compare
    tnp_plain = ('experiments/configs/synthetic1dRBF/gp_plain_tnp_rangesame.yml',
        'REMOVED', 'TNP-D') # Removed w&b link to preserve anonymity
    models = [tnp_plain]
    return models

def model_list_rbf_nc_64():
    tnp_plain = ('experiments/configs/incTNPCheckpoints/RBF/Nc_64/config_tnpd_rbf_Nc64_epoch_0220.yaml',
        'experiments/configs/incTNPCheckpoints/RBF/Nc_64/tnpd_rbf_Nc64_epoch_0220.ckpt', 'TNP-D', True)
    incTNP = ('experiments/configs/incTNPCheckpoints/RBF/Nc_64/config_inctnp_rbf_Nc64.yaml',
        'experiments/configs/incTNPCheckpoints/RBF/Nc_64/inctnp_rbf_Nc64_epoch_0199.ckpt', 'incTNP', True)
    batchedTNP = ('experiments/configs/incTNPCheckpoints/RBF/Nc_64/config_inctnpb_rbf_Nc64.yaml',
        'experiments/configs/incTNPCheckpoints/RBF/Nc_64/inctnpb_rbf_Nc64_epoch_0199.ckpt', 'incTNP-Batched', True)
    mcnp = ('experiments/configs/incTNPCheckpoints/RBF/Nc_64/config_cnp_rbf_nc64.yaml',
            'experiments/configs/incTNPCheckpoints/RBF/Nc_64/cnp_rbf_nc64_E220.ckpt', 'MCNP', True)
    convcnp = ('experiments/configs/incTNPCheckpoints/RBF/Nc_64/config_convcnp_rbf_nc64.yaml',
               'experiments/configs/incTNPCheckpoints/RBF/Nc_64/convcnp_rbf_nc64_E220.ckpt', 'ConvCNP', True)
    lbanp = ('experiments/configs/incTNPCheckpoints/RBF/Nc_64/config_lbanp_rbf_nc64_E220.yaml',
             'experiments/configs/incTNPCheckpoints/RBF/Nc_64/lbanp_rbf_nc64_E220.ckpt', 'LBANP', True)
    tnpa = ('experiments/configs/incTNPCheckpoints/RBF/Nc_64/config_tnpa_rbf_nc64_E220.yaml',
            'experiments/configs/incTNPCheckpoints/RBF/Nc_64/tnpa_rbf_nc64_E220.ckpt', 'TNP-A', True)
    models = [tnp_plain, incTNP, batchedTNP, mcnp, convcnp, lbanp, tnpa]
    return models

def model_list_rbf_nc_512():
    tnp_plain = ('experiments/configs/incTNPCheckpoints/RBF/Nc_512/config_tnpd_rbf_Nc512_epoch_0249.yaml',
        'experiments/configs/incTNPCheckpoints/RBF/Nc_512/tnpd_rbf_Nc512_epoch_0249.ckpt', 'TNP-D', True)
    incTNP = ('experiments/configs/incTNPCheckpoints/RBF/Nc_512/config_incTNP_rbf_Nc512_epoch_0249.yaml',
        'experiments/configs/incTNPCheckpoints/RBF/Nc_512/incTNP_rbf_Nc512_epoch_0249.ckpt', 'incTNP', True)
    batchedTNP = ('experiments/configs/incTNPCheckpoints/RBF/Nc_512/config_incTNPb_rbf_Nc512_epoch_0199.yaml',
        'experiments/configs/incTNPCheckpoints/RBF/Nc_512/inctnpb_rbf_Nc512_epoch_0199.ckpt', 'incTNP-Batched', True)
    models = [tnp_plain, incTNP, batchedTNP]
    return models

def model_list_combined_nc_64():
    tnp_plain = ('experiments/configs/incTNPCheckpoints/Combined/Nc_64/config_tnpd_combined_Nc64_epoch_0439.yaml',
        'experiments/configs/incTNPCheckpoints/Combined/Nc_64/tnpd_combined_Nc64_epoch_0439.ckpt', 'TNP-D', True)
    incTNP = ('experiments/configs/incTNPCheckpoints/Combined/Nc_64/config_inctnp_combined_Nc64_epoch_0439.yaml',
        'experiments/configs/incTNPCheckpoints/Combined/Nc_64/inctnp_combined_Nc64_epoch_0439.ckpt', 'incTNP', True)
    batchedTNP = ('experiments/configs/incTNPCheckpoints/Combined/Nc_64/config_inctnpb_combined_Nc64_epoch_0399.yaml',
        'experiments/configs/incTNPCheckpoints/Combined/Nc_64/inctnpb_combined_Nc64_epoch_0399.ckpt', 'incTNP-Batched', True)
    models = [tnp_plain, incTNP, batchedTNP]
    return models

def model_list_combined_nc_512():
    tnp_plain = ('experiments/configs/incTNPCheckpoints/Combined/Nc_512/config_tnpd_combined_Nc512_epoch_0499.yaml',
        'experiments/configs/incTNPCheckpoints/Combined/Nc_512/tnpd_combined_Nc512_epoch_0499.ckpt', 'TNP-D', True)
    incTNP = ('experiments/configs/incTNPCheckpoints/Combined/Nc_512/config_incTNP_combined_Nc512_epoch_0499.yaml',
        'experiments/configs/incTNPCheckpoints/Combined/Nc_512/inctnp_combined_Nc512_epoch_0499.ckpt', 'incTNP', True)
    batchedTNP = ('experiments/configs/incTNPCheckpoints/Combined/Nc_512/config_inctnpb_combined_Nc512_epoch_0399.yaml',
        'experiments/configs/incTNPCheckpoints/Combined/Nc_512/inctnpb_combined_Nc512_epoch_0399.ckpt', 'incTNP-Batched', True)
    models = [tnp_plain, incTNP, batchedTNP]
    return models

# Temporal list to use for timing code
def model_list_hadtemporal_timings():
    models = []
    return models

def get_hadisd_temporal_data(samples_per_epoch):
    dem_path = "REMOVED"
    cache_dem_dir = "REMOVED"
    data_root = "REMOVED"
    num_grid_points_plot = 100
    # Normal hypers
    N_c_min = 100
    N_c_max = 2100
    N_t_min = 250
    N_t_max = 250
    split ="test"
    batch_size = 32
    num_val_workers = 1
    # Change these depending on the training / eval dist desired
    delta_hours = 6
    h_window = 8

    seed = 42 # Seed to make gen reproducabkle between runs
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Loads had dataset
    gen_test = TemporalHadISDDataGenerator(N_c_min=N_c_min, N_c_max=N_c_max, N_t_min=N_t_min, N_t_max=N_t_max,
        split=split, samples_per_epoch=samples_per_epoch, batch_size=batch_size, data_root=data_root,
        ordering="ctx_time", delta_hours=delta_hours, h_window=h_window)
    
    # Wraps data set in a proper torch set loader for less IO bottlenecking
    test_loader = torch.utils.data.DataLoader(
       gen_test,
        batch_size=None,
        num_workers=num_val_workers,
        worker_init_fn=(
            (
                adjust_num_batches
            )
            if num_val_workers > 0
            else None
        ),
        persistent_workers=False,
        pin_memory=True,
    )

    data = list(test_loader) # Returns list to ensure runtime mearued on same nc and nt
    return data


def get_rbf_data(samples_per_epoch, max_nc):
    min_nc = 1
    nt = 128
    num_workers = 1
    context_range = [[-2.0, 2.0]]
    target_range = [[-2.0, 2.0]]
    batch_size = 16
    noise_std = 0.1
    deterministic = True
    ard_num_dims = 1
    min_log10_lengthscale = -0.602
    max_log10_lengthscale = 0.0
    rbf_kernel_factory = partial(RBFKernel, ard_num_dims=ard_num_dims, min_log10_lengthscale=min_log10_lengthscale,
                         max_log10_lengthscale=max_log10_lengthscale)
    kernels = [rbf_kernel_factory]
    gen_test = RandomScaleGPGenerator(dim=1, min_nc=min_nc, max_nc=max_nc, min_nt=nt, max_nt=nt, batch_size=batch_size,
        context_range=context_range, target_range=target_range, samples_per_epoch=samples_per_epoch, noise_std=noise_std,
        deterministic=deterministic, kernel=kernels)
    test_loader = torch.utils.data.DataLoader(gen_test, batch_size=None, num_workers=num_workers, worker_init_fn=(adjust_num_batches if num_workers > 0 else None),persistent_workers=False, pin_memory=True,)
    return test_loader

def get_combined_data(samples_per_epoch, max_nc):
    min_nc = 1
    nt = 128
    num_workers = 1
    context_range = [[-2.0, 2.0]]
    target_range = [[-2.0, 2.0]]
    batch_size = 16
    noise_std = 0.1
    deterministic = True
    ard_num_dims = 1
    min_log10_lengthscale = -0.602
    max_log10_lengthscale = 0.0
    min_log10_period = 0.301
    max_log10_period = 0.301
    rbf_kernel_factory = partial(RBFKernel, ard_num_dims=ard_num_dims, min_log10_lengthscale=min_log10_lengthscale,
                         max_log10_lengthscale=max_log10_lengthscale)
    matern12_kernel_factory = partial(MaternKernel, ard_num_dims=ard_num_dims, min_log10_lengthscale=min_log10_lengthscale,
                         max_log10_lengthscale=max_log10_lengthscale, nu=0.5)
    matern32_kernel_factory = partial(MaternKernel, ard_num_dims=ard_num_dims, min_log10_lengthscale=min_log10_lengthscale,
                         max_log10_lengthscale=max_log10_lengthscale, nu=1.5)
    matern52_kernel_factory = partial(MaternKernel, ard_num_dims=ard_num_dims, min_log10_lengthscale=min_log10_lengthscale,
                         max_log10_lengthscale=max_log10_lengthscale, nu=2.5)
    periodic_kernel_factory = partial(PeriodicKernel, ard_num_dims=ard_num_dims, min_log10_lengthscale=min_log10_lengthscale,
                         max_log10_lengthscale=max_log10_lengthscale, min_log10_period=min_log10_period, max_log10_period=max_log10_period)
    kernels = [rbf_kernel_factory, matern12_kernel_factory, matern32_kernel_factory, matern52_kernel_factory, periodic_kernel_factory]
    gen_test = RandomScaleGPGenerator(dim=1, min_nc=min_nc, max_nc=max_nc, min_nt=nt, max_nt=nt, batch_size=batch_size,
        context_range=context_range, target_range=target_range, samples_per_epoch=samples_per_epoch, noise_std=noise_std,
        deterministic=deterministic, kernel=kernels)
    test_loader = torch.utils.data.DataLoader(gen_test, batch_size=None, num_workers=num_workers, worker_init_fn=(adjust_num_batches if num_workers > 0 else None),persistent_workers=False, pin_memory=True,)
    return test_loader

@torch.no_grad()
def compare_models(models, data, base_out_txt_file: str, eval_ar: bool, device: str = "cuda"):
    # Hypers to select - also look at dataset hypers
    ordering = "random"
    output_file = base_out_txt_file + f'_{ordering}.txt'
    if os.path.exists(output_file):
            print(f"Removing old results file: {output_file}")
            os.remove(output_file)
    # End of hypers
    out_txt = ""
    for (model_yml, model_wab, model_name, is_local) in models:
        ll_list, rmse_list = [], []
        ll_standard_list, rmse_standard_list = [], []
        model = get_model(model_yml, model_wab, seed=False, device=device, local_weights=is_local)
        model.eval()
        for batch in tqdm(data, desc=f'{model_name} eval'):
            # Moves batch to device
            batch.xc, batch.yc = batch.xc.to(device), batch.yc.to(device)
            batch.xt, batch.yt = batch.xt.to(device), batch.yt.to(device)
            batch.x, batch.y = batch.x.to(device), batch.y.to(device)

            # AR ll
            mean_ll, mean_rmse = -1, -1
            if eval_ar and model_name != "TNP-A":
                ll, rmse = ar_loglik(np_model=model, xc=batch.xc.to(device), yc=batch.yc.to(device),
                    xt=batch.xt.to(device), yt=batch.yt.to(device), normalise=True, order=ordering)
                mean_ll = torch.mean(ll).item() # Goes from [m] to a float
                mean_rmse = torch.mean(rmse).item()
            ll_list.append(mean_ll)
            rmse_list.append(mean_rmse)

            # Standard LL and RMSE
            with torch.no_grad():
                pred_dist = np_pred_fn(model, batch)
            loglik_temp_standard = pred_dist.log_prob(batch.yt).sum() / batch.yt[..., 0].numel()
            rmse_standard = nn.functional.mse_loss(pred_dist.mean, batch.yt).sqrt().cpu().mean()
            ll_standard_list.append(torch.mean(loglik_temp_standard).item())
            rmse_standard_list.append(torch.mean(rmse_standard).item())
        ll_average = np.mean(ll_list)
        ll_standard = np.std(ll_list, ddof=1) / np.sqrt(len(ll_list))
        rmse_average = np.mean(rmse_list)
        rmse_standard = np.std(rmse_list, ddof=1) / np.sqrt(len(rmse_list))

        ll_standard_average = np.mean(ll_standard_list)
        ll_standard_std = np.std(ll_standard_list, ddof=1) / np.sqrt(len(ll_standard_list))
        rmse_standard_average = np.mean(rmse_standard_list)
        rmse_standard_std = np.std(rmse_standard_list, ddof=1) / np.sqrt(len(rmse_standard_list))
        mod_sum = ("-" * 20) + f"\nModel: {model_name} Mean LL: {ll_average} Std LL: {ll_standard} RMSE: {rmse_average} RMSE STD: {rmse_standard} LL(non-ar): {ll_standard_average} LL STD(non-ar): {ll_standard_std} rmse(non-ar): {rmse_standard_average} rmse STD(non-ar): {rmse_standard_std}"
        print(mod_sum)
        mod_sum += '\n'
        out_txt += mod_sum
        with open(output_file, 'a') as file:
            file.write(mod_sum)

# Measures runtime and memory for a function
def measure_perf_func(callable_function, num_warmup, average_over):
    assert average_over >= 1, "Must call function at least once"
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(num_warmup): callable_function() # wamrup runs
    runtimes, mems = [], []
    for _ in range(average_over):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        starter.record()
        callable_function()
        ender.record()
        torch.cuda.synchronize()
        peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        runtime_ms = starter.elapsed_time(ender)
        runtimes.append(runtime_ms)
        mems.append(peak_memory_mb)
    runtime_ms_mean = np.mean(np.array(runtimes))
    peak_memory_mb_mean = np.mean(np.array(mems))
    return runtime_ms_mean, peak_memory_mb_mean

@torch.no_grad()
def measure_ar_unroll_time(models, data, samples_unroll, use_flash, priotise_fixed,  base_out_txt_file: str, average_over, burn_in, cache_mhca: bool, run_mode: Literal["normal, flash", "normal_cast"], tnpa_modes_list, device: str = "cuda"):
    assert run_mode in ("normal", "flash", "normal_cast"), "incorrect run mode config"
    assert torch.backends.cuda.flash_sdp_enabled(), "Flash Attention is not available on this device/environment!"
    # Hypers to select - also look at dataset hypers
    ordering = "given"
    output_file = base_out_txt_file + f'_{ordering}.txt'
    if os.path.exists(output_file):
            print(f"Removing old results file: {output_file}")
            os.remove(output_file)
    # End of hypers
    out_txt = ""
    tnpa_i = 0
    for (model_yml, model_wab, model_name, is_local) in models:
        # Loads HadISD timings slightly differently
        if isinstance(is_local, str) and is_local.startswith("hadISD-"):
            is_local_var = is_local.split("hadISD-")[-1]
            weights_only_evalhad_call = is_local_var == "True"
            load_mod_weights = is_local_var != "config"
            model = get_model(model_yml, model_wab, seed=False, device=device, weights_only_evalhad_call=weights_only_evalhad_call, load_mod_weights=load_mod_weights)
        else: # Normal gp model loading
            model = get_model(model_yml, model_wab, seed=False, device=device, local_weights=is_local)
        # Casts model if using flash
        data_type = torch.float32 if run_mode == "normal" else torch.bfloat16
        model.to(dtype=data_type)
        model.eval()
        runtime_list, memory_list = [], []
        first_run = True
        if model_name == "TNP-A": # Adds TNP-A variants to eval to list
            model_name = f"{model_name}_{tnpa_modes_list[0]}"
            for j in range(1, len(tnpa_modes_list)): models.append((model_yml, model_wab, f"TNP-A_{tnpa_modes_list[j]}", is_local)) 
        for batch in tqdm(data, desc=f'{model_name} eval'):
            # Moves data to device (GPU)
            batch.xc, batch.yc = batch.xc.to(device), batch.yc.to(device)
            batch.xt, batch.yt = batch.xt.to(device), batch.yt.to(device)
            batch.x, batch.y = batch.x.to(device), batch.y.to(device)
            # Predict function to be measured
            def _arunroll_pred_fun_param():
                    ar_predict(model=model, xc=batch.xc, yc=batch.yc, xt=batch.xt,order=ordering,num_samples=samples_unroll,prioritise_fixed=priotise_fixed,device=device,device_ret=device,use_flash=use_flash, run_mode=run_mode, cast_model=False, cache_mhca=cache_mhca)

            if model_name.startswith("TNP-A"): # TNP-A is already autoregressive
                model.rollout_mode = tnpa_modes_list[tnpa_i]
                model.num_samples = samples_unroll
                def _arunroll_pred_fun_param():
                    with torch.no_grad(), torch.autocast(device_type=device, dtype=data_type):
                        np_pred_fn(model, batch, predict_without_yt_tnpa=True)

            if first_run: # Warmup calls
                for _ in range(burn_in): _arunroll_pred_fun_param()
                first_run = False
            runtime, memory = measure_perf_func(callable_function=_arunroll_pred_fun_param, average_over=average_over, num_warmup=0)
            runtime_list.append(runtime)
            memory_list.append(memory)
            print(f"[{model_name}] Runtime = {runtime} Memory = {memory} S={samples_unroll} BS={batch.xc.shape[0]} N_c={batch.xc.shape[1]} N_t={batch.xt.shape[1]}")
        if model_name.startswith("TNP-A"): tnpa_i = (tnpa_i + 1) % len(tnpa_modes_list)
        runtime_mean = np.mean(np.array(runtime_list))
        memory_mean = np.mean(np.array(memory_list))
        mod_txt = f"Model: {model_name} Runtime: {runtime_mean} Memory: {memory_mean}\n"
        with open(output_file, 'a') as file:
            file.write(mod_txt)
        del model
        gc.collect()
        torch.cuda.empty_cache()
                

# Runs the experiments needed to compare all models
def had_isd_timings():
    # Time of each AR model - needs flash attention from
    S = 50 # Number of samples to unroll
    NUM_SAMPLES_AVERAGE_OVER = 512
    burn_in = 2
    average_over = 1
    use_flash = True
    prioritise_fixed = False
    device = "cuda"
    run_mode = "normal_cast"
    cache_mhca = True # caches mhca but this does come at cost of more memory and only slightly better runtime
    tnpa_modes_list = ["cache", "normal", "fast"] # List of tnpa rollout modes to try
    measure_ar_unroll_time(models=model_list_hadtemporal_timings(), data=get_hadisd_temporal_data(samples_per_epoch=NUM_SAMPLES_AVERAGE_OVER), samples_unroll=S, use_flash=use_flash, priotise_fixed=prioritise_fixed,  base_out_txt_file="experiments/plot_results/ar/table/arTime_hadTemporal", average_over=average_over, burn_in=burn_in, device=device, run_mode=run_mode, cache_mhca=cache_mhca, tnpa_modes_list=tnpa_modes_list)



# Runs the experiments needed to compare all models
def gen_comparison_table():
    # Time of each AR model - needs flash attention from
    S = 50 # Number of samples to unroll
    NUM_SAMPLES_AVERAGE_OVER = 1088
    burn_in = 5
    average_over = 2
    use_flash = True
    prioritise_fixed = False
    device = "cuda"
    run_mode = "normal_cast"
    cache_mhca = False
    tnpa_modes_list = ["cache", "normal", "fast"] # List of tnpa rollout modes to try
    measure_ar_unroll_time(models=model_list_rbf_nc_64(), data=get_rbf_data(samples_per_epoch=NUM_SAMPLES_AVERAGE_OVER, max_nc=64), samples_unroll=S, use_flash=use_flash, priotise_fixed=prioritise_fixed,  base_out_txt_file="experiments/plot_results/ar/table/arTime_rbf_nc64", average_over=average_over, burn_in=burn_in, device=device, run_mode=run_mode, cache_mhca=cache_mhca, tnpa_modes_list=tnpa_modes_list)
    #measure_ar_unroll_time(models=model_list_rbf_nc_512(), data=get_rbf_data(samples_per_epoch=NUM_SAMPLES_AVERAGE_OVER, max_nc=512), samples_unroll=S, use_flash=use_flash, priotise_fixed=prioritise_fixed,  base_out_txt_file="experiments/plot_results/ar/table/arTime_rbf_nc512", average_over=average_over, burn_in=burn_in, device=device,run_mode=run_mode,cache_mhca=cache_mhca)
    #measure_ar_unroll_time(models=model_list_combined_nc_64(), data=get_combined_data(samples_per_epoch=NUM_SAMPLES_AVERAGE_OVER, max_nc=64), samples_unroll=S, use_flash=use_flash, priotise_fixed=prioritise_fixed,  base_out_txt_file="experiments/plot_results/ar/table/arTime_combined_nc64", average_over=average_over, burn_in=burn_in, device=device,run_mode=run_mode,cache_mhca=cache_mhca)
    #measure_ar_unroll_time(models=model_list_combined_nc_512(), data=get_combined_data(samples_per_epoch=NUM_SAMPLES_AVERAGE_OVER, max_nc=512), samples_unroll=S, use_flash=use_flash, priotise_fixed=prioritise_fixed,  base_out_txt_file="experiments/plot_results/ar/table/arTime_combined_nc512", average_over=average_over, burn_in=burn_in, device=device,run_mode=run_mode,cache_mhca=cache_mhca)

    NUM_SAMPLES_AR = 8_192
    compare_models(models=model_list_rbf_nc_64(), data=get_rbf_data(samples_per_epoch=NUM_SAMPLES_AR, max_nc=64), base_out_txt_file="experiments/plot_results/ar/table/ar_rbf_nc64", eval_ar=True)
    #compare_models(models=model_list_rbf_nc_512(), data=get_rbf_data(samples_per_epoch=NUM_SAMPLES_AR, max_nc=512), base_out_txt_file="experiments/plot_results/ar/table/ar_rbf_nc512", eval_ar=True)
    #compare_models(models=model_list_combined_nc_64(), data=get_combined_data(samples_per_epoch=NUM_SAMPLES_AR, max_nc=64), base_out_txt_file="experiments/plot_results/ar/table/ar_combined_nc64", eval_ar=True)
    #compare_models(models=model_list_combined_nc_512(), data=get_combined_data(samples_per_epoch=NUM_SAMPLES_AR, max_nc=512), base_out_txt_file="experiments/plot_results/ar/table/ar_combined_nc512", eval_ar=True)

    # Non AR LLs (note this includes TNP-A)
    NUM_SAMPLES_STANDARD = NUM_SAMPLES_AR
    compare_models(models=model_list_rbf_nc_64(), data=get_rbf_data(samples_per_epoch=NUM_SAMPLES_STANDARD, max_nc=64), base_out_txt_file="experiments/plot_results/ar/table/standard_rbf_nc64", eval_ar=False)
    #compare_models(models=model_list_rbf_nc_512(), data=get_rbf_data(samples_per_epoch=NUM_SAMPLES_STANDARD, max_nc=512), base_out_txt_file="experiments/plot_results/ar/table/standard_rbf_nc512", eval_ar=False)
    #compare_models(models=model_list_combined_nc_64(), data=get_combined_data(samples_per_epoch=NUM_SAMPLES_STANDARD, max_nc=64), base_out_txt_file="experiments/plot_results/ar/table/standard_combined_nc64", eval_ar=False)
    #compare_models(models=model_list_combined_nc_512(), data=get_combined_data(samples_per_epoch=NUM_SAMPLES_STANDARD, max_nc=512), base_out_txt_file="experiments/plot_results/ar/table/standard_combined_nc512", eval_ar=False)


#if __name__ == "__main__":
    #had_isd_timings()
