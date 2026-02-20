# AR NP eval on HadISD Temporal predictions
import torch
from check_shapes import check_shapes
from torch import nn
from typing import Optional, Union, Literal, Callable, Tuple
from tnp.utils.np_functions import np_pred_fn
from tnp.data.base import Batch
from tnp.models.incUpdateBase import IncUpdateEff, IncUpdateEffFixed
from plot_adversarial_perms import get_model
from functools import partial
from tqdm import tqdm
import numpy as np
import torch.distributions as td
import os
import matplotlib.pyplot as plt
import matplotlib
from tnp.data.hadISDTemporal import TemporalHadISDDataGenerator, scale_pred_temp_dist, get_true_temp, TemporalHadISDBatch
from plot_hadISD_temporal import plot_hadISD_temporal
import numpy as np
from tnp.utils.data_loading import adjust_num_batches
from data_temp.data_processing.elevations import get_cached_elevation_grid
from arnp import ar_loglik
from tnp.utils.ram_data_loader import RamDiskDataLoader
import copy
import gc


matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"
matplotlib.rcParams["axes.titlesize"]= 14


@check_shapes(
    "xc: [m, nc, dx]", "yc: [m, nc, dy]", "xt: [m, nt, dx]", "yt: [m, nt, dy]",
)
@torch.no_grad
def _shuffle_targets(np_model: nn.Module, xc: torch.Tensor, yc: torch.Tensor, xt: torch.Tensor, yt: Optional[torch.Tensor],
    order: Literal["random", "given", "left-to-right", "variance", "time", "spatiotemporal"]):
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
    elif order == "time":
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
@torch.no_grad
def ar_metrics(np_model: nn.Module, xc: torch.Tensor, yc: torch.Tensor, xt: torch.Tensor, yt: torch.Tensor, raw_batch: TemporalHadISDBatch,
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
        # Converts to degrees celsius
        pred_dist = scale_pred_temp_dist(raw_batch, pred_dist)
        yt_sel = get_true_temp(raw_batch, yt_sel)

        log_probs += pred_dist.log_prob(yt_sel).sum(dim=(-1, -2))

        squared_errors += (pred_dist.mean - yt_sel).to(squared_errors.dtype).pow(2).sum(dim=(-1, -2))
    if normalise:
        log_probs /= (nt * dy)
    rmse = torch.sqrt(squared_errors / (nt * dy)).to(xt.dtype)
    return log_probs, rmse


@check_shapes(
    "xc: [m, nc, dx]", "yc: [m, nc, dy]", "xt: [m, nt, dx]"
)
@torch.no_grad
def ar_predict(model, xc: torch.Tensor, yc: torch.Tensor, xt: torch.Tensor,
    order: Literal["random", "given", "left-to-right", "variance"] = "random",
    num_samples: int = 10,
    prioritise_fixed: bool = False, # If incremental updates are available prioritise fixed or true dynamic algorithm
    device: str = "cuda", # Device for computing
    device_ret: str = "cpu", # Return device
    use_flash: bool = False, # Use flash kernel if posible,
    return_rollout_samples: bool = False, # Returns rollout samples too
    ):
    m, nt, dx = xt.shape
    _, nc, dy = yc.shape
    xc, yc, xt = xc.to(device), yc.to(device), xt.to(device)

    xc_stacked = xc.repeat_interleave(num_samples, dim=0)
    yc_stacked = yc.repeat_interleave(num_samples, dim=0)
    xt_stacked = xt.repeat_interleave(num_samples, dim=0)

    xt_stacked, _, perm = _shuffle_targets(model, xc_stacked, yc_stacked, xt_stacked, None, order)

    yt_preds_mean, yt_preds_std = torch.empty((m * num_samples, nt, dy), device=device), torch.empty((m * num_samples, nt, dy), device=device)

    if return_rollout_samples:
        roll_samples = torch.empty_like(yt_preds_mean)

    is_fixed_inc_update = isinstance(model, IncUpdateEffFixed)
    is_inc_gen_update = isinstance(model, IncUpdateEff)
    is_fixed_inc_update = (is_fixed_inc_update and prioritise_fixed) or (is_fixed_inc_update and not is_inc_gen_update)
    is_inc_gen_update = (is_inc_gen_update and not prioritise_fixed) or (is_inc_gen_update and not is_fixed_inc_update)
    assert is_fixed_inc_update != is_inc_gen_update or (not is_fixed_inc_update and not is_inc_gen_update), "Xor onf fixed vs inc update"
    if is_inc_gen_update:
        model.init_inc_structs(m=xc_stacked.shape[0], max_nc=nc+nt, device=device,use_flash=use_flash)
        model.update_ctx(xc=xc_stacked, yc=yc_stacked,use_flash=use_flash)
    elif is_fixed_inc_update:
        model.init_inc_structs_fixed(m=xc_stacked.shape[0], max_nc=nc+nt, xt=xt_stacked, dy=dy, device=device, use_flash=use_flash)
        model.update_ctx_fixed(xc=xc_stacked, yc=yc_stacked,use_flash=use_flash) # Added in degubg this maybe

    for i in range(nt):
        xt_tmp = xt_stacked[:, i:i+1,:]
        if is_inc_gen_update:
            pred_dist = model.query(xt=xt_tmp, dy=dy,use_flash=use_flash)
        elif is_fixed_inc_update:
            pred_dist = model.query_fixed(tgt_start_ind=i, tgt_end_ind=i+1, use_flash=use_flash)
        else:
            batch = Batch(xc=xc_stacked, yc=yc_stacked, xt=xt_tmp, yt=None, x=None, y=None)
            pred_dist = np_pred_fn(model, batch)
        assert isinstance(pred_dist, td.Normal), "Must predict a gaussian"
        pred_mean, pred_std = pred_dist.mean, pred_dist.stddev
        yt_preds_mean[:,i:i+1,:] = pred_mean
        yt_preds_std[:,i:i+1,:] = pred_std
        if return_rollout_samples:
            y_sample = pred_dist.sample()
            roll_samples[:,i:i+1,:] = y_sample
        # Samples from the predictive distribution and updates the context
        if i < nt - 1:
            #yt_sampled = pred_dist.sample() # [m * num_samples, 1, dy]
            yt_sampled = pred_dist.sample() if not return_rollout_samples else y_sample # [m * num_samples, 1, dy]
            if is_inc_gen_update:
                model.update_ctx(xc=xt_tmp, yc=yt_sampled,use_flash=use_flash)
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
    if return_rollout_samples:
        roll_samples = roll_samples.gather(dim=1, index=idx)
        roll_samples = roll_samples.view(num_samples, m, nt, dy).permute(1,2,3,0)

    mix = td.Categorical(torch.full((m, nt, dy, num_samples), 1.0 / num_samples, device=device_ret))
    comp = td.Normal(yt_preds_mean.to(device_ret), yt_preds_std.to(device_ret))
    approx_dist = td.MixtureSameFamily(mix, comp)

    # For sample draws return raw samples and run through model again for smooth samples (see paper / code)
    return approx_dist if not return_rollout_samples else (approx_dist, roll_samples)



# -------------------------------------------------------------------------------------------------------

# Measures timings of different models - this from some perspecticves is v similar to ar.py (doesnt use dataset per se)
def measure_perf_timings():
    # Measure hypers
    burn_in = 1 # Number of burn in runs to ignore
    aggregate_over = 1 # Number of runs to aggregate data over
    token_step = 500 # How many increments of tokens to go up in
    min_nt, max_nt = 1, 2003
    dx, dy, m = 4, 1, 1
    nc_start = 1
    num_samples=50 # Samples to unroll in ar_predict
    device = "cuda"
    order="random"
    prioritise_fixed = False
    plot_name_folder = "experiments/plot_results/hadar/perf/"
    # End of measure hypers
    models = get_model_list()
    max_high = 2
    xc = (torch.rand((m, nc_start, dx), device=device) * max_high * 2) - max_high
    yc = (torch.rand((m, nc_start, dy), device=device) * max_high * 2) - max_high
    target_sizes = np.arange(start=min_nt, stop=max_nt, step=token_step, dtype=int)
    runtime = np.zeros((len(models), aggregate_over, len(target_sizes)))
    memory = np.zeros((len(models), aggregate_over, len(target_sizes)))
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for model_idx, (model_yml, model_wab, model_name) in enumerate(models):
        model = get_model(model_yml, model_wab, seed=False, device=device)
        model.eval() 
        for t_index, nt in tqdm(enumerate(target_sizes), desc=f'Targ {model_name}'):
            xt = (torch.rand((m, nt, dx), device=device) * max_high * 2) - max_high
            yt = (torch.rand((m, nt, dy), device=device) * max_high * 2) - max_high

            for j in range(burn_in + aggregate_over):
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                starter.record()
                with torch.no_grad():
                    pred_dist = ar_predict(model=model, xc=xc, yc=yc, xt=xt, order=order, num_samples=num_samples,
                        device=device, device_ret=device, prioritise_fixed=prioritise_fixed)
                # Measures time and memory
                ender.record()
                torch.cuda.synchronize()
                peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
                runtime_ms = starter.elapsed_time(ender)
                # Stores results
                write_idx = j - burn_in
                if write_idx >= 0:
                    runtime[model_idx, write_idx, t_index] = runtime_ms
                    memory[model_idx, write_idx, t_index] = peak_memory_mb
    # Aggregates results
    runtime = np.mean(runtime, axis=1) # [no_models, len(target_sizes)]
    memory = np.mean(memory, axis=1)
    # Plots runtime
    runtime_file_name = plot_name_folder + f'runtime_od_{order}_samples_{num_samples}_nc{nc_start}.png'
    fig, ax = plt.subplots(figsize=(7, 5))
    for model_idx, (model_yml, model_wab, model_name) in enumerate(models):
        ax.plot(target_sizes, runtime[model_idx] / 1000.0, label=model_name)
    ax.set_xlabel('Target Size')
    ax.set_ylabel('Runtime (s)')
    ax.legend()
    ax.set_title(f'Runtime of AR NPs (S={num_samples} NC={nc_start})')
    ax.grid(True, linestyle='--', alpha=0.4)
    fig.tight_layout()
    plt.savefig(runtime_file_name, dpi=300)
    # Plots memory
    memory_file_name = plot_name_folder + f'memory_od_{order}_samples_{num_samples}_nc{nc_start}.png'
    fig, ax = plt.subplots(figsize=(7, 5))
    for model_idx, (model_yml, model_wab, model_name) in enumerate(models):
        ax.plot(target_sizes, memory[model_idx], label=model_name)
    ax.set_xlabel('Target Size')
    ax.set_ylabel('Memory Usage (MB)')
    ax.legend()
    ax.set_title(f'Memory Usage of AR NPs (S={num_samples} NC={nc_start})')
    ax.grid(True, linestyle='--', alpha=0.4)
    fig.tight_layout()
    plt.savefig(memory_file_name, dpi=300)



# Plots a handful of kernels
def plot_ar_unrolls():
    # Hypers
    huge_grid_plots = False # Whether to plot enormous grid prediction - very slow computationally for AR
    order="given"
    #no_samples = [1, 2, 5, 10, 50, 100, 500, 1000]
    no_samples = [1, 2, 10, 50]
    folder_name = "experiments/plot_results/hadISDTime/plots/"
    no_kernels = 5#20
    device="cuda"
    use_flash = False
    ordering_loader = "random"
    # End of hypers
    models = get_model_list(ordering_loader=ordering_loader)
    data, lat_mesh, lon_mesh, elev_np = get_had_testset_and_plot_stuff(ordering_loader=ordering_loader)

    batches_plot = []
    original_root= data.dataset.data_root
    with RamDiskDataLoader(False, original_root) as active_root:
        # If data has been moved to RAM on HPC system update generator paths - quite hacky should only use this when speedup justified and changes well tested
        if active_root != original_root:
            print(f"Redirecting generators to RAM: {active_root}")
            if data is not None and hasattr(data, 'update_root'):
                data.update_root(active_root)

        # Loads batches to be plotted
        for i, batch in enumerate(data):
            batches_plot.append(batch)
            if i >= no_kernels: break

    for (model_yml, model_wab, model_name) in models:
        model = get_model(model_yml, model_wab, seed=False, device=device)
        model.eval()
        model_folder = f"{folder_name}/{model_name}"
        if not os.path.isdir(model_folder):
            os.makedirs(model_folder)
        for sample in no_samples:
            def pred_fn_pred(model, batch, predict_without_yt_tnpa=True):
                if use_flash:
                    with torch.no_grad(), torch.autocast(device_type=device, dtype=torch.float16), torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
                        pred_dist = ar_predict(model, batch.xc, batch.yc, batch.xt, order, sample, device=device, use_flash=use_flash)
                else:
                    with torch.no_grad():
                        pred_dist = ar_predict(model, batch.xc, batch.yc, batch.xt, order, sample, device=device, use_flash=use_flash)
                return pred_dist
            plot_hadISD_temporal(
                model=model,
                batches=batches_plot,
                num_fig=len(batches_plot),
                name=model_folder+f"/ns_{sample}_od_{order}",
                pred_fn=pred_fn_pred,
                lat_mesh=lat_mesh,
                lon_mesh=lon_mesh,
                elev_np=elev_np,
                savefig=True, 
                logging=False,
                model_lbl=f"AR {model_name} (S={sample}) ",
                huge_grid_plots=huge_grid_plots,
                delta_hours=int(data.dataset.delta_hours),
            )
                    

# Loads hadISD set
def get_had_testset_and_plot_stuff(ordering_loader,
                                   samples_per_epoch=4_000,
                                   N_c_min=100, N_c_max=2100,
                                   N_t_min=250, N_t_max=250,
                                   batch_size=32,
                                   h_window=8,
                                   delta_hours=6):
    dem_path = "REMOVED"
    cache_dem_dir = "REMOVED"
    data_root = "REMOVED"
    num_grid_points_plot = 100
    # Normal hypers
    #N_c_min = 100
    #N_c_max = 2100
    #N_t_min = 250
    #N_t_max = 250
    split ="test"
    #samples_per_epoch= 4_000
    #batch_size = 32
    num_val_workers = 1
    # Change these depending on the training / eval dist desired
    #delta_hours = 6
    #h_window = 8

    # Loads had dataset
    gen_test = TemporalHadISDDataGenerator(N_c_min=N_c_min, N_c_max=N_c_max, N_t_min=N_t_min, N_t_max=N_t_max,
        split=split, samples_per_epoch=samples_per_epoch, batch_size=batch_size, data_root=data_root,
        ordering=ordering_loader, delta_hours=delta_hours, h_window=h_window)
    
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

    # Loads elevation data from DEM file
    lat_mesh, lon_mesh, elev_np = get_cached_elevation_grid(gen_test.lat_range, gen_test.long_range,
        num_grid_points_plot, cache_dem_dir,
        dem_path)

    return test_loader, lat_mesh, lon_mesh, elev_np


def get_model_list(ordering_loader):
    assert ordering_loader in ("random", "ctx_time", "full_time", "forecasting"), "Invalid time ordering loader"
    folder = f"hadISDTemporal"
    tnp_plain = (f'experiments/configs/{folder}/hadtemp_tnp_plain.yml',
        'REMOVED', 'TNP-D', True)
    models = []
    return models

@torch.no_grad()
def eval_had_ar_model(model, data, model_name, device, ordering, rollout_rmse, num_samples, prioritise_fixed, use_flash, max_no_batches):
    model.eval()
    batch_i = 0
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ll_list, rmse_list = [], []
    ll_standard_list, rmse_standard_list = [], []
    rmse_unroll_list = []
    tot_time_unroll = 0
    for batch in tqdm(data, desc=f'{model_name} eval'):
        _, nt, dy = batch.yt.shape

        batch.xc, batch.yc = batch.xc.to(device), batch.yc.to(device)
        batch.xt, batch.yt = batch.xt.to(device), batch.yt.to(device)
        batch.x, batch.y = batch.x.to(device), batch.y.to(device)
        
        # Computes ar ll
        #ll = ar_loglik(np_model=model, xc=batch.xc, yc=batch.yc,
        #    xt=batch.xt, yt=batch.yt, normalise=True, order=ordering)
        # standard ll
        with torch.no_grad():
            pred_dist = np_pred_fn(model, batch)
            pred_dist_temp = scale_pred_temp_dist(batch, pred_dist)
            yt_correct_units = get_true_temp(batch, batch.yt)
            loglik_temp_standard = pred_dist_temp.log_prob(yt_correct_units).sum() / yt_correct_units[..., 0].numel()
            rmse_temp_standard = nn.functional.mse_loss(pred_dist_temp.mean, yt_correct_units).sqrt().cpu().mean()
        mean_ll_standard = torch.mean(loglik_temp_standard).item()
        mean_rmse_standard = torch.mean(rmse_temp_standard).item()

        #print(ll)
        #rmse=ll
        if model_name == "TNP-A":
            mean_ll = -1.0
            mean_rmse = -1.0
        else:
            ll, rmse = ar_metrics(np_model=model, xc=batch.xc, yc=batch.yc,
                xt=batch.xt, yt=batch.yt, normalise=True, order=ordering, raw_batch=batch)
            if rollout_rmse:
                torch.cuda.synchronize()
                starter.record()
                pred_dist = ar_predict(model=model, xc=batch.xc, yc=batch.yc, xt=batch.xt,
                    order=ordering, num_samples=num_samples,
                    device=device, device_ret=device, prioritise_fixed=prioritise_fixed, use_flash=use_flash,
                    return_rollout_samples=False)
                ender.record()
                torch.cuda.synchronize()
                runtime_ms_unroll = starter.elapsed_time(ender)
                tot_time_unroll += runtime_ms_unroll
                yt_temp_units = get_true_temp(batch, batch.yt)
                pred_mean = get_true_temp(batch, pred_dist.mean.to(device))
                #roll_temp_units = roll_samples * batch.std_temp + batch.mean_temp
                #diff2 = (roll_temp_units - yt_temp_units.unsqueeze(-1)).pow(2)
                diff2 = (pred_mean - yt_temp_units).pow(2)
                rmse_rollout = torch.sqrt(diff2.mean(dim=(-3, -2, -1)))
            if rollout_rmse:
                mean_rmse_unroll = rmse_rollout.mean().item()
                rmse_unroll_list.append(mean_rmse_unroll)
            #mean_ll = torch.mean(ll).item() - np.log(batch.std_temp) # Goes from [m] to a float
            mean_ll = torch.mean(ll).item()
            mean_rmse = torch.mean(rmse).item()
        
        #mean_ll = mean_ll - np.log(batch.std_temp) # Norm dist appropriately
        print(f"ll {mean_ll} rmse {mean_rmse} rmseroll {mean_rmse_unroll if rollout_rmse else None} ll_stand {mean_ll_standard} rmse_stand {mean_rmse_standard}")
        #print(ll)
        ll_list.append(mean_ll)
        rmse_list.append(mean_rmse)
        ll_standard_list.append(mean_ll_standard)
        rmse_standard_list.append(mean_rmse_standard)
        
        if max_no_batches is not None and batch_i + 1 >= max_no_batches: break
        batch_i += 1
    ll_average = np.mean(ll_list)
    ll_std = np.std(ll_list, ddof=1) / np.sqrt(len(ll_list))
    if rollout_rmse:
        rmse_unroll_average = np.mean(rmse_unroll_list)
        rmse_unroll_std = np.std(rmse_unroll_list, ddof=1) / np.sqrt(len(rmse_unroll_list))
        tot_time_unroll /= 1000 # To seconds

    rmse_average = np.mean(rmse_list)
    ll_standard_average = np.mean(ll_standard_list)
    ll_standard_std = np.std(ll_standard_list, ddof=1) / np.sqrt(len(ll_standard_list))
    rmse_standard_average = np.mean(rmse_standard_list)
    rmse_standard_std = np.std(rmse_standard_list, ddof=1) / np.sqrt(len(rmse_standard_list))
    rmse_std = np.std(rmse_list, ddof=1) / np.sqrt(len(rmse_list))
    mod_sum = ("-" * 20) + f"\nModel: {model_name}\nMean LL: {ll_average} STD LL: {ll_std} Mean RMSE: {rmse_average} STD RMSE: {rmse_std} Mean LL(Standard): {ll_standard_average} STD LL(Standard): {ll_standard_std} MEAN RMSE(Standard): {rmse_standard_average} STD RMSE(Standard): {rmse_standard_std}"
    if rollout_rmse:
        mod_sum += f" Num Samples: {num_samples} Mean UnrRMSE: {rmse_unroll_average} STD UnrRMSE: {rmse_unroll_std} Total Unroltime: {tot_time_unroll}"
    mod_sum += "\n"
    print(mod_sum)

    # Returns JSON also
    out_struct = {
        "Model": model_name,
        "Mean_LL_AR": ll_average,
        "STD_LL_AR": ll_std,
        "Mean_RMSE_AR": rmse_average,
        "STD_RMSE_AR": rmse_std,
        "Mean_LL_Standard": ll_standard_average,
        "STD_LL_Standard": ll_standard_std,
        "Mean_RMSE_Standard": rmse_standard_average,
        "STD_RMSE_Standard": rmse_standard_std
    }
    if rollout_rmse:
        out_struct["num_samples_ar_unroll"] = num_samples
        out_struct["mean_rmseunroll_ar"] = rmse_unroll_average
        out_struct["std_rumseunrll_ar"] = rmse_unroll_std
        out_struct["tot_unrolltime_ar"] = tot_time_unroll

    return mod_sum, out_struct


def compare_had_models(base_out_txt_file: str, rollout_rmse: bool, device: str = "cuda"):
    # Hypers to select - also look at dataset hypers
    ordering = "given"
    num_samples = 50
    prioritise_fixed = True
    use_flash = False
    max_no_batches = None # None for whole test set
    ordering_loader = "ctx_time"
    # End of hypers
    # Main loop - loads each model than compares writes performances to a text file
    models = get_model_list(ordering_loader=ordering_loader)
    data, lat_mesh, lon_mesh, elev_np = get_had_testset_and_plot_stuff(ordering_loader=ordering_loader)
    out_txt = ""

    # Output file
    output_file = base_out_txt_file + f'_{ordering}.txt'
    if os.path.exists(output_file):
            print(f"Removing old results file: {output_file}")
            os.remove(output_file)

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for (model_yml, model_wab, model_name, weights_only_evalhad_call) in models:
        model = get_model(model_yml, model_wab, seed=False, device=device, weights_only_evalhad_call=weights_only_evalhad_call)
        mod_sum, _ = eval_had_ar_model(model=model, data=data, model_name=model_name, device=device, ordering=ordering, rollout_rmse=rollout_rmse, num_samples=num_samples, prioritise_fixed=prioritise_fixed, use_flash=use_flash, max_no_batches=max_no_batches)
        out_txt += mod_sum
        with open(output_file, 'a') as file:
            file.write(mod_sum)
        del model
        gc.collect()
        torch.cuda.empty_cache()


#if __name__ == "__main__":
    #compare_had_models(base_out_txt_file="experiments/plot_results/eval_hadTemp/ar_had_comp", rollout_rmse=False)