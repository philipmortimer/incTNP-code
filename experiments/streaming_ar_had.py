# Tracks the streaming performance in AR mode for hadISD Temporal models
import json
from pathlib import Path
from arnp import ar_predict # This ar_predict has been optimised for maximum speed. In HadISD temporal we need to convert this back to the correct units
from arnp_hadTemporal import ar_metrics, get_model_list, get_had_testset_and_plot_stuff # (ar_metrics is correct LL teacher forcing calculation in temp units)
from data_temp.data_processing.elevations import get_cached_elevation_grid
from tnp.data.hadISDTemporal import TemporalHadISDBatch, TemporalHadISDDataGenerator, get_true_temp, scale_pred_temp_dist
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
import lightning.pytorch as pl
from tueplots import bundles
from contextlib import nullcontext

plt.rcParams.update(bundles.icml2024())

# debug function
def compare_distributions(dist1, dist2, atol=1e-4, rtol=1e-4):
    # Mean check
    mean1 = dist1.mean if hasattr(dist1, 'mean') else dist1.component_distribution.mean.mean(dim=-2)
    mean2 = dist2.mean if hasattr(dist2, 'mean') else dist2.component_distribution.mean.mean(dim=-2)
    mean_check = torch.allclose(mean1, mean2, atol=atol, rtol=rtol)
    # Variance checl
    var1 = dist1.variance if hasattr(dist1, 'variance') else dist1.component_distribution.variance.mean(dim=-2)
    var2 = dist2.variance if hasattr(dist2, 'variance') else dist2.component_distribution.variance.mean(dim=-2)
    var_check = torch.allclose(var1, var2, atol=atol, rtol=rtol)
    if not mean_check:
        diff = (mean1 - mean2).abs().max().item()
        print(f"DEBUG FAIL: Means diverge. Max diff: {diff}")
    if not var_check:
        diff = (var1 - var2).abs().max().item()
        print(f"DEBUG FAIL: Variances diverge. Max diff: {diff}")
    #assert mean_check and var_check, "Streaming update output does not match batch output!"
    diff_mean = (mean1 - mean2).abs().max().item()
    diff_var = (var1 - var2).abs().max().item()
    if mean_check and var_check: print(f"DEBUG PASS: Streaming update matches batch calculation. mean_max={diff_mean} var_max={diff_var}")


# Measuring the runtime (y axis) and memory (y axis) for AR streaming approaches vs the streamed context size (x axis)
@torch.no_grad()
def measure_streaming_ar_timings(models, aggregate_over, token_step, min_nc, max_nc, dx, dy, num_samples, device, cache_mhca, folder, use_flash, run_experiments, m, nt, run_mode, order, plot_ctx_end, trained_ctx_end):
    os.makedirs(folder, exist_ok=True)
    json_file_path = folder + f'summary.json'
    assert order == "random", "order should be random for AR"
    assert run_mode in ("normal", "flash", "normal_cast"), "run mode must be specified"
    if run_experiments:
        # Hypers that should be fixed
        prioritise_fixed = False
        device_ret = "cpu"
        cast_model_setting = False # We already cast ourselves
        test_ar_updates = False # Debug ar updates
        #
        xc = (torch.rand((m, max_nc, dx), device=device) * 2) - 1 # Only need to use tensors of correct SHAPE when measuring computation costs, dont need data loader
        yc = (torch.rand((m, max_nc, dy), device=device)* 2) -1
        xt = (torch.rand((m, nt, dx), device=device) * 2) -1
        yt = (torch.rand((m, nt, dy), device=device)* 2) -1
        last_conditioned_ctx_size_inc = 0 # We have conditioned on nothing yet
        cast_ctx_manager = nullcontext() if run_mode  == "normal" else torch.autocast(device_type=device, dtype=torch.bfloat16)
        ctx_sizes = [i for i in range(min_nc, max_nc, token_step)]#np.arange(start=min_nc, stop=max_nc, step=token_step, dtype=int)
        if max(ctx_sizes) != max_nc: ctx_sizes.append(max_nc)
        ctx_sizes = np.array(ctx_sizes)
        runtime = np.zeros((len(models), aggregate_over, len(ctx_sizes)))
        memory = np.zeros((len(models), aggregate_over, len(ctx_sizes)))
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        for model_idx, (model_yml, model_wab, model_name, weights_only_evalhad_call) in enumerate(models):
            file_name_npz = f'{model_name}.npz'
            npz_arr_path = folder + file_name_npz
            if Path(npz_arr_path).exists(): 
                plot_computation_plots(folder, plot_ctx_end) # Plots after each loop for quicker iteration
                continue # No need to redo computation for model thats already been computed - this assumes the setups are the same
            for i in range(aggregate_over):
                model = get_model(model_yml, model_wab, seed=False, device=device, weights_only_evalhad_call=weights_only_evalhad_call)
                if run_mode != "normal": model.to(dtype=torch.bfloat16)
                model.eval()
                is_inc = isinstance(model, IncUpdateEff)
                is_tnpa = model_name.startswith("TNP-A")
                if is_tnpa:
                    model.num_samples = num_samples
                    model.rollout_mode = model_name[7:-1]
                if is_inc:
                    model.init_inc_structs(m=m, max_nc=max_nc, device=device, use_flash=use_flash, cache_mhca=cache_mhca, persist_small=True)
                with torch.no_grad(), cast_ctx_manager:
                    for t_index, nc in tqdm(enumerate(ctx_sizes), desc=f'Model {model_name}'):
                        if is_inc: # AR models that support incremental updating of ctx
                            # We are measuring the runtime of conditioning a single new ctx point (pure streaming). Thus for cases where ctx_step != 1, we first need to condition on everything we have not conditioned on since last step
                            if last_conditioned_ctx_size_inc < nc - 1:
                                for j in range(last_conditioned_ctx_size_inc, nc - 1):
                                    model.update_ctx(
                                        xc=xc[:, j:j+1, :],
                                        yc=yc[:, j:j+1, :],
                                        use_flash=use_flash,
                                        cache_mhca=cache_mhca,
                                        persist_small=True,
                                    )
                            if test_ar_updates: pl.seed_everything(1)
                            xc_new, yc_new = xc[:,nc-1:nc,:], yc[:, nc-1:nc, :]
                            ar_pred_fn = lambda: ar_predict(model=model, xc=xc_new, yc=yc_new, xt=xt, order=order, num_samples=num_samples, prioritise_fixed=prioritise_fixed, device=device, device_ret=device_ret, use_flash=use_flash, run_mode=run_mode, cast_model=cast_model_setting, cache_mhca=cache_mhca, persist_small=True)
                            last_conditioned_ctx_size_inc = nc
                        elif is_tnpa: # TNPA
                            xc_curr, yc_curr = xc[:,:nc,:], yc[:,:nc,:]
                            batch_ar = Batch(xc=xc_curr, yc=yc_curr, xt=xt, yt=yt, x=torch.cat([xc_curr, xt], dim=1), y=torch.cat([yc_curr, yt], dim=1))
                            ar_pred_fn = lambda: np_pred_fn(model, batch_ar, predict_without_yt_tnpa=True)
                        else: # AR models that do not support incremental ctx updates (e.g. TNP-D)
                            ar_pred_fn = lambda: ar_predict(model=model, xc=xc[:,:nc,:], yc=yc[:,:nc,:], xt=xt, order=order, num_samples=num_samples, prioritise_fixed=prioritise_fixed, device=device, device_ret=device_ret, use_flash=use_flash, run_mode=run_mode, cast_model=cast_model_setting, cache_mhca=cache_mhca, persist_small=False)
                        torch.cuda.reset_peak_memory_stats()
                        torch.cuda.synchronize()
                        starter.record()
                        ret = ar_pred_fn()
                        # Measures time and memory
                        ender.record()
                        torch.cuda.synchronize()
                        peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
                        runtime_ms = starter.elapsed_time(ender)
                        # Stores results
                        runtime[model_idx, i, t_index] = runtime_ms
                        memory[model_idx, i, t_index] = peak_memory_mb
                        if test_ar_updates and is_inc:
                            model_cpy = get_model(model_yml, model_wab, seed=False, device=device, weights_only_evalhad_call=weights_only_evalhad_call)
                            if run_mode != "normal": model_cpy.to(dtype=torch.bfloat16)
                            pl.seed_everything(1)
                            ret_old = ar_predict(model=model_cpy, xc=xc[:,:nc,:], yc=yc[:,:nc,:], xt=xt, order=order, num_samples=num_samples, prioritise_fixed=prioritise_fixed, device=device, device_ret=device_ret, use_flash=use_flash, run_mode=run_mode, cast_model=cast_model_setting, cache_mhca=cache_mhca, persist_small=False)
                            compare_distributions(ret, ret_old)

            # Writes results to file
            np.savez(npz_arr_path, runtime=runtime[model_idx], memory=memory[model_idx], ctx_size=ctx_sizes)
            del model
            gc.collect()
            torch.cuda.empty_cache()
            plot_computation_plots(folder, plot_ctx_end) # Plots after each loop for quicker iteration
        # Saves summary
        summary_meta = {
            "model_names": [m[2] for m in models],
            "folder": folder,
            "trained_ctx_end": trained_ctx_end,
        }
        with open(json_file_path, 'w') as fileobj:
            json.dump(summary_meta, fileobj, indent=4)
        print(f"Summary at {json_file_path}")
    plot_computation_plots(folder, plot_ctx_end)


def plot_computation_plots(folder, plot_ctx_end):
    if not Path(folder).is_dir():
        print(f"No folder file found at {folder}")
        return
    json_path = folder + f'summary.json'
    trained_ctx_end = None
    if Path(json_path).exists():
        with open(json_path, 'r') as fileobj:
            metadata = json.load(fileobj)
        model_names = metadata['model_names']
        trained_ctx_end = metadata['trained_ctx_end']
        #folder = metadata['folder']

    folder_path = Path(folder)
    npz_list = sorted(list(folder_path.glob("*.npz")))
    fig_run, ax_run = plt.subplots()
    fig_mem, ax_mem = plt.subplots()
    fig_cum, ax_cum = plt.subplots()
    max_ctx = -1
    for npz_file in npz_list:
        model_name = npz_file.stem
        data = np.load(npz_file)
        runtime = data['runtime'] / 1000.0
        memory = data['memory']
        ctx_size = data['ctx_size']
        max_ctx = max(max(ctx_size), max_ctx)
        mean_runtime = np.mean(runtime, axis=(0))
        mean_memory = np.mean(memory, axis=(0))
        cumulative_runtime = np.zeros_like(mean_runtime) # Computes cumulative runtime using linear interpolation between steps
        for i in range(1, len(ctx_size)):
            dt_steps = ctx_size[i] - ctx_size[i-1]
            avg_latency = (mean_runtime[i] + mean_runtime[i-1]) / 2.0
            cumulative_runtime[i] = cumulative_runtime[i-1] + (avg_latency * dt_steps)
        ax_run.plot(ctx_size, mean_runtime, label=model_name)
        ax_mem.plot(ctx_size, mean_memory, label=model_name)
        ax_cum.plot(ctx_size, cumulative_runtime, label=model_name)
    plots = [
        (fig_run, ax_run, "runtime_ar.pdf", "Runtime (s)"), 
        (fig_mem, ax_mem, "memory_ar.pdf", "Memory (MB)"),
        (fig_cum, ax_cum, "cumulative_runtime_ar.pdf", "Cumulative Runtime (s)")
    ]
    for fig, ax, filename, ylabel in plots:
        ax.set_ylabel(ylabel)
        if trained_ctx_end is not None and max_ctx > trained_ctx_end and plot_ctx_end:
            ax.axvline(x=trained_ctx_end, color='red', linestyle=':')
            ax.text(x=trained_ctx_end + 5, y=ax.get_ylim()[1] * 0.40, s='Max Trained NC', color='red', rotation=90, verticalalignment='top')
        ax.set_xlabel(r"$N_{s}$")
        ax.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.legend(frameon=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        fig.savefig(folder_path / filename, bbox_inches='tight')
        print(f"Saved {filename}")
    plt.close(fig_run)
    plt.close(fig_mem)


# Gathers results for streaming ar performance. Y axis is NLL (or LL) and X Axis is N_s (number of streamed points) using teacher forcing.
@torch.no_grad()
def streaming_ar_performance(models, task, device, seed, folder, trained_ctx_end, run_experiments, plot_ctx_end):
    os.makedirs(folder, exist_ok=True)
    json_file_path = folder + f'summary.json'
    pl.seed_everything(seed)
    ctx = list(range(task["start_ctx"], task["N_c_max"], task["ctx_step"],))
    ctx.append(task["N_c_max"])
    ctx = np.array(ctx)
    no_data = (task["samples_per_epoch"] // task["batch_size"])
    if task["samples_per_epoch"] % task["batch_size"] != 0: no_data +=1
    data, _, _, _ = get_had_testset_and_plot_stuff(ordering_loader=task["ordering"],samples_per_epoch=task["samples_per_epoch"],N_c_min=task["N_c_min"], N_c_max=task["N_c_max"],N_t_min=task["N_t_min"], N_t_max=task["N_t_max"], batch_size=task["batch_size"],h_window=task["h_window"],delta_hours=task["delta_hours"])
    ll_list = np.zeros((len(models), len(ctx), no_data))
    if run_experiments:
        for model_idx, (model_yml, model_wab, model_name, weights_only_evalhad_call) in enumerate(models):
            file_name_npz = f'{model_name}.npz'
            npz_arr_path = folder + file_name_npz
            if Path(npz_arr_path).exists(): continue # Skip already computed setup
            model = get_model(model_yml, model_wab, seed=False, device=device, weights_only_evalhad_call=weights_only_evalhad_call)
            model.eval()
            for batch_idx, batch in tqdm(enumerate(data), desc=f'{model_name}'):
                # Moves batch to gpu
                batch.xc, batch.yc, batch.xt, batch.yt = batch.xc.to(device), batch.yc.to(device), batch.xt.to(device), batch.yt.to(device)
                xc, yc, xt, yt = batch.xc, batch.yc, batch.xt, batch.yt
                m, nt, dy = yt.shape
                yt_temp = get_true_temp(batch, yt)
                for ctx_idx, ctx_upper in enumerate(ctx):
                    ctx_lower = 0 if ctx_idx == 0 else ctx[ctx_idx - 1]
                    xc_new, yc_new = xc[:, :ctx_upper, :], yc[:, :ctx_upper, :]
                    if model_name == "TNP-A":
                        fake_batch = Batch(xc=xc_new, yc=yc_new, xt=batch.xt, yt=batch.yt, x=None, y=None)
                        with torch.no_grad():
                            pred_dist = np_pred_fn(model, fake_batch, predict_without_yt_tnpa=False) # Teacher forced TNP-A
                        pred_dist = scale_pred_temp_dist(batch, pred_dist)
                        log_prob_mean = pred_dist.log_prob(yt_temp).sum() / yt_temp[..., 0].numel()
                    else:
                        with torch.no_grad():
                            log_probs, _ = ar_metrics(np_model=model, xc=xc_new, yc=yc_new, xt=batch.xt, yt=batch.yt, normalise=True, order=task["target_shuffle"], raw_batch=batch)
                        log_prob_mean = torch.mean(log_probs).item()
                    ll_list[model_idx, ctx_idx, batch_idx] = log_prob_mean
            # Writes model results to file
            np.savez(npz_arr_path, ll=ll_list[model_idx], ctx=ctx)
            del model
            gc.collect()
            torch.cuda.empty_cache()
            plot_saved_info(folder, plot_ctx_end) # Plots for quicker iteration
        # Saves summary
        summary_meta = {
            "model_names": [m[2] for m in models],
            "trained_ctx_end": trained_ctx_end,
            "folder": folder,
            "data_task": task,
        }
        with open(json_file_path, 'w') as fileobj:
            json.dump(summary_meta, fileobj, indent=4)
        print(f"Summary at {json_file_path}")
    plot_saved_info(folder, plot_ctx_end)


# Plots the LL with ctx and the SEM
def plot_saved_info(folder, plot_ctx_end):
    if not Path(folder).is_dir():
        print(f"No folder file found at {folder}")
        return
    json_path = folder + f'summary.json'
    trained_ctx_end = None
    if Path(json_path).exists():
        with open(json_path, 'r') as fileobj:
            metadata = json.load(fileobj)
        model_names = metadata['model_names']
        trained_ctx_end = metadata['trained_ctx_end']
        #folder = metadata['folder']
        task = metadata["data_task"]
    # Loads np arrays
    folder_path = Path(folder)
    npz_list = sorted(list(folder_path.glob("*.npz")))
    fig, ax = plt.subplots()
    for npz_file in npz_list:
        model_name = npz_file.stem
        data = np.load(npz_file)
        ll=data['ll'] # Shape should be [len(ctx), len(data)]
        no_runs = ll.shape[1]
        ctx = data['ctx']
        ll_sem = np.std(ll, axis=(1), ddof=1) / np.sqrt(no_runs)
        ll_mean = np.mean(ll, axis=(1))
        ax.plot(ctx, ll_mean, label=model_name)
        ax.fill_between(ctx, ll_mean - ll_sem, ll_mean + ll_sem, alpha=0.3, linewidth=0)
    if trained_ctx_end is not None and ctx[-1] > trained_ctx_end and plot_ctx_end:
        ax.axvline(x=trained_ctx_end, color='red', linestyle=':')
        ax.text(x=trained_ctx_end + 5, y=ax.get_ylim()[1] * 0.40, s='Max Trained NC', color='red', rotation=90, verticalalignment='top')
    ax.set_xlabel(r"$N_{s}$")
    ax.set_ylabel("Log Likelihood")
    ax.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.legend(frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.savefig(folder_path / "streaming_ll_ar.pdf", bbox_inches='tight')


# Gets file name
def file_name(task):
    key_map = {
        "ordering": "ord",
        "h_window": "hw",
        "delta_hours": "dh",
        "start_ctx": "sctx",
        "N_c_min": "nc",
        "N_c_max": "ncMx",
        "N_t_min": "nt",
        "N_t_max": "ntMx",
        "ctx_step": "step",
        "batch_size": "bs",
        "samples_per_epoch": "spe",
        "target_shuffle": "shuf"
    }
    parts = []
    for k, v in sorted(task.items()):
        short_key = key_map.get(k, k)
        parts.append(f"{short_key}{v}")
    filename_short = "_".join(parts)
    return filename_short


# All streaming tasks to perform
def streaming_tasks_list(samples_per_epoch=80_000, batch_size=32, ctx_step=1):
    spatiotemporal_infilling ={
        "h_window": 8,
        "delta_hours": 6,
        "start_ctx": 100,
        "N_c_min": 2000,
        "N_c_max": 2000,
        "N_t_min": 250,
        "N_t_max": 250,
        "ctx_step": ctx_step,
        "batch_size": batch_size,
        "samples_per_epoch": samples_per_epoch,
        "ordering": "ctx_time",
        "target_shuffle": "given",
    }
    forecasting ={
        "h_window": 8,
        "delta_hours": 6,
        "start_ctx": 100,
        "N_c_min": 2000,
        "N_c_max": 2000,
        "N_t_min": 250,
        "N_t_max": 250,
        "ctx_step": ctx_step,
        "batch_size": batch_size,
        "samples_per_epoch": samples_per_epoch,
        "ordering": "forecasting",
        "target_shuffle": "given"
    }
    spatiotemporal_infilling_500tgt ={
        "h_window": 8,
        "delta_hours": 6,
        "start_ctx": 100,
        "N_c_min": 2000,
        "N_c_max": 2000,
        "N_t_min": 500,
        "N_t_max": 500,
        "ctx_step": ctx_step,
        "batch_size": batch_size,
        "samples_per_epoch": samples_per_epoch,
        "ordering": "ctx_time",
        "target_shuffle": "given",
    }
    return [spatiotemporal_infilling, forecasting, spatiotemporal_infilling_500tgt]

# Gets list of models to measure ar runtime for
def get_model_list_runtime():
    # note we remove all W&B model weights to maintain annoymity but retain the code to showcase our methodology.
    folder = f"hadISDTemporal"
    tnp_plain = (f'experiments/configs/{folder}/hadtemp_tnp_plain.yml',
        '', 'TNP-D', True)
    incTNP = (f'experiments/configs/{folder}/hadtemp_incTNP.yml', 
        '', 'incTNP', True)
    batchedTNP = (f'experiments/configs/{folder}/hadtemp_incTNP_batched.yml',
        '', 'incTNP-Batched', False)
    cnp = (f'experiments/configs/{folder}/hadtemp_mcnp.yml',
        '', 'MCNP', False)
    lbanp_128 = (f"experiments/configs/{folder}/hadtemp_lbanp_128.yml",
             '', 'LBANP (128)', False)
    lbanp_256 = (f"experiments/configs/{folder}/hadtemp_lbanp_256.yml",
             '', 'LBANP (256)', False)
    tnpa_cache = (f"experiments/configs/{folder}/hadtemp_tnpa.yml",
             '', 'TNP-A (cache)', False)
    tnpa_fast = (f"experiments/configs/{folder}/hadtemp_tnpa.yml",
             '', 'TNP-A (fast)', False)
    tnpa_normal = (f"experiments/configs/{folder}/hadtemp_tnpa.yml",
             '', 'TNP-A (normal)', False)
    models = [tnp_plain, batchedTNP , incTNP, lbanp_256, cnp, tnpa_cache]
    return models

def get_model_list_perf(ordering_loader):
    assert ordering_loader in ("random", "ctx_time", "full_time", "forecasting"), "Invalid time ordering loader"
    folder = f"hadISDTemporal"
    tnp_plain = (f'experiments/configs/{folder}/hadtemp_tnp_plain.yml',
        '', 'TNP-D', True)
    incTNP = (f'experiments/configs/{folder}/hadtemp_incTNP.yml', 
        '', 'incTNP', True)
    batchedTNP = (f'experiments/configs/{folder}/hadtemp_incTNP_batched.yml',
        '', 'incTNP-Batched', False)
    cnp = (f'experiments/configs/{folder}/hadtemp_mcnp.yml',
        '', 'MCNP', False)
    lbanp = (f"experiments/configs/{folder}/hadtemp_lbanp_256.yml",
             '', 'LBANP', False)
    tnpa = (f"experiments/configs/{folder}/hadtemp_tnpa.yml",
             '', 'TNP-A', False)
    models = [batchedTNP, tnpa, tnp_plain, incTNP, cnp, lbanp]
    return models

# Generates the computational streaming ar plots
def generate_all_computational_ar_main():
    # Hyper parameters
    batch_size = 32
    device = "cuda"
    trained_ctx_end = 2_100 # Max N_c trained with
    base_folder = "experiments/plot_results/ar_streaming_time_main/"
    run_experiments = True
    plot_ctx_end = False
    aggregate_over=2
    token_step = 50
    min_nc = 1
    max_nc = 2_000
    nt = 250
    dx = 4
    dy = 1
    num_samples = 50
    use_flash=True
    cache_mhca=False
    order = "random"
    run_mode = "normal_cast"
    # End of hypers
    os.makedirs(base_folder, exist_ok=True)
    task_folder = base_folder + f"m_{batch_size}_cstep_{token_step}_ag_{aggregate_over}_minc_{min_nc}_maxc_{max_nc}_nt_{nt}_dx{dx}_dy{dy}_S_{num_samples}_run_{run_mode}_cmhca_{cache_mhca}_flsh_{use_flash}/"
    model_list = get_model_list_runtime()
    measure_streaming_ar_timings(models=model_list, aggregate_over=aggregate_over, token_step=token_step, min_nc=min_nc, max_nc=max_nc, dx=dx, dy=dy, num_samples=num_samples, device=device, cache_mhca=cache_mhca, folder=task_folder, use_flash=use_flash, run_experiments=run_experiments, m=batch_size, nt=nt, run_mode=run_mode, order=order, plot_ctx_end=plot_ctx_end, trained_ctx_end=trained_ctx_end)

# Generates streaming ar plots we want
def generate_all_streaming_ar_main():
    # Hyper parameters
    samples_per_epoch = 2_000
    batch_size = 32
    device = "cuda"
    seed = 1
    trained_ctx_end = 2_100 # Max N_c trained with
    ctx_step = 50
    base_folder = "experiments/plot_results/ar_streaming/"
    run_experiments = True
    plot_ctx_end = False
    # End of hypers
    os.makedirs(base_folder, exist_ok=True)
    tasks = streaming_tasks_list(samples_per_epoch=samples_per_epoch, batch_size=batch_size, ctx_step=ctx_step)
    for task in tasks:
        task_folder = f"{base_folder}{file_name(task)}/"
        model_list = get_model_list_perf(ordering_loader=task["ordering"])
        streaming_ar_performance(models=model_list, task=task, device=device, seed=seed, folder=task_folder, trained_ctx_end=trained_ctx_end, run_experiments=run_experiments, plot_ctx_end=plot_ctx_end)

#if __name__ == "__main__":
    #generate_all_streaming_ar_main()
    #generate_all_computational_ar_main()
