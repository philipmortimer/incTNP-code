# PLots performance of incTNP vs TNP-A to show how lack of incremental updating makes it non-viable
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

# Measuring the runtime (y axis) and memory (y axis) for AR streaming approaches vs the streamed context size (x axis)
@torch.no_grad()
def measure_streaming_ar_timings(phase, aggregate_over, dx, dy, num_samples, device, cache_mhca, folder, use_flash, run_experiments, m, nt, run_mode, order, base_folder, phases_list):
    os.makedirs(folder, exist_ok=True)
    json_file_path = folder + f'summary.json'
    assert order == "random", "order should be random for AR"
    assert run_mode in ("normal", "flash", "normal_cast"), "run mode must be specified"
    if run_experiments:
        # Hypers that should be fixed
        prioritise_fixed = False
        device_ret = "cpu"
        cast_model_setting = False # We already cast ourselves
        min_nc, max_nc = phase["start_nc"], phase["end_nc"]
        token_step = phase["token_step"]
        models = phase["models"]
        xc = (torch.rand((m, max_nc, dx), device=device) * 2) - 1 # Only need to use tensors of correct SHAPE when measuring computation costs, dont need data loader
        yc = (torch.rand((m, max_nc, dy), device=device)* 2) -1
        xt = (torch.rand((m, nt, dx), device=device) * 2) -1
        yt = (torch.rand((m, nt, dy), device=device)* 2) -1
        xc_empty = torch.empty((m, 0, dx), device=device) # Used for zero ctx conditioning
        yc_empty = torch.empty((m, 0, dy), device=device)
        last_conditioned_ctx_size_inc = 0 # We have conditioned on nothing yet
        cast_ctx_manager = nullcontext() if run_mode  == "normal" else torch.autocast(device_type=device, dtype=torch.bfloat16)
        ctx_sizes = [i for i in range(min_nc, max_nc, token_step)]#np.arange(start=min_nc, stop=max_nc, step=token_step, dtype=int)
        if max(ctx_sizes) != max_nc: ctx_sizes.append(max_nc)
        ctx_sizes = np.array(ctx_sizes)
        runtime = np.zeros((len(models), aggregate_over, len(ctx_sizes)))
        condition_time = np.full((len(models), aggregate_over, len(ctx_sizes)), np.nan) # nan conditioning time indicates that model does not support conditioning
        memory = np.zeros((len(models), aggregate_over, len(ctx_sizes)))
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        for model_idx, (model_yml, model_wab, model_name, weights_only_evalhad_call) in enumerate(models):
            file_name_npz = f'{model_name}.npz'
            npz_arr_path = folder + file_name_npz
            if Path(npz_arr_path).exists(): 
                stitch_and_plot_results(base_folder, phases_list) # Plots after each loop for quicker iteration
                continue # No need to redo computation for model thats already been computed - this assumes the setups are the same
            for i in range(aggregate_over):
                model = get_model(model_yml, model_wab, seed=False, device=device, weights_only_evalhad_call=weights_only_evalhad_call)
                # As a runtime test we want to ensure the sampling doesnt underflow / overflow
                model.likelihood.runtime_clamp = True
                model.likelihood.min_noise = 1.0e-4
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
                            # Update conditioning set and measure time of conditioning for last update as conditioning time
                            for j in range(last_conditioned_ctx_size_inc, nc - 1):
                                model.update_ctx(xc=xc[:, j:j+1, :],yc=yc[:, j:j+1, :],use_flash=use_flash,cache_mhca=cache_mhca,persist_small=True,)
                            # Times conditioning step cost
                            torch.cuda.synchronize()
                            starter.record()
                            model.update_ctx(xc=xc[:,nc-1:nc,:],yc=yc[:, nc-1:nc, :],use_flash=use_flash,cache_mhca=cache_mhca,persist_small=True,)
                            ender.record()
                            torch.cuda.synchronize()
                            condition_ms = starter.elapsed_time(ender)
                            condition_time[model_idx, i, t_index]  = condition_ms
                            # AR function
                            ar_pred_fn = lambda: ar_predict(model=model, xc=xc_empty, yc=yc_empty, xt=xt, order=order, num_samples=num_samples, prioritise_fixed=prioritise_fixed, device=device, device_ret=device_ret, use_flash=use_flash, run_mode=run_mode, cast_model=cast_model_setting, cache_mhca=cache_mhca, persist_small=True)
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
            # Writes results to file
            np.savez(npz_arr_path, runtime=runtime[model_idx], memory=memory[model_idx], ctx_size=ctx_sizes, condition_time=condition_time[model_idx])
            del model
            gc.collect()
            torch.cuda.empty_cache()
            stitch_and_plot_results(base_folder, phases_list) # Plots after each loop for quicker iteration
        print(f"Phase done max_nc={max_nc}")
    stitch_and_plot_results(base_folder, phases_list)

def stitch_and_plot_results(experiment_folder, phases):
    base_path = Path(experiment_folder)
    model_data = {}

    # gathers all phase info
    for p_name in phases:
        p_path = base_path / p_name
        if not p_path.exists(): continue
        
        for f in p_path.glob("*.npz"):
            m_name = f.stem
            if m_name not in model_data:
                model_data[m_name] = {'ctx': [], 'run': [], 'cond': [], 'mem': []}
            
            dat = np.load(f)
            model_data[m_name]['ctx'].append(dat['ctx_size'])
            model_data[m_name]['run'].append(np.mean(dat['runtime'], axis=0)) # Average runs
            model_data[m_name]['mem'].append(np.mean(dat['memory'], axis=0))
            
            c_time = dat['condition_time']
            if c_time.size > 0 and not np.isnan(c_time).all():
                model_data[m_name]['cond'].append(np.mean(c_time, axis=0))
            else:
                model_data[m_name]['cond'].append(np.zeros(dat['ctx_size'].shape))

    
    fig, ax = plt.subplots()
    fig_cum, ax_cum = plt.subplots()
    colors = plt.cm.tab10(np.linspace(0, 1, len(model_data)))
    for i, (m_name, data) in enumerate(model_data.items()):
        if not data['ctx']: continue
        ctx = np.concatenate(data['ctx'])
        run = np.concatenate(data['run']) / 1000.0
        # Handle conditioning
        cond_list = data['cond']
        if len(cond_list) > 0:
            cond = np.concatenate(cond_list) / 1000.0
        else:
            cond = np.zeros_like(run)
        sort_idx = np.argsort(ctx)
        ctx = ctx[sort_idx]
        run = run[sort_idx]
        cond = cond[sort_idx]
        total_cost = run + cond
        # Caclulates cumulative cost - apprxoimates between steps
        run_cum = np.zeros_like(run)
        cond_cum = np.zeros_like(cond)
        for j in range(1, len(ctx)):
            dt_steps = ctx[j] - ctx[j-1]
            avg_latency_run = (run[j] + run[j-1]) / 2.0
            run_cum[j] = run_cum[j-1] + (avg_latency_run * dt_steps)
            avg_latency_cond = (cond[j] + cond[j-1]) / 2.0
            cond_cum[j] = cond_cum[j-1] + (avg_latency_cond * dt_steps)
        total_cum = run_cum + cond_cum
        color = colors[i]
        
        ax.plot(ctx, total_cost, label=f"{m_name} (Total)", color=color, linestyle='-', alpha=0.9)
        if np.max(cond) > 0.001:
            ax.plot(ctx, cond, label=f"{m_name} (Update)", color=color, linestyle=':', alpha=0.7)

        ax_cum.plot(ctx, total_cum, label=f"{m_name} (Total)", color=color, linestyle='-', alpha=0.9)
        if np.max(cond_cum) > 0.001:
            ax_cum.plot(ctx, cond_cum, label=f"{m_name} (Update)", color=color, linestyle=':', alpha=0.7)

    for ax_i, fig_i, nm in [(ax, fig, "streaming"), (ax_cum, fig_cum, "cumulative")]:
        ax_i.set_ylabel("Time (s)")
        ax_i.set_xlabel(r"Stream Length ($N_{s}$)")
        ax_i.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.7)
        ax_i.legend()
        ax_i.legend(frameon=False)
        out_path = base_path / f"{nm}.pdf"
        fig_i.savefig(out_path, bbox_inches='tight')
        print(f"Saved plot: {out_path}")
    plt.close(fig)
    plt.close(fig_cum)


# Stores the phases to run
def model_phases():
    # List of models
    folder = f"hadISDTemporal"
    tnpa_cache = (f"experiments/configs/{folder}/hadtemp_tnpa.yml",
             'W&BLINKREMOVEDFORANNOYMICML', 'TNP-A (cache)', False)
    tnpa_fast = (f"experiments/configs/{folder}/hadtemp_tnpa.yml",
             'BLINKREMOVEDFORANNOYMICML', 'TNP-A (fast)', False)
    tnpa_normal = (f"experiments/configs/{folder}/hadtemp_tnpa.yml",
             'BLINKREMOVEDFORANNOYMICML', 'TNP-A (normal)', False)
    incTNP = (f'experiments/configs/{folder}/hadtemp_incTNP.yml', 
        'BLINKREMOVEDFORANNOYMICML', 'incTNP', True)
    tnp_plain = (f'experiments/configs/{folder}/hadtemp_tnp_plain.yml',
        'BLINKREMOVEDFORANNOYMICML', 'TNP-D', True)
    # End of model list
    # Defines computation phases to showcase performance
    phase1 = {
        "start_nc": 1,
        "end_nc": 5_000,
        "token_step": 500,
        "models": [tnp_plain, tnpa_normal, tnpa_fast, tnpa_cache, incTNP],
    }
    phase2 = {
        "start_nc": 5_001,
        "end_nc": 10_000,
        "token_step": 1_000,
        "models": [tnp_plain, tnpa_cache, incTNP]
    }
    phase3 = {
        "start_nc": 10_001,
        "end_nc": 100_000,
        "token_step": 5_000,
        "models": [tnpa_cache, incTNP]
    }
    phases = [phase1, phase2, phase3]
    return phases

# Generates the computational streaming ar plots
def generate_all_computational_ar_main():
    # Hyper parametersS
    batch_size = 1
    device = "cuda"
    base_folder = "experiments/plot_results/ar_tnpatime/full/"
    run_experiments = True
    nt = 250
    dx = 4
    dy = 1
    num_samples = 50
    use_flash=True
    cache_mhca=False
    order = "random"
    run_mode = "normal_cast"
    aggregate_over=1
    # End of hypers
    os.makedirs(base_folder, exist_ok=True)
    phase_list = model_phases()
    phase_names = [f"Phase{i}/" for i in range(len(phase_list))]
    for i, phase in enumerate(phase_list):
        task_folder = base_folder + phase_names[i]
        os.makedirs(task_folder, exist_ok=True)
        measure_streaming_ar_timings(phase=phase, aggregate_over=aggregate_over, dx=dx, dy=dy, num_samples=num_samples, device=device, cache_mhca=cache_mhca, folder=task_folder, use_flash=use_flash, run_experiments=run_experiments, m=batch_size, nt=nt, run_mode=run_mode, order=order, base_folder=base_folder, phases_list=phase_names)


#if __name__ == "__main__":
    #generate_all_computational_ar_main()
