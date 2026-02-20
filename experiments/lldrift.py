# Plots Log likelihood over time with a kernel that drifts inspired partially by kernel from https://proceedings.mlr.press/v51/herlands16.pdf
from tnp.data.gp import RandomScaleGPGenerator, ChangeKernelGPGenerator, GPRegressionModel
from tnp.networks.gp import RBFKernel, MaternKernel, PeriodicKernel
from functools import partial
from plot_adversarial_perms import get_model
import numpy as np
from tnp.models.incUpdateBase import IncUpdateEff
from tnp.data.base import Batch
from tnp.utils.np_functions import np_pred_fn
import os
import torch
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm
from arnp import ar_loglik
import json
import gpytorch



def get_change_data(kernel_factories, nc: int, nt: int, batch_size: int, kernel_name: str,
    t0: int, tau: float):
    min_nc = nc
    max_nc = nc
    context_range = [[-2.0, 2.0]]
    target_range = [[-2.0, 2.0]]
    samples_per_epoch = 4_096
    batch_size = batch_size
    noise_std = 0.1
    deterministic = True
    gen_test = ChangeKernelGPGenerator(dim=1, min_nc=min_nc, max_nc=max_nc, min_nt=nt, max_nt=nt, batch_size=batch_size,
        context_range=context_range, target_range=target_range, samples_per_epoch=samples_per_epoch, noise_std=noise_std,
        deterministic=deterministic, kernels=tuple(kernel_factories), t0=t0, tau=tau)
    data = list(gen_test)
    return data, kernel_name


def get_rbf_change_data(nc: int, nt: int, batch_size: int, t0: int, tau: float):
    ard_num_dims=1
    min_log10_lengthscale = -0.602
    max_log10_lengthscale = 0.0
    rbf_kernel_factory = partial(RBFKernel, ard_num_dims=ard_num_dims, min_log10_lengthscale=min_log10_lengthscale,
                         max_log10_lengthscale=max_log10_lengthscale)
    kernels = [rbf_kernel_factory]
    return get_change_data(kernel_factories=kernels, nc=nc, nt=nt, batch_size=batch_size, kernel_name="RBF Kernel",
        t0=t0, tau=tau)

def get_combined_change_data(nc: int, nt: int, batch_size: int, t0: int, tau: float):
    ard_num_dims=1
    min_log10_lengthscale = -0.602
    max_log10_lengthscale = 0.0
    min_log10_period = 0.301
    max_log10_period = 0.301
    rbf_kernel_factory = partial(RBFKernel, ard_num_dims=ard_num_dims, min_log10_lengthscale=min_log10_lengthscale,
                         max_log10_lengthscale=max_log10_lengthscale)
    matern12_kernel_factory = partial(MaternKernel, nu=0.5, ard_num_dims=ard_num_dims, min_log10_lengthscale=min_log10_lengthscale,
                         max_log10_lengthscale=max_log10_lengthscale)
    matern32_kernel_factory = partial(MaternKernel, nu=1.5, ard_num_dims=ard_num_dims, min_log10_lengthscale=min_log10_lengthscale,
                         max_log10_lengthscale=max_log10_lengthscale)
    matern52_kernel_factory = partial(MaternKernel, nu=2.5, ard_num_dims=ard_num_dims, min_log10_lengthscale=min_log10_lengthscale,
                         max_log10_lengthscale=max_log10_lengthscale)
    periodic_kernel_factory = partial(PeriodicKernel, min_log10_period=min_log10_period, max_log10_period=max_log10_period, 
            ard_num_dims=ard_num_dims, min_log10_lengthscale=min_log10_lengthscale, max_log10_lengthscale=max_log10_lengthscale)
    kernels = [rbf_kernel_factory, matern12_kernel_factory, matern32_kernel_factory, matern52_kernel_factory, periodic_kernel_factory]
    return get_change_data(kernel_factories=kernels, nc=nc, nt=nt, batch_size=batch_size, kernel_name="Combined Kernel",
        t0=t0, tau=tau)

def get_model_list_combined():
    # List of models to compare trained on combined kernel - removed due to W&B links breaking annoynm requirements
    return []

def get_model_list_rbf():
    # List of models to compare trained on rbf kernel - removed due to W&B links breaking annoynm requirements
    tnp_plain = ('experiments/configs/synthetic1dRBF/gp_plain_tnp_rangesame.yml',
        '', 'TNP-D', False)
    # More models defined here in og code with W&B links
    return []


def drift_stream_data_test_rbf():
    # Hypers
    t_0_vals = [2, 10, 20, 65, 75, 100, 200, 400]
    tau_vals = [0, 0.5, 1.0, 2.0, 4.0, 5.0, 7.5, 10.0, 15.0, 20.0]
    t_0_vals = [20]
    tau_vals=[0, 5.0, 10.0, 15.0, 20.0]

    t0_tau_list = [(t0, tau) for t0 in t_0_vals for tau in tau_vals]
    print(t0_tau_list)
    burn_in = 0
    aggregate_over = 1
    batch_size = 16
    max_batches = None # Set to None for no limit
    max_nc = 500
    nt = 128
    start_ctx = 1
    end_ctx = max_nc
    ctx_step = 1
    trained_ctx_end = 64
    device="cuda"
    folder = "experiments/plot_results/lldrift/rbf/"
    # End of hypers
    for t0, tau in t0_tau_list:
        drift_stream_data_test(get_rbf_change_data(max_nc, nt, batch_size, t0, tau), get_model_list_rbf(), max_nc, nt, start_ctx, end_ctx, ctx_step, device, folder, trained_ctx_end, max_batches, burn_in, aggregate_over, t0, tau)

def drift_stream_data_test_combined():
    # Hypers
    t_0_vals = [2, 10, 20, 65, 75, 100, 200, 400]
    tau_vals = [0, 0.5, 1.0, 2.0, 4.0, 5.0, 7.5, 10.0, 15.0, 20.0]

    t_0_vals = [20]
    tau_vals = [10.0]

    t0_tau_list = [(t0, tau) for t0 in t_0_vals for tau in tau_vals]
    print(t0_tau_list)
    burn_in = 0
    aggregate_over = 1
    batch_size = 16
    max_batches = None # Set to None for no limit
    max_nc = 500
    nt = 128
    start_ctx = 1
    end_ctx = max_nc
    ctx_step = 1
    trained_ctx_end = 64
    device="cuda"
    folder = "experiments/plot_results/lldrift/combined/"
    # End of hypers
    for t0, tau in t0_tau_list:
        drift_stream_data_test(get_combined_change_data(max_nc, nt, batch_size, t0, tau), get_model_list_combined(), max_nc, nt, start_ctx, end_ctx, ctx_step, device, folder, trained_ctx_end, max_batches, burn_in, aggregate_over, t0, tau)


@torch.no_grad
def drift_stream_data_test(dataset, models, max_nc, nt, start_ctx, end_ctx, ctx_step, device, folder, trained_ctx_end, max_batches, burn_in, aggregate_over, t0, tau):
    data, kernel_name = dataset
    ctx = list(range(start_ctx, end_ctx, ctx_step))
    ctx.append(end_ctx)
    ctx = np.array(ctx)
    ll_list = np.zeros((len(models), len(ctx), len(data), aggregate_over))
    gt_ll_curve = np.zeros((len(ctx), len(data)))
    condition_time_list = np.zeros((len(models), len(ctx), len(data), aggregate_over))
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    gt_lls = np.zeros(len(data))

    # Caches expensive moving ll calcs to save O(model) repeats - could do this one level higher at data level to save calls for different runs with same data
    yt_curr_cache = [[None for _ in range(len(ctx))] for _ in range(len(data))] # Empty 2d list
    for batch_idx, batch in tqdm(enumerate(data), desc="Generating GP data"):
        if max_batches is not None and batch_idx >= max_batches: break
        # Moves batch to gpu
        batch.xc, batch.yc, batch.xt, batch.yt = batch.xc.to(device), batch.yc.to(device), batch.xt.to(device), batch.yt.to(device)
        xc_ll, yc_ll, xt_ll, yt_ll = batch.xc, batch.yc, batch.xt, batch.yt
        xc, xt, yc, yt = xc_ll[..., :-1], xt_ll[..., :-1], yc_ll, yt_ll
        m, nt, _ = yt.shape

        for ctx_idx, ctx_upper in enumerate(ctx):
            batch.gt_pred._result_cache = None # Clears cache - slow but now used for changing kernel

            t_stamp = torch.full_like(xt[..., :1], ctx_upper, device=device)
            xt_curr = torch.cat([xt, t_stamp], dim=-1)

            xc_sub, yc_sub = xc_ll[:, :ctx_upper, :], yc_ll[:, :ctx_upper, :]

            yt_samples = []
            ll_sum = 0
            for task in range(m):
                gp_model = GPRegressionModel(likelihood=batch.gt_pred.likelihood, kernel=batch.gt_pred.kernel, 
                    train_inputs=xc_sub[task], train_targets=yc_sub[task, ..., 0]).to(device)
                gp_model.eval()
                gp_model.likelihood.eval()
                with gpytorch.settings.fast_pred_var(True):
                    dist_task = gp_model.likelihood.marginal(gp_model(xt_curr[task])) 

                y_task = dist_task.rsample()
                yt_samples.append(y_task.unsqueeze(0))
                ll_sum += dist_task.log_prob(y_task).sum().item()
            yt_curr = torch.cat(yt_samples, dim=0).unsqueeze(-1)
            gt_ll_curve[ctx_idx, batch_idx] = ll_sum / (m * nt)
            dy = yt_curr.shape[-1]
            # Caches results
            yt_curr_cache[batch_idx][ctx_idx] = yt_curr

    # Main model calculation loop
    for model_idx, (model_yml, model_wab, model_name, use_ar) in enumerate(models):
        model = get_model(model_yml, model_wab, seed=False, device=device)
        model.eval()
        is_model_inc = isinstance(model, IncUpdateEff)
        for batch_idx, batch in tqdm(enumerate(data), desc=f'{model_name}'):
            if max_batches is not None and batch_idx >= max_batches: break
            # Moves batch to gpu
            batch.xc, batch.yc, batch.xt, batch.yt = batch.xc.to(device), batch.yc.to(device), batch.xt.to(device), batch.yt.to(device)
            xc_ll, yc_ll, xt_ll, yt_ll = batch.xc, batch.yc, batch.xt, batch.yt
            xc, xt, yc, yt = xc_ll[..., :-1], xt_ll[..., :-1], yc_ll, yt_ll
            m, nt, _ = yt.shape
            if is_model_inc: model.init_inc_structs(m=m, max_nc=max_nc, device=device)
            for ctx_idx, ctx_upper in enumerate(ctx):
                yt_curr = yt_curr_cache[batch_idx][ctx_idx]

                # Does model burn in and aggregation
                for j in range(burn_in + aggregate_over):
                    ctx_lower = 0 if ctx_idx == 0 else ctx[ctx_idx - 1]
                    if use_ar:
                        xc_new, yc_new = xc[:, :ctx_upper, :], yc[:, :ctx_upper, :]
                        # Times LL but this doesnt really make sense for a timing as is teacher forcing
                        torch.cuda.synchronize()
                        starter.record()
                        with torch.no_grad():
                            loglik = ar_loglik(np_model=model, xc=xc_new, yc=yc_new, xt=xt, yt=yt_curr, 
                                           normalise=True, order="random").mean().item()
                        ender.record()
                        torch.cuda.synchronize()
                        runtime_ms = starter.elapsed_time(ender)
                    elif is_model_inc:
                        xc_new, yc_new = xc[:, ctx_lower:ctx_upper, :], yc[:, ctx_lower:ctx_upper, :]
                        # Time the conditioning phase
                        torch.cuda.synchronize()
                        starter.record()
                        with torch.no_grad():
                            model.update_ctx(xc=xc_new, yc=yc_new)
                        ender.record()
                        torch.cuda.synchronize()
                        runtime_ms = starter.elapsed_time(ender)
                        # Gets predictive distribution
                        pred_dist = model.query(xt=xt, dy=dy)
                        loglik = (pred_dist.log_prob(yt_curr).sum() / yt_curr[..., 0].numel()).item()
                    else:
                        xc_new, yc_new = xc[:, :ctx_upper, :], yc[:, :ctx_upper, :]
                        batch_np = Batch(xc=xc_new, yc=yc_new, xt=xt, yt=yt_curr, x=None, y=None)
                        # Times whole prediction and treats it as conditioning cost
                        torch.cuda.synchronize()
                        starter.record()
                        with torch.no_grad():
                            pred_dist = np_pred_fn(model, batch_np, predict_without_yt_tnpa=False) # uses teacher forcing if possible
                        ender.record()
                        torch.cuda.synchronize()
                        runtime_ms = starter.elapsed_time(ender)
                        loglik = (pred_dist.log_prob(yt_curr).sum() / yt_curr[..., 0].numel()).item()
                    # Records likelihood and runtime
                    write_idx = j - burn_in
                    if write_idx >= 0:
                        ll_list[model_idx, ctx_idx, batch_idx, write_idx] = loglik
                        condition_time_list[model_idx, ctx_idx, batch_idx, write_idx] = runtime_ms
    # Averages over batches
    ll_list = np.mean(ll_list, axis=(2, 3))
    gt_average_curve = gt_ll_curve.mean(axis=1)
    gt_average = np.mean(gt_average_curve, axis=0)
    condition_time_list = np.mean(condition_time_list, axis=(2, 3))

    # Saves data to output file to be used when plotting
    file_name_npz = f'npz_kernel_{kernel_name}_t0_{t0}_tau_{tau}_maxnc_{max_nc}_ctxstep_{ctx_step}.npz'
    npz_arr_path = folder + file_name_npz
    np.savez(npz_arr_path, ll=ll_list, ctx=ctx, time=condition_time_list, gt_ll_curve=gt_average_curve)
    json_file_path = folder + f'json_{kernel_name}_t0_{t0}_tau_{tau}_maxnc_{max_nc}_ctxstep_{ctx_step}.json'
    summary_meta = {
        "model_names": [m[2] for m in models],
        "trained_ctx_end": trained_ctx_end,
        "gt_average_ll": gt_average,
        "kernel_name": kernel_name,
        "npz_path": npz_arr_path,
        "folder": folder,
        "t0": t0,
        "tau": tau,
    }
    with open(json_file_path, 'w') as fileobj:
        json.dump(summary_meta, fileobj, indent=4)
    print(f"Summary at {json_file_path}")
    
    plot_saved_info_drift(json_file_path)


def plot_saved_info_drift(json_path):
    with open(json_path, 'r') as fileobj:
        metadata = json.load(fileobj)
    model_names = metadata['model_names']
    trained_ctx_end = metadata['trained_ctx_end']
    gt_average = metadata['gt_average_ll']
    kernel_name = metadata['kernel_name']
    folder = metadata['folder']
    npz_path = metadata['npz_path']
    t0 = metadata['t0']
    tau = metadata['tau']
    # Loads np arrays
    data = np.load(npz_path)
    ll_list = data['ll']
    condition_time_list = data['time']
    ctx = data['ctx']
    gt_ll_curve = data['gt_ll_curve'] if 'gt_ll_curve' in data.files else None
    max_nc = ctx.max()

    # Plots LL as context size increases - red dotted line to show when going beyond trained context size
    ll_file_name = folder + f'll_kernel_{kernel_name}_t0_{t0}_tau_{tau}_maxnc_{max_nc}.png'
    fig, ax = plt.subplots(figsize=(7, 5))
    

    #ax.set_xlim(0, 49)
    #ax.set_ylim(np.min(ll_list[:, :49]), max(np.max(ll_list[:, :49]), np.max(gt_ll_curve)))

    # Plots LLs
    for model_idx, model_name in enumerate(model_names):
        ax.plot(ctx, ll_list[model_idx], label=model_name)
    if True:
        if gt_ll_curve is not None: ax.plot(ctx, gt_ll_curve, color='grey', linestyle='--', linewidth=2, label='GT LL') 
        #ax.axhline(y=gt_average, color='grey', linestyle='--', label='Mean GT LL')
    ax.axvline(x=t0, color='red', linestyle=':', label='t0')
    #ax.text(x=trained_ctx_end + 5, y=ax.get_ylim()[1] * 0.40, s='t0', color='red', rotation=90, verticalalignment='top')

    # Plots gradient background to show mixing of kernels - until the 95 % interval
    start_colour = (0.85, 0.85, 0.85) # Initial colour to go from
    end_colour = (1.0, 0.2, 0.2) # Colour to fade to
    x_grid = np.linspace(ctx.min(), ctx.max(), 400) # Set more samples for better resolution
    if tau == 0:
         w = (x_grid >= t0).astype(float) # Hard shift between two kernels
         left_5_bound, right_95_bound = t0, t0
    else: 
        w = 1 / (1 + np.exp(-(x_grid - t0) / tau)) # Sigmoid changing
        left_5_bound = t0 + tau * np.log(0.05 / 0.95)
        right_95_bound = t0 + tau * np.log(0.95 / 0.05)
        mix_col = np.outer(1 - w, start_colour) + np.outer(w, end_colour) # belnds colours over interval
        img = np.repeat(mix_col[np.newaxis, :, :], 2, axis=0)
        y0, y1 = ax.get_ylim()
        ax.imshow(img, extent=[left_5_bound, right_95_bound, y0, y1], aspect='auto', origin='lower', alpha=0.35, zorder=-2)
    ax.axvspan(ctx.min(), left_5_bound, color=start_colour, alpha=0.20, zorder=-3)
    ax.axvspan(right_95_bound, ctx.max(), color=end_colour, alpha=0.20, zorder=-3)

    ax.set_xlabel('Number of Context Points')
    ax.set_ylabel('Mean Log-Likelihood')
    ax.legend()
    ax.set_title(rf'Streamed Performance on {kernel_name} ($\tau={tau}$)')
    ax.grid(True, linestyle='--', alpha=0.4)
    fig.tight_layout()
    plt.savefig(ll_file_name, dpi=300)

    # Plots conditioning time vs number of context points
    runtime_file_name = folder + f'runtime_kernel_{kernel_name}_t0_{t0}_tau_{tau}_maxnc_{max_nc}.png'
    fig, ax = plt.subplots(figsize=(7, 5))
    for model_idx, model_name in enumerate(model_names):
        ax.plot(ctx, condition_time_list[model_idx], label=model_name)
    ax.set_xlabel('Number of Context Points')
    ax.set_ylabel('Mean Conditioning Time (ms)')
    ax.legend()
    ax.set_title(f'Conditioning Time of NP Models')
    ax.grid(True, linestyle='--', alpha=0.4)
    fig.tight_layout()
    plt.savefig(runtime_file_name, dpi=300)


#if __name__ == "__main__":
#    drift_stream_data_test_rbf()
#    drift_stream_data_test_combined()