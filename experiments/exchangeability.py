# Measure of implicit bayesianess of NP models
import copy
import math
import numpy as np
import torch
from scipy import stats
from check_shapes import check_shapes
from tnp.utils.experiment_utils import initialize_experiment
from tnp.utils.data_loading import adjust_num_batches
from tnp.utils.lightning_utils import LitWrapper
import time
import warnings
from tnp.data.gp import RandomScaleGPGenerator
from tnp.networks.gp import RBFKernel
from functools import partial
import wandb
import os
from typing import Optional
from plot_adversarial_perms import get_model
import matplotlib
import matplotlib.pyplot as plt
from itertools import cycle
import random
from tnp.utils.np_functions import np_pred_fn
from tnp.data.base import Batch
from matplotlib.ticker import LogFormatterMathtext
from tnp.models.tnpa import TNPA
import gpytorch
import torch.distributions as td
from pathlib import Path
import re
from tnp.models.wiskigp import ExchangeCalcWiskiRBF, WiskiGP
from tueplots import bundles
from online_gp.utils.cuda import try_cuda


plt.rcParams.update(bundles.icml2024())


# Computes log joint variance of model - use Eq 5 but only for a fixed target and context set
@check_shapes(
    "x_fix: [m, n_fix, dx]", "y_fix: [m, n_fix, dy]", "x_eval: [m, n_eval, dx]", "y_eval: [m, n_eval, dy]"
)
def m_var_fixed(tnp_model, x_fix: torch.Tensor, y_fix: torch.Tensor, x_eval: torch.Tensor, y_eval: torch.Tensor,
    gt_pred,
    monte_carlo_samples: int, # S (number of monte carlo samples to approximate KL over)
    sub_batch_size: Optional[int], # Maximum chunk size to prevent OOM
    n: int, init: int, b: int, # b is AR block size
    number_of_mog_permutations: int, # G number of joints to use to apporximate the bayesian decision rule
    joints_average_over: int, # J, number of joints to use to compute mean KL between joint and MoG
    use_torch_grad: bool = False,
    set_wiski_params_gt: bool =  False, # If using wiski model, do we set kernel to mimic groundtruth kernel
    ):
    with torch.set_grad_enabled(use_torch_grad):
        S = monte_carlo_samples
        G = number_of_mog_permutations
        J = joints_average_over

        # Computes ground truth nll measure (as mentioned in caption of figure 2)
        _, _, gt_loglik = gt_pred.get_joint_loglik_fixed_context(
            x_fix=x_fix,
            y_fix=y_fix,
            x_eval=x_eval,
            y_eval=y_eval,
        ) # [m]
        gt_nll = -gt_loglik.to(x_fix.device)
        gt_nll = gt_nll.unsqueeze(0) # [1, m] - allows for broacasting when we subtract later on

        def log_prob_targets(dist, y_targets):
            # Sum over Last two dims (nt and dy)
            return dist.log_prob(y_targets).sum(dim=(-1, -2))

        if set_wiski_params_gt: tnp_model.update_covar_params(gt_pred)

        _, n_eval, _ = x_eval.shape
        _, n_fix, dy = y_fix.shape
        m, _, dx = x_fix.shape
        assert n == n_eval and init == n_fix, "Clashing dimensions between batch and specified split"
        assert b >= 1 and b <= n_eval, "Invalid block size b"

        # Pretrains wiski on fixed to reduce compute cost (too expensive to do retraining every time with wiskis for pretraining step)
        is_wiski = isinstance(tnp_model, WiskiGP)
        if is_wiski: 
            print("Pretraining start")
            tnp_model.pretrain_on_fixed(x_fix, y_fix, x_eval, y_eval)
            print("Pretraining done")

        # -----------------------------------------------------------------------------------------
        # Builds the MoG components (teacher forced) + teacher-forced MoG components (for y-axis)
        # -----------------------------------------------------------------------------------------

        perms_G = torch.stack([torch.randperm(n_eval, device=x_fix.device) for _ in range(G)])
        # Expands to G*m batch
        x_eval_exp = x_eval.unsqueeze(0).expand(G, -1, -1, -1).reshape(G * m, n_eval, dx)
        y_eval_exp = y_eval.unsqueeze(0).expand(G, -1, -1, -1).reshape(G * m, n_eval, dy)
        perms_exp = perms_G.unsqueeze(1).expand(-1, m, -1).reshape(G * m, n_eval)
        inv_perms_exp = torch.argsort(perms_exp, dim=1)
        x_fix_exp = x_fix.unsqueeze(0).expand(G, -1, -1, -1).reshape(G * m, n_fix, dx)
        y_fix_exp = y_fix.unsqueeze(0).expand(G, -1, -1, -1).reshape(G * m, n_fix, dy)

        # Permutes the ordering of points
        gather_idx_x = perms_exp.unsqueeze(-1).expand(-1, -1, dx)
        gather_idx_y = perms_exp.unsqueeze(-1).expand(-1, -1, dy)
        x_eval_permuted = torch.gather(x_eval_exp, 1, gather_idx_x)
        y_eval_permuted = torch.gather(y_eval_exp, 1, gather_idx_y)

        # Output storage
        all_means, all_stds = [], []
        all_means_teacher_force, all_stds_teacher_force = [], []

        bs = sub_batch_size if sub_batch_size is not None else G * m
        for i in range(0, G * m, bs):
            end = min(i + bs, G * m)

            cx_fix = x_fix_exp[i:end]
            cy_fix = y_fix_exp[i:end]
            cx_eval = x_eval_permuted[i:end]
            cy_eval = y_eval_permuted[i:end]
            cinv = inv_perms_exp[i:end]

            if is_wiski:
                batch_indices = (torch.arange(i, end, device=x_fix.device) % m).cpu().tolist()
                active_models = []
                active_stats = []
                for idx in batch_indices: # Copies models to allow for proper streaming updates without altering base pretrained models
                    model_copy = copy.deepcopy(tnp_model.pretrained_base_models[idx])
                    try_cuda(model_copy)
                    active_models.append(model_copy)
                    active_stats.append(tnp_model.norm_stats[idx])

            # Context that grows with AR blocks
            cx_ctx = cx_fix
            cy_ctx = cy_fix

            means_blocks, stds_blocks = [], []

            # Autoregressive factorisation in blocks of size b
            for s in range(0, n_eval, b):
                e = min(s + b, n_eval)
                xt_blk = cx_eval[:, s:e, :]
                yt_blk = cy_eval[:, s:e, :]
                
                if is_wiski:
                    dist_blk= tnp_model.stream_batch_step(active_models, active_stats, xt_blk, yt_blk)
                else:
                    batch_blk = Batch(
                        xc=cx_ctx, yc=cy_ctx,
                        xt=xt_blk, yt=None,
                        y=None, x=None
                    )
                    dist_blk = np_pred_fn(tnp_model, batch_blk, predict_without_yt_tnpa=True)

                means_blocks.append(dist_blk.mean)
                stds_blocks.append(dist_blk.stddev)

                cx_ctx = torch.cat([cx_ctx, xt_blk], dim=1)
                cy_ctx = torch.cat([cy_ctx, yt_blk], dim=1)

            # Concatenate blocks back to [B_chunk, n_eval, dy] (still in permuted order)
            mu_chunk_perm = torch.cat(means_blocks, dim=1)
            std_chunk_perm = torch.cat(stds_blocks, dim=1)

            # Unpermute back to canonical x_eval order
            gather_inv = cinv.unsqueeze(-1).expand(-1, -1, dy)
            mu_chunk_can = torch.gather(mu_chunk_perm, 1, gather_inv)
            std_chunk_can = torch.gather(std_chunk_perm, 1, gather_inv)

            all_means.append(mu_chunk_can)
            all_stds.append(std_chunk_can)

        final_means_gmm = torch.cat(all_means, dim=0).reshape(G, m, n_eval, dy)
        final_stds_gmm = torch.cat(all_stds, dim=0).reshape(G, m, n_eval, dy)

        # -----------------------------------------------------------------------------------------
        # Randomly samples joints for J perms and computes the KL between that and the MoG
        # Also computes teacher-forced joint LL for y-axis performance metric.
        # -----------------------------------------------------------------------------------------

        perms_J = torch.stack([torch.randperm(n_eval, device=x_fix.device) for _ in range(J)])
        x_eval_J = x_eval.unsqueeze(0).expand(J, -1, -1, -1).reshape(J * m, n_eval, dx)
        y_eval_J = y_eval.unsqueeze(0).expand(J, -1, -1, -1).reshape(J * m, n_eval, dy)

        perms_J_exp = perms_J.unsqueeze(1).expand(-1, m, -1).reshape(J * m, n_eval)
        inv_perms_J = torch.argsort(perms_J_exp, dim=1)

        # Permute data
        gather_idx_J_x = perms_J_exp.unsqueeze(-1).expand(-1, -1, dx)
        gather_idx_J_y = perms_J_exp.unsqueeze(-1).expand(-1, -1, dy)
        x_perm_J = torch.gather(x_eval_J, 1, gather_idx_J_x)
        y_perm_J = torch.gather(y_eval_J, 1, gather_idx_J_y)

        x_fix_J = x_fix.unsqueeze(0).expand(J, -1, -1, -1).reshape(J * m, n_fix, dx)
        y_fix_J = y_fix.unsqueeze(0).expand(J, -1, -1, -1).reshape(J * m, n_fix, dy)

        # ---- Teacher-forced joint LL (block-AR teacher forcing over J permutations) ----
        cx_tf = x_fix_J
        cy_tf = y_fix_J
        lp_teacher_force = torch.zeros((J * m,), device=x_fix.device)

        if is_wiski:
            batch_indices = (torch.arange(0, J * m, device=x_fix.device) % m).cpu().tolist()
            active_models_tf = []
            active_stats_tf = []
            for idx in batch_indices:
                model_copy = copy.deepcopy(tnp_model.pretrained_base_models[idx])
                try_cuda(model_copy)
                active_models_tf.append(model_copy)
                active_stats_tf.append(tnp_model.norm_stats[idx])

        for s in range(0, n_eval, b):
            e = min(s + b, n_eval)
            xt_blk = x_perm_J[:, s:e, :]
            yt_blk = y_perm_J[:, s:e, :]

            if is_wiski:
                d_tf= tnp_model.stream_batch_step(active_models_tf, active_stats_tf, xt_blk, yt_blk)
            else:
                b_tf = Batch(
                    xc=cx_tf, yc=cy_tf,
                    xt=xt_blk, yt=None,
                    y=None,
                    x=None
                )
                d_tf = np_pred_fn(tnp_model, b_tf, predict_without_yt_tnpa=True)
            lp_teacher_force = lp_teacher_force + log_prob_targets(d_tf, yt_blk)

            cx_tf = torch.cat([cx_tf, xt_blk], dim=1)
            cy_tf = torch.cat([cy_tf, yt_blk], dim=1)
        if is_wiski: del active_models_tf

        # ---- Monte Carlo samples from the joint for KL  ----
        # Flatten S into batch: [S*J*m, ...]
        xc_flat = x_fix_J.unsqueeze(0).expand(S, -1, -1, -1).reshape(S * J * m, n_fix, dx)
        yc_flat = y_fix_J.unsqueeze(0).expand(S, -1, -1, -1).reshape(S * J * m, n_fix, dy)

        log_p_joint = torch.zeros((S, J * m), device=x_fix.device, dtype=torch.double)
        y_blocks = []

        if is_wiski:
            batch_indices = (torch.arange(0, J * m, device=x_fix.device) % m).cpu().tolist()
            active_models_mc = []
            active_stats_mc = []
            for idx in batch_indices:
                model_copy = copy.deepcopy(tnp_model.pretrained_base_models[idx])
                try_cuda(model_copy)
                active_models_mc.append(model_copy)
                active_stats_mc.append(tnp_model.norm_stats[idx])

            for s in range(0, n_eval, b):
                e = min(s + b, n_eval)
                blk_len = e - s

                xt_blk = x_perm_J[:, s:e, :]
                yt_blk_true = y_perm_J[:, s:e, :]

                d_blk = tnp_model.stream_batch_step(active_models_mc, active_stats_mc, xt_blk, yt_blk_true)
                y_samp = d_blk.sample((S,))
                log_p_joint += d_blk.log_prob(y_samp).sum(dim=(-1, -2)).double()
                y_blocks.append(y_samp.reshape(S * J * m, blk_len, dy))
        else:

            bs = sub_batch_size if sub_batch_size else S * J * m

            for s in range(0, n_eval, b):
                e = min(s + b, n_eval)
                blk_len = e - s

                xt_blk = x_perm_J[:, s:e, :] # [J*m, blk_len, dx]
                yt_blk_true = y_perm_J[:, s:e, :]
                xt_blk_flat = xt_blk.unsqueeze(0).expand(S, -1, -1, -1).reshape(S * J * m, blk_len, dx)
                yt_blk_flat_true = yt_blk_true.unsqueeze(0).expand(S, -1, -1, -1).reshape(S * J * m, blk_len, dy)

                # Chunked to avoid OOM
                y_blk_flat_list = []
                lp_blk_flat_list = []

                for i in range(0, S * J * m, bs):
                    end = min(i + bs, S * J * m)
                    b_chunk = Batch(
                        xc=xc_flat[i:end], yc=yc_flat[i:end],
                        xt=xt_blk_flat[i:end], yt=None,
                        y=None,
                        x=None
                    )
                    d_chunk = np_pred_fn(tnp_model, b_chunk, predict_without_yt_tnpa=True)
                    y_chunk = d_chunk.sample() # [B, blk_len, dy]
                    lp_chunk = log_prob_targets(d_chunk, y_chunk) # [B]

                    y_blk_flat_list.append(y_chunk)
                    lp_blk_flat_list.append(lp_chunk)

                y_blk_flat = torch.cat(y_blk_flat_list, dim=0)                  # [S*J*m, blk_len, dy]
                lp_blk_flat = torch.cat(lp_blk_flat_list, dim=0).view(S, J * m) # [S, J*m]

                y_blocks.append(y_blk_flat)
                log_p_joint = log_p_joint + lp_blk_flat.double()

                # Update contexts
                xc_flat = torch.cat([xc_flat, xt_blk_flat], dim=1)
                yc_flat = torch.cat([yc_flat, yt_blk_flat_true], dim=1)

        # Assemble full sampled sequence (still in permuted order)
        y_full_perm_flat = torch.cat(y_blocks, dim=1) # [S*J*m, n_eval, dy]
        y_full_perm = y_full_perm_flat.view(S, J * m, n_eval, dy)

        # Unpermute back to canonical order for MoG evaluation
        inv_idx_S = inv_perms_J.unsqueeze(0).unsqueeze(-1).expand(S, -1, -1, dy) # [S, J*m, n_eval, dy]
        y_full_canon = torch.gather(y_full_perm, 2, inv_idx_S)                   # [S, J*m, n_eval, dy]
        y_eval_final = y_full_canon.view(S, J, m, n_eval, dy).reshape(S * J, m, n_eval, dy)

        # Evaluate MoG Probability
        def evaluate_mog_fast(y_input, final_means, final_std):
            # Expand y to compare against all G components
            y_exp = y_input.unsqueeze(1)        # [Batch, 1, m, n_eval, dy]
            mu_exp = final_means.unsqueeze(0)   # [1, G, m, n_eval, dy]
            std_exp = final_std.unsqueeze(0)    # [1, G, m, n_eval, dy]
            dist = torch.distributions.Normal(mu_exp, std_exp)
            lps = log_prob_targets(dist, y_exp).double() # [Batch, G, m]
            return torch.logsumexp(lps, dim=1) - math.log(G)

        log_p_mog = evaluate_mog_fast(y_eval_final, final_means_gmm, final_stds_gmm).double() # [S*J, m]

        # Teacher-forced MoG joint LL on the groundtruth y (for y-axis uplift)
        log_p_gt_under_mog = evaluate_mog_fast(
            y_eval.unsqueeze(0), final_means_gmm, final_stds_gmm
        ).double().squeeze(0) # [m]

        # KL Difference (x axis)
        log_p_joint = log_p_joint.view(S * J, m)           # [S*J, m]
        kl_all = log_p_joint - log_p_mog                   # [S*J, m]
        kl_reshaped = kl_all.view(S, J, m)
        kls = kl_reshaped.mean(dim=0).mean(dim=0)          # [m]
        mean_kl = kls.mean().item()

        # Performance metric (y axis): teacher-forced joint NLL vs GT
        neg_lp = -lp_teacher_force.view(J, m)              # [J, m]
        excess_nll = neg_lp - gt_nll                       # [J, m] (broadcast)
        mean = excess_nll.mean(dim=0)                      # [m]
        m_mean_val = mean.mean().item()                    # scalar

        # Compute performance difference for GMM (teacher-forced MoG) vs GT
        nll_mog_ensemble = -log_p_gt_under_mog             # [m]
        gmm_excess_nll = nll_mog_ensemble - gt_nll.squeeze(0)
        mean_gmm_excess = gmm_excess_nll.mean().item()

        sample_kls = kls.detach().cpu()
        sample_nlls = mean.detach().cpu()
        sample_gmm_nlls = gmm_excess_nll.detach().cpu()

        return mean_kl, m_mean_val, sample_kls, sample_nlls, mean_gmm_excess, sample_gmm_nlls


# Samples variance over trained models with different seeds
def exchange(models_with_different_seeds, data_loader, device, n, init, b, use_torch_grad, monte_carlo_samples, sub_batch_size, max_samples, set_wiski_params_gt, number_of_mog_permutations, joints_average_over):
    no_models = len(models_with_different_seeds)
    # Note - may want to consider diving by nt in future? (or even nt * dy)

    m_vars = []
    m_nlls = []
    m_gmm_nlls = []
    kl_samples = []
    nll_samples = []
    nll_gmm_samples = []
    i = 0
    nc_prev, nt_prev = None, None
    for data in data_loader:
        # Ensures sequence length is correct and that nc and nt remain constant over samples.
        # This prevents greater variance for longer sequences (ie lacking comparison) but comes at cost of expressivity.
        # Could normalise variances or look at multiple sequence lengths.
        seq_len_data = data.x.shape[1]
        batch_size = data.xc.shape[0]
        assert seq_len_data == n + init, f"Data sequence length {seq_len_data} does not match required sequence length {n + init}."
        if nc_prev is not None and nt_prev is not None:
            assert data.xc.shape[1] == nc_prev, f"Context set size {data.xc.shape[1]} does not match previous size {nc_prev}."
            assert data.xt.shape[1] == nt_prev, f"Target set size {data.xt.shape[1]} does not match previous size {nt_prev}."
        nc_prev, nt_prev = data.xc.shape[1], data.xt.shape[1]

        xc, yc, xt, yt = data.xc, data.yc, data.xt, data.yt
        xc, yc, xt, yt = xc.to(device), yc.to(device), xt.to(device), yt.to(device)
        nc, nt = xc.shape[1], xt.shape[1]
        #perms = torch.stack([torch.randperm(n, device=device) for _ in range(no_permutations)])
        mods_out_mvar = []
        mods_out_mnll = []
        mods_out_gmm_mnll = []
            
        # Computes m_var for each model
        for model in models_with_different_seeds:
            val = m_var_fixed(model, xc, yc, xt, yt, gt_pred=data.gt_pred, use_torch_grad=use_torch_grad, monte_carlo_samples=monte_carlo_samples, sub_batch_size=sub_batch_size, set_wiski_params_gt=set_wiski_params_gt, n=n, init=init, b=b, number_of_mog_permutations=number_of_mog_permutations, joints_average_over=joints_average_over)
            mods_out_mvar.append(val[0])
            mods_out_mnll.append(val[1])
            kl_samples.extend(val[2])
            nll_samples.extend(val[3])
            mods_out_gmm_mnll.append(val[4])
            nll_gmm_samples.extend(val[5])
        m_vars.append(mods_out_mvar)
        m_nlls.append(mods_out_mnll)
        m_gmm_nlls.append(mods_out_gmm_mnll)
        i += 1
        print(f'i={i} max_samples={max_samples} processed_so_far={i * batch_size}')

    assert i>=1, "No data batches were processed."

    m_vars = np.array(m_vars)
    m_nlls = np.array(m_nlls)
    m_gmm_nlls = np.array(m_gmm_nlls)

    model_vars = m_vars.mean(axis=0) # Average for each model over the data batches
    model_nlls = m_nlls.mean(axis=0) # Average NLL over the data batches
    model_gmm_nlls = m_gmm_nlls.mean(axis=0)

    mean_m_vars = model_vars.mean() # Mean over the models
    mean_m_nlls = model_nlls.mean() # Mean NLL over the models
    mean_m_gmm_nlls = model_gmm_nlls.mean()

    m_var_nll_gmmnll_samples = list(zip(kl_samples, nll_samples, nll_gmm_samples))

    # Can't do t test with single model
    if no_models == 1:
        return (mean_m_vars, None), (mean_m_nlls, None), m_var_nll_gmmnll_samples, mean_m_gmm_nlls
    student_t_crit = stats.t.ppf(0.975, df=no_models - 1)
    sem_m_var = stats.sem(model_vars)
    sem_m_nll = stats.sem(model_nlls)
    half_w_m_var = student_t_crit * sem_m_var
    half_w_m_nll = student_t_crit * sem_m_nll
    return (mean_m_vars, half_w_m_var), (mean_m_nlls, half_w_m_nll), m_var_nll_gmmnll_samples, mean_m_gmm_nlls


# Converts string to file that can be written safely with regex
def _slug(s: str) -> str:
    return re.sub(r"[^\w\-\.]", "_", s)

def get_plot_rbf(n, init, samples_per_epoch, batch_size):
    # Data loader - RBF kernel in this case
    ard_num_dims = 1
    min_log10_lengthscale = -0.602
    max_log10_lengthscale = 0.0
    context_range = [[-2.0, 2.0]]
    target_range = [[-2.0, 2.0]]
    noise_std=0.1
    deterministic = True

    rbf_kernel_factory = partial(RBFKernel, ard_num_dims=ard_num_dims, min_log10_lengthscale=min_log10_lengthscale,
                         max_log10_lengthscale=max_log10_lengthscale)
    kernels = [rbf_kernel_factory]
    gen_val = RandomScaleGPGenerator(dim=1, min_nc=init, max_nc=init, min_nt=n, max_nt=n, batch_size=batch_size,
        context_range=context_range, target_range=target_range, samples_per_epoch=samples_per_epoch, noise_std=noise_std,
        deterministic=deterministic, kernel=kernels)
    return gen_val

# Attempts to recreate something like figure 2
def plot_models_setup_rbf_same(ar_samples, wiski_gridsize_updategp_paran, wiski_gridsize_initsize_pretrain_params):
    # Defines each model
    tnp_plain = ['experiments/configs/synthetic1dRBF/best_trained/tnpd.yaml', 'experiments/configs/synthetic1dRBF/best_trained/weights/tnpd.ckpt', "TNP-D", "local"]
    inc_tnp = ['experiments/configs/synthetic1dRBF/best_trained/incTNP.yaml', 'experiments/configs/synthetic1dRBF/best_trained/weights/incTNP.ckpt', "incTNP", "local"]
    inc_tnp_batched=['experiments/configs/synthetic1dRBF/best_trained/incTNP-Batched.yaml', 'experiments/configs/synthetic1dRBF/best_trained/weights/incTNP-Batched.ckpt', "incTNP-Batched", "local"]
    models_tnp = [tnp_plain, inc_tnp, inc_tnp_batched]

    tnp_ar_cptk, tnp_ar_yml, tnp_name = 'experiments/configs/synthetic1dRBF/gp_tnpa_rangesame_scheduler.yml', 'REMOVED', "TNP-A"
    tnp_ar_list = [[tnp_ar_cptk, tnp_ar_yml, tnp_name, samp] for samp in ar_samples]

    wiski_name = "WISKI"
    wiski_no_pretrain = [["", "", wiski_name, grid_size, update_gp] for (grid_size, update_gp) in wiski_gridsize_updategp_paran]

    wiski_name = "WISKI-Pre"
    wiski_list_pretrain = [["", "", wiski_name, grid_size, init_size] for (grid_size, init_size) in wiski_gridsize_initsize_pretrain_params]

    wiski_list = wiski_list_pretrain + wiski_no_pretrain

    models_all = wiski_list+ models_tnp + tnp_ar_list
    return models_all

def plot_models_setup_rbf_nc_large(wiski_gridsize_initsize_pretrain_params):
    # Defines each model
    tnp_plain = ['experiments/configs/incTNPCheckpoints/RBF/Nc_512/config_tnpd_rbf_Nc512_epoch_0249.yaml', 'experiments/configs/incTNPCheckpoints/RBF/Nc_512/tnpd_rbf_Nc512_epoch_0249.ckpt', "TNP-D", "local"]
    inc_tnp = ['experiments/configs/incTNPCheckpoints/RBF/Nc_512/config_incTNPb_rbf_Nc512_epoch_0199.yaml', 'experiments/configs/incTNPCheckpoints/RBF/Nc_512/incTNP_rbf_Nc512_epoch_0249.ckpt', "incTNP", "local"]
    inc_tnp_batched=['experiments/configs/incTNPCheckpoints/RBF/Nc_512/config_incTNPb_rbf_Nc512_epoch_0199.yaml', 'experiments/configs/incTNPCheckpoints/RBF/Nc_512/inctnpb_rbf_Nc512_epoch_0199.ckpt', "incTNP-Batched", "local"]
    models_tnp = [tnp_plain, inc_tnp, inc_tnp_batched]

    wiski_name = "WISKI-Pre"
    wiski_list_pretrain = [["", "", wiski_name, grid_size, init_size] for (grid_size, init_size) in wiski_gridsize_initsize_pretrain_params]

    wiski_list = wiski_list_pretrain

    models_all = models_tnp + wiski_list
    return models_all

def extract_vars_from_folder_name(folder_name):
    patterns = {
        'n': r'n_(\d+)',
        'b': r'b_(\d+)',
        'init': r'init_(\d+)'
    }
    variables_found = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, folder_name)
        assert match, "Invalid folder name no match found"
        value_str_found = match.group(1)
        # Sorts types
        variables_found[key] = int(value_str_found)
    return variables_found

def generate_folder_name(n, init, b):
    file_str = f'n_{n}_init_{init}_b_{b}'
    return file_str

# Takes a folder with data written and plots the fig
def plot_from_folder(folder):
    # Plot hypers
    max_samples_plot = 8 # Max number of samples
    max_samples_wiski_plot = 8
    EPSILON = 1e-7
    min_nll_thresh_samples = 1e-9 # Removes small dots with better nll than GT GP from vis to keep scales sensible for interpretation (they guess and get lucky)
    rename_wiski = True
    plot_samples = False
    plot_error_bars = True
    # End of plot hypers

    pars = extract_vars_from_folder_name(folder)
    n, init, b = pars["n"], pars["init"], pars["b"]

    data_directory = Path(folder)
    model_folders_unstr = [p for p in data_directory.iterdir() if p.is_dir()]
    # Imposes plot ordering
    remaining_folders = set(model_folders_unstr)
    order = ["TNP-D", "incTNP", "incTNP-Batched", "TNP-A", "GP-Expanding"]
    model_folders = []
    for prefix in order:
        matches = [p for p in remaining_folders if p.name.startswith(prefix)]
        if matches:
            matches.sort(key=lambda p: p.name)
            model_folders.extend(matches)
            remaining_folders.difference_update(matches)
    if remaining_folders:
        model_folders.extend(sorted(list(remaining_folders), key=lambda p: p.name))

    # Colour pallete to use - sensible but pretty
    tableau_colorblind_10 = ['#006BA4','#FF800E','#ABABAB','#595959','#5F9ED1','#C85200','#898989','#A2C8EC','#FFBC79','#CFCFCF']
    colours = cycle(tableau_colorblind_10)
    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    #fig, ax = plt.subplots()
    # Stores all plotted points to calculate graph limits
    all_xs = []
    all_ys = []
    # Loops through models and plots them
    for (model_folder, colour) in zip(model_folders, colours):
        with open(model_folder / 'summary.txt', 'r', encoding='utf-8') as f:
            model_summary_txt = f.read()
        # Extracts from the summary fixed format
        lines = model_summary_txt.split("\n")
        model_name = lines[1].split(": ")[1]
        if model_name.startswith("Streamed GP-S"): continue
        mean_m_var = float(lines[2].split(" ")[1])
        mean_m_nlls = float(lines[3].split(" ")[1])
        mean_m_gmm_nlls = float(lines[5].split(" ")[1]) if len(lines) >= 6 else None
        npz_file = lines[4].split(": ")[1]
        data = np.load(model_folder / npz_file)
        samples_m_var = data["samples_m_var"]
        samples_m_nll = data["samples_m_nll"]
        #samples_m_gmm_nll_np = data["samples_m_gmm_nll_np"] if "samples_m_gmm_nll_np" in data is not None else None

        if rename_wiski and model_name.startswith("WISKI"): model_name = "WISKI"
        # Clamps KL estimates that are below 0 (due to MC noise estimates)
        samples_m_var = np.maximum(samples_m_var, EPSILON)
        mean_m_var = max(mean_m_var, EPSILON)

        idx_to_rem = [i for i in range(len(samples_m_nll)) if samples_m_nll[i] < min_nll_thresh_samples]
        samples_m_nll = [samples_m_nll[i] for i in range(len(samples_m_nll)) if i not in idx_to_rem]
        samples_m_var = [samples_m_var[i] for i in range(len(samples_m_var)) if i not in idx_to_rem]
        max_samples_this_model = max_samples_plot if not model_name.startswith("WISKI") else max_samples_wiski_plot
        samples_m_nll = samples_m_nll[:max_samples_this_model] if len(samples_m_nll) >= max_samples_this_model else samples_m_nll
        samples_m_var = samples_m_var[:max_samples_this_model] if len(samples_m_var) >= max_samples_this_model else samples_m_var

        all_xs.extend(samples_m_var)
        all_xs.append(mean_m_var)
        all_ys.extend(samples_m_nll)
        all_ys.append(mean_m_nlls)

        if plot_samples:
            # Plots small dots
            ax.scatter(
                samples_m_var,
                samples_m_nll,
                s=50,
                c=[colour],
                alpha=1.0,
                marker='o',
                edgecolors='none',
                zorder=2,
                clip_on=False,
            )

            # Line to centroid dot
            for (sx, sy) in zip(samples_m_var, samples_m_nll):
                ax.plot([mean_m_var, sx], [mean_m_nlls, sy], lw=1.6, c=colour, alpha=1.0,zorder=1)

        if plot_error_bars and len(samples_m_var) > 0:
            x_low, x_high = min(samples_m_var), max(samples_m_var)
            y_low, y_high = min(samples_m_nll), max(samples_m_nll)
            #xerr = np.array([[mean_m_var - x_low], [x_high - mean_m_var]])
            #yerr = np.array([[mean_m_nlls - y_low], [y_high - mean_m_nlls]])

            x_sem = stats.sem(samples_m_var)
            y_sem = stats.sem(samples_m_nll)
            x_low = max(mean_m_var - x_sem, EPSILON)
            x_high = mean_m_var + x_sem
            y_low = max(mean_m_nlls - y_sem, EPSILON)
            y_high = mean_m_nlls + y_sem
            xerr = np.array([[mean_m_var - x_low], [x_high - mean_m_var]])
            yerr = np.array([[mean_m_nlls - y_low], [y_high - mean_m_nlls]])

            #x_std, y_std = np.std(samples_m_var), np.std(samples_m_nll)
            #xerr, yerr = x_std, y_std 
            ax.errorbar(
                mean_m_var, 
                mean_m_nlls, 
                xerr=np.abs(xerr),
                yerr=np.abs(yerr), 
                fmt='none',
                ecolor=colour, 
                elinewidth=1.6, 
                capsize=5, 
                capthick=1.6,
                alpha=0.8,
                zorder=1
            )

        # Plots large dot
        ax.scatter(
            mean_m_var,
            mean_m_nlls,
            s=200,
            c=[colour],
            alpha=1.0,
            marker='o',
            edgecolors='none',
            label=model_name,
            zorder=3,
            clip_on=False
        )

        # Plots GMM mean also
        if mean_m_gmm_nlls is not None and mean_m_gmm_nlls >= min_nll_thresh_samples and not (model_name.startswith("WISKI") or model_name.startswith("TNP-A")):
            all_ys.append(mean_m_gmm_nlls)
            ax.plot(
                [mean_m_var, mean_m_var], 
                [mean_m_nlls, mean_m_gmm_nlls], 
                ls='--',
                lw=1.5, 
                c=colour, 
                alpha=0.6,
                zorder=1
            )

            ax.scatter(
                mean_m_var,
                mean_m_gmm_nlls,
                s=100,
                facecolors='white',
                edgecolors=colour,
                linewidth=2.0,
                alpha=1.0,
                marker='D',
                zorder=3,
                clip_on=False
            )
                        

    # Calculates log limits for graphs - clamped reasonably in case of negative / zero x and ys
    min_x_log = np.floor(np.log10(min(all_xs)))
    max_x_log = np.ceil(np.log10(max(all_xs)))
    min_y_log = np.floor(np.log10(min(all_ys)))
    max_y_log = np.ceil(np.log10(max(all_ys)))
    
    x_pad_factor = 1.2 
    y_pad_factor = 1.2

    #ax.set_xlim(10**min_x_log, max(all_xs) * x_pad_factor)
    #ax.set_ylim(10**min_y_log, max(all_ys) * y_pad_factor)
    #ax.set_xlim(10**min_x_log, 10**max_x_log)
    #ax.set_ylim(10**min_y_log, 10**max_y_log)
    ax.set_xlim(min(all_xs) / x_pad_factor, max(all_xs) * x_pad_factor)
    ax.set_ylim(min(all_ys) / y_pad_factor, max(all_ys) * y_pad_factor)

    ax.xaxis.set_major_formatter(LogFormatterMathtext())
    ax.yaxis.set_major_formatter(LogFormatterMathtext())
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.spines['left'].set_position(('outward', 8))
    ax.spines['bottom'].set_position(('outward', 8))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.set_xlabel("KL-Gap")
    ax.set_ylabel("Neg. Joint Log-Likelihood Mean (- Optimal)")

    # Tick params
    ax.tick_params(axis='both', which='major', length=4, width=0.8)

    ax.legend(scatterpoints=1, markerscale=0.55)

    ax.set_title(f"Exchangeability vs Performance (N={n}, b={b}, init={init})")

    plt.savefig(folder + "/bayesianness.pdf", bbox_inches="tight")


# Attempts to recreate something like figure 2. Use plot_models_setup as helper for this func.
def gather_stats_models(helper_tuple, base_folder_name, n, init, b):
    skip_existing_folders = True # Skips existing file writes - no need to do work again
    # Default hypers
    default_cfg = {
        "samples_per_epoch": 2_000, # How many datapoints in datasets
        "joints_average_over": 128 ,# Number of joints to compute the KL average between that and the MoG 
        "number_of_mog_permutations": 256, # MoG permutations to construct bayesified prediction rule
        "batch_size": 16, # Just a perf side thing - higher batch can lead to faster throughput but at greater mem
        "monte_carlo_samples": 32, # Number of monte carlo samples S to use to estimate the KL between gaussian and GMM
        "sub_batch_size": None, # Max chunked size to prevent OOMs
    }
    # WISKI hypers - no pretraining
    wiski_cfg = {
        "samples_per_epoch": 128,
        "joints_average_over": 32 ,# Number of joints to compute the KL average between that and the MoG 
        "number_of_mog_permutations": 64, # MoG permutations to construct bayesified prediction rule
        "batch_size": 16,
        "monte_carlo_samples": 20,
        "sub_batch_size": 8_192,
    }
    # WISKI pretrain hypers
    wiski_pretrain_cfg = {
        "samples_per_epoch": 256,
        "joints_average_over": 32 ,# Number of joints to compute the KL average between that and the MoG 
        "number_of_mog_permutations": 64, # MoG permutations to construct bayesified prediction rule
        "batch_size": 32,
        "monte_carlo_samples": 20,
        "sub_batch_size": 8_192,
    }
    # TNP-A hypers
    tnpa_cfg = {
        "samples_per_epoch": 1024,
        "joints_average_over": 128 ,# Number of joints to compute the KL average between that and the MoG 
        "number_of_mog_permutations": 128, # MoG permutations to construct bayesified prediction rule
        "batch_size": 512,
        "monte_carlo_samples": 128,
        "sub_batch_size": 256,
    }
    
    #else: print(f"configs specified reverting to sensible defaults for nc={nc} nt={nt}")
    # -------------- End of hypers -------------------------------

    (models) = helper_tuple

    # Ensures base and data folder already exists
    base_path = Path(base_folder_name)
    base_path.mkdir(exist_ok=True)
    data_folder = base_folder_name + f"/{generate_folder_name(n=n, init=init, b=b)}"
    data_path = Path(data_folder)
    data_path.mkdir(exist_ok=True)

    for mod_data in models:
        mod_cptk, mod_yml, model_name = mod_data[0], mod_data[1], mod_data[2]
        local = True if len(mod_data) >= 4 and mod_data[3] == "local" else False
        # Formats model names
        if model_name == "GP-Expanding":
            _, _, name_base, chunk_size, strat = mod_data
            gp_ext = "" if strat == "Expanding" else ""
            model_name_fmt = model_name + gp_ext+ f' (ch={chunk_size})'
        elif model_name == "TNP-A":
            model_name_fmt = model_name + f' ({mod_data[3]} samples)'
        elif model_name == "Streamed Sparse GP":
            _, _, name_base, chunk_size, strat = mod_data
            model_name_fmt = model_name + f' (ch={chunk_size})'
        elif model_name == "WISKI":
            _, _, name_base, inducing_size, updategp = mod_data
            model_name_str = "WISKI-Learn" if updategp else "WISKI-Fixed"
            model_name_fmt = model_name_str + f' (ind={inducing_size})'
        elif model_name == "WISKI-Pre":
            _, _, name_base, inducing_size, init_chunk_size = mod_data
            model_name_fmt = model_name + f' (ind={inducing_size}, pre={init_chunk_size})'       
        else: model_name_fmt = model_name
        print(model_name_fmt)

        # Checks to see if exact model run has already been written (no need to do again if it has)
        model_folder = data_folder + "/" + _slug(model_name_fmt)
        if skip_existing_folders and os.path.exists(model_folder): continue

        use_torch_grad = False
        cfg = default_cfg
        set_wiski_params_gt = False
        if model_name == "TNP-A":
            model = get_model(mod_cptk, mod_yml, local_weights=local)
            model.num_samples = mod_data[3]
            model.permute = False # Whether to permute the target order during AR rollout
            model.eval()
            cfg = tnpa_cfg
        elif model_name == "WISKI":
            model = ExchangeCalcWiskiRBF(grid_size=inducing_size, init_chunk_size=0, 
                                         pretrain=False, update_gp_and_stem=updategp)
            set_wiski_params_gt = True
            use_torch_grad = True
            cfg = wiski_cfg
        elif model_name == "WISKI-Pre":
            model = ExchangeCalcWiskiRBF(grid_size=inducing_size, init_chunk_size=init_chunk_size, 
                                         pretrain=True)
            use_torch_grad = True
            cfg = wiski_pretrain_cfg
        else:
            model = get_model(mod_cptk, mod_yml, local_weights=local)
            model.eval()

        (mean_m_var, _,), (mean_m_nlls, _), m_var_nll_samples, mean_m_gmm_nlls = exchange(
            [model], get_plot_rbf(n, init, cfg["samples_per_epoch"], batch_size=cfg["batch_size"]),
            device='cuda', n=n, init=init, b=b, 
            use_torch_grad=use_torch_grad, monte_carlo_samples=cfg["monte_carlo_samples"], 
            sub_batch_size=cfg["sub_batch_size"], max_samples=(cfg["samples_per_epoch"]),
            set_wiski_params_gt=set_wiski_params_gt,
            joints_average_over=cfg["joints_average_over"],
            number_of_mog_permutations=cfg["number_of_mog_permutations"]
        )
        samples_m_var = [x[0].item() for x in m_var_nll_samples]
        samples_m_nll = [x[1].item() for x in m_var_nll_samples]
        samples_m_gmm_nll = [x[2].item() for x in m_var_nll_samples]

        # Writes recorded results to folder
        model_folder_path = Path(model_folder)
        model_folder_path.mkdir(exist_ok=True)
        samples_m_var_np = np.array(samples_m_var)
        samples_m_nll_np = np.array(samples_m_nll)
        samples_m_gmm_nll_np = np.array(samples_m_gmm_nll)
        rel_sample_name = "samples.npz"
        save_samples_path = model_folder + "/" + rel_sample_name
        summary_block = ("-" * 20) + "\n" + f"Model_Name: {model_name_fmt}\nMean_M_Var: {mean_m_var}\nMean_M_NLL: {mean_m_nlls}\nSamples_File (samples_m_var and samples_m_nll): {rel_sample_name}\nMean_M_GMM_NLL: {mean_m_gmm_nlls}"
        print(summary_block)
        np.savez_compressed(save_samples_path, samples_m_var=samples_m_var_np, samples_m_nll=samples_m_nll_np, samples_m_gmm_nll_np=samples_m_gmm_nll_np)
        with open(model_folder + '/summary.txt', 'w') as file_object:
            file_object.write(summary_block)

    plot_from_folder(data_folder) # Plots generated data



if __name__ == "__main__":
    gather_stats_models(plot_models_setup_rbf_same(ar_samples=[], 
        wiski_gridsize_updategp_paran=[], 
        wiski_gridsize_initsize_pretrain_params=[(32, 20)]), 
        base_folder_name="experiments/plot_results/exchange/klmeas_joint_true", n=40, init=20, b=1)
    gather_stats_models(plot_models_setup_rbf_nc_large(
        wiski_gridsize_initsize_pretrain_params=[]), 
        base_folder_name="experiments/plot_results/exchange/klmeas_joint_true", n=100, init=3, b=1)