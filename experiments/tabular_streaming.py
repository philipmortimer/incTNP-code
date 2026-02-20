# Shows tabular streaming performance as more ctx points arrive somewhat like figure 3 in https://proceedings.mlr.press/v130/stanton21a/stanton21a.pdf
import argparse
import gc
import os
import re
from contextlib import nullcontext

import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from plot_adversarial_perms import get_model
from scipy.io import arff
from torch import nn
from torch.distributions import Normal
from tueplots import bundles

import wandb
from tnp.data.base import Batch
from tnp.data.tabular_data import TabularDataGeneratorUniqueMLPPerDataset
from tnp.models.incUpdateBase import IncUpdateEff

from tnp.models.wiskigp import (
     ElevatorsWiski,
     GenericWiski,
     PowerplantWiski,
     ProteinWiski,
     SkillcraftWiski,
 )
from tnp.utils.np_functions import np_pred_fn

plt.rcParams.update(bundles.icml2024())


def get_epoch_number(artifact):
    """
    Parses 'model-epoch-0399-gj03lc2w:v0' and returns 399.
    Returns -1 if 'epoch-' is not found (e.g. if name is 'model-best').
    """
    # Look for the specific pattern "epoch-" followed by digits
    match = re.search(r"epoch-(\d+)", artifact.name)

    if match:
        return int(match.group(1))

    # If the artifact is named "model-best" or doesn't have an epoch,
    # treat it as -1 so it is never selected as the "last" one.
    return -1


def denormalize_predictions(pred_dist, batch):
    """Denormalize predictions back to original scale."""
    if hasattr(batch, "y_mean") and hasattr(batch, "y_std"):
        # Create new distribution with denormalized parameters

        # Denormalize mean and stddev
        denorm_mean = pred_dist.mean * batch.y_std + batch.y_mean
        denorm_stddev = pred_dist.stddev * batch.y_std

        # Return new distribution with denormalized values
        return Normal(denorm_mean, denorm_stddev)
    return pred_dist


def denormalize_targets(yt, batch):
    """Denormalize targets back to original scale."""
    if hasattr(batch, "y_mean") and hasattr(batch, "y_std"):
        return yt * batch.y_std + batch.y_mean
    return yt


# Loads raw uci dataset files
def load_uci_dataset(data_path, dataset_name, seed):
    assert dataset_name in (
        "Skillcraft",
        "Powerplant",
        "Elevators",
        "Protein",
        "Synthetic",
    ), "Must be one of pre specified uci sets"
    if dataset_name == "Skillcraft":
        print(f"Loading Skillcraft CSV data from: {data_path}")
        data = pd.read_csv(data_path, na_values=["?"]).dropna()
        data = data.drop(columns=["GameID"])
        target_col = "LeagueIndex"
        y = data[target_col].values
        X = data.drop(columns=[target_col]).values
        X = np.log1p(X)
        data = np.column_stack([X, y])
    elif dataset_name == "Powerplant":
        print(f"Loading Power Plant Excel data from: {data_path}")
        data = np.array(pd.read_excel(data_path, engine="openpyxl"))
    elif dataset_name == "Elevators":
        print(f"Loading Elevators ARFF data from: {data_path}")
        data, meta = arff.loadarff(data_path)
        data = pd.DataFrame(data)
        for col in data.select_dtypes([object]):
            data[col] = data[col].astype(str).str.decode("utf-8")
        data = data.apply(pd.to_numeric, errors="coerce")
        std_devs = data.std()
        near_constant_cols = std_devs[std_devs < 1e-5].index
        if len(near_constant_cols) > 0:
            print(f"Warning: Dropping constant columns: {list(near_constant_cols)}")
            data = data.drop(columns=near_constant_cols)
        data = pd.DataFrame(data).values
    elif dataset_name == "Protein":
        print(f"Loading Protein CSV data from: {data_path}")
        data = pd.read_csv(data_path)
        # Move target from first to last column
        first_col = data.columns[0]
        data = data[[col for col in data.columns if col != first_col] + [first_col]]
        data = data.drop_duplicates()
        data = data.values
    elif dataset_name == "Synthetic":
        no_items = data_path  # In this specific case the data_path is an int showcasing the number of ctx points
        data = TabularDataGeneratorUniqueMLPPerDataset(
            dim=20,
            min_nc=no_items - 1,
            max_nc=no_items - 1,
            min_nt=1,
            max_nt=1,
            samples_per_epoch=1,
            batch_size=1,
            deterministic=True,
            deterministic_seed=seed,
        ).sample_batch(nc=no_items - 1, nt=1, batch_shape=torch.Size([1]))
        data.to(device="cpu")  # Moves data to cpu
    return data


def get_model_list():
    tnp_plain = ('experiments/configs/incTNPCheckpoints/Tabular/config_tnpd_tab_epoch_0679.yaml',
                 'experiments/configs/incTNPCheckpoints/Tabular/tnpd_tab_epoch_0679.ckpt', 'TNP_D')
    incTNP = ('experiments/configs/incTNPCheckpoints/Tabular/config_inctnp_tab_epoch_0499.yaml',
        'experiments/configs/incTNPCheckpoints/Tabular/inctnp_tab_epoch_0499.ckpt', 'incTNP')
    batchedTNP = ('experiments/configs/incTNPCheckpoints/Tabular/config_inctnpb_tab_epoch_0499.yaml',
        'experiments/configs/incTNPCheckpoints/Tabular/inctnpb_tab_epoch_0499.ckpt', 'incTNP-Batched')
    wiski = ('', '', 'WISKI')
    return [wiski, batchedTNP, incTNP, tnp_plain]


def get_uci_sets():
    skillcraft = ('experiments/uci_datasets/skillcraft/SkillCraft1_Dataset.csv', 'Skillcraft')
    powerplant = ('experiments/uci_datasets/powerplant/CCPP/Folds5x2_pp.xlsx', "Powerplant")
    elevators = ('experiments/uci_datasets/elevators/dataset_2202_elevators.arff', 'Elevators')
    protein = ('experiments/uci_datasets/protein/CASP.csv', 'Protein')
    synthetic_prior_nc_50k = (50_000, 'Synthetic')
    return [skillcraft, powerplant, synthetic_prior_nc_50k, protein, elevators]


# Takes the raw data, applies a random permutation and loads the batch of all the ctx and tgt data for that permutation. Also includes a return index to show where the pretrain split is
def load_batch(
    data,
    train_split,
    pre_train_prop,
    max_features,
    device,
    normalisation_mode,
    dataset_name,
):
    if dataset_name == "Synthetic":
        # ... (Synthetic block remains unchanged) ...
        _, N, dx = data.x.shape
        _, _, dy = data.y.shape
        nc = int(round(N * train_split))
        pre_train_idx = int(round(nc * pre_train_prop))
        indices = np.arange(N)
        np.random.shuffle(indices)
        x_shuffled = data.x[:, indices]
        y_shuffled = data.y[:, indices]
        xc, yc = x_shuffled[:, :nc, :], y_shuffled[:, :nc, :]
        xt, yt = x_shuffled[:, nc:, :], y_shuffled[:, nc:, :]
        batch_obj = Batch(xc=xc, yc=yc, xt=xt, yt=yt, x=x_shuffled, y=y_shuffled)
        batch_obj.pre_pad_dx = dx
        batch_obj.x_mean = torch.tensor(0.0).to(device)
        batch_obj.x_std = torch.tensor(1.0).to(device)
        batch_obj.y_mean = torch.tensor(0.0).to(device)
        batch_obj.y_std = torch.tensor(1.0).to(device)
        batch_obj.pre_train_items = pre_train_idx
        return batch_obj

    # Add "online" to valid modes
    valid_modes = (
        "strict",
        "wiski_style_zscore_targets",
        "wiski_style_iqr_targets",
        "calibration_zscore_targets",
        "calibration_iqr_targets",
        "online_zscore",
        "",
    )
    assert normalisation_mode in valid_modes, f"Invalid mode: {normalisation_mode}"

    x_raw = data[:, :-1]
    y_raw = data[:, -1].reshape(-1, 1)
    N, dx = x_raw.shape
    _, dy = y_raw.shape
    if dx > max_features:
        raise ValueError(f"Dataset has {dx} features, max is {max_features}")

    nc = int(round(N * train_split))
    pre_train_idx = int(round(nc * pre_train_prop))

    indices = np.arange(N)
    np.random.shuffle(indices)

    x_shuffled = x_raw[indices]
    y_shuffled = y_raw[indices]

    xc = x_shuffled[:nc]
    yc = y_shuffled[:nc]
    xt = x_shuffled[nc:]
    yt = y_shuffled[nc:]

    # --- NORMALIZATION LOGIC ---

    if normalisation_mode == "strict":
        # ... (Strict block unchanged) ...
        c_min = np.min(xc, axis=0)
        c_max = np.max(xc, axis=0)
        x_center = (c_max + c_min) / 2.0
        x_scale = (c_max - c_min) / 2.0
        x_scale[x_scale < 1e-8] = 1.0
        y_min = np.min(yc, axis=0)
        y_max = np.max(yc, axis=0)
        y_center = (y_max + y_min) / 2.0
        y_scale = (y_max - y_min) / 2.0
        y_scale[y_scale < 1e-8] = 1.0
        xc_norm = (xc - x_center) / x_scale
        yc_norm = (yc - y_center) / y_scale
        xt_norm = (xt - x_center) / x_scale
        yt_norm = (yt - y_center) / y_scale
        xc_norm = np.clip(xc_norm, -5, 5)
        xt_norm = np.clip(xt_norm, -5, 5)

    elif "wiski_style" in normalisation_mode:
        # ... (Wiski block unchanged) ...
        x_center = np.mean(x_shuffled, axis=0)
        x_scale = np.std(x_shuffled, axis=0)
        x_scale[x_scale < 1e-8] = 1.0
        xc_norm = (xc - x_center) / x_scale
        xt_norm = (xt - x_center) / x_scale
        if "zscore_targets" in normalisation_mode:
            y_center = np.mean(y_shuffled, axis=0)
            y_scale = np.std(y_shuffled, axis=0)
        elif "iqr_targets" in normalisation_mode:
            y_center = np.median(y_shuffled, axis=0)
            q75 = np.percentile(y_shuffled, 75, axis=0)
            q25 = np.percentile(y_shuffled, 25, axis=0)
            y_scale = q75 - q25
        y_scale[y_scale < 1e-8] = 1.0
        yc_norm = (yc - y_center) / y_scale
        yt_norm = (yt - y_center) / y_scale

    elif "calibration" in normalisation_mode:
        n_calibration = max(200, int(0.2 * nc))

        # A. INPUTS (X) - Compute Fixed Stats on Head
        x_calib = xc[:n_calibration]
        x_center = np.mean(x_calib, axis=0)
        x_scale = np.std(x_calib, axis=0)
        x_scale[x_scale < 1e-5] = 1.0  # Safety Floor

        # Normalize
        xc_norm = (xc - x_center) / x_scale
        xt_norm = (xt - x_center) / x_scale

        # Clip & Slice (Remove calibration head)
        xc_norm = np.clip(xc_norm[n_calibration:], -5.0, 5.0)
        xt_norm = np.clip(xt_norm, -5.0, 5.0)

        # B. TARGETS (Y) - Compute Fixed Stats on Head
        if "zscore_targets" in normalisation_mode:
            y_calib = yc[:n_calibration]
            y_center = np.mean(y_calib, axis=0)
            y_scale = np.std(y_calib, axis=0)
        elif "iqr_targets" in normalisation_mode:
            y_calib = yc[:n_calibration]
            y_center = np.median(y_calib, axis=0)
            q75 = np.percentile(y_calib, 75, axis=0)
            q25 = np.percentile(y_calib, 25, axis=0)
            y_scale = q75 - q25

        y_scale[y_scale < 1e-5] = 1.0

        # Normalize & Slice
        yc_norm = (yc - y_center) / y_scale
        yc_norm = yc_norm[n_calibration:]  # <--- MATCHING SLICE
        yt_norm = (yt - y_center) / y_scale

        # IMPORTANT: Adjust pre_train_idx
        pre_train_idx = max(0, pre_train_idx - n_calibration)

    # --- UPDATED ONLINE MODE ---
    elif "online" in normalisation_mode:
        # 1. INPUTS (X): Warm-Start Online Normalization

        # A. Determine Calibration "Warm Up" Size
        # We need enough data to get a stable initial variance (at least 200 or 10%)
        n_warmup = max(200, int(0.2 * nc))
        if n_warmup > len(x_shuffled):
            n_warmup = len(x_shuffled)  # Safety

        # --- 1. INPUTS (X) - Warm-Start Online Z-Score ---
        df_x = pd.DataFrame(x_shuffled)

        # Expanding stats (shifted by 1)
        x_mean_running = df_x.expanding().mean().shift(1)
        x_std_running = df_x.expanding().std(ddof=1).shift(1)

        # Warm-up overwrite (Stabilize the start)
        x_warmup = x_shuffled[:n_warmup]
        x_w_mean = np.mean(x_warmup, axis=0)
        x_w_std = np.std(x_warmup, axis=0)
        x_w_std[x_w_std < 1e-5] = 1.0

        x_mean_running.iloc[:n_warmup] = x_w_mean
        x_std_running.iloc[:n_warmup] = x_w_std

        # Cleanup NaNs
        x_mean_running = x_mean_running.fillna(0.0).values
        x_std_running = x_std_running.fillna(1.0).values
        x_std_running[x_std_running < 1e-8] = 1.0

        # Apply
        x_norm_full = (x_shuffled - x_mean_running) / x_std_running
        x_norm_full = np.clip(x_norm_full, -5.0, 5.0)
        xc_norm = x_norm_full[n_warmup:nc]
        xt_norm = x_norm_full[nc:]

        # PART B: TARGETS (Y) - Fixed Z-Score Calibration
        # -----------------------------------------------
        # We define the "unit" of error based on the initial calibration set.
        # This ensures LL at t=0 is comparable to LL at t=1000.

        y_calib = yc[:n_warmup]

        # Z-Score Statistics (Mean & Std)
        y_center = np.mean(y_calib, axis=0)
        y_scale = np.std(y_calib, axis=0)
        y_scale[y_scale < 1e-5] = 1.0  # Avoid division by zero

        # Apply FIXED scaling to the whole stream
        yc_norm = (yc - y_center) / y_scale
        yc_norm = yc_norm[n_warmup:nc]  # Remove warm-up points from context set
        yt_norm = (yt - y_center) / y_scale

        # Dummy stats for X (since X is dynamic)
        x_center = np.zeros(dx)
        x_scale = np.ones(dx)

        pre_train_idx = max(0, pre_train_idx - n_warmup)

    # Padding
    if dx < max_features:
        pad_width = max_features - dx
        xc_norm = np.pad(xc_norm, ((0, 0), (0, pad_width)), mode="constant")
        xt_norm = np.pad(xt_norm, ((0, 0), (0, pad_width)), mode="constant")

    x_full = np.concatenate([xc_norm, xt_norm], axis=0)
    y_full = np.concatenate([yc_norm, yt_norm], axis=0)

    # Convert to Tensor (CPU)
    batch_obj = Batch(
        xc=torch.from_numpy(np.stack([xc_norm])).float(),
        yc=torch.from_numpy(np.stack([yc_norm])).float(),
        xt=torch.from_numpy(np.stack([xt_norm])).float(),
        yt=torch.from_numpy(np.stack([yt_norm])).float(),
        x=torch.from_numpy(np.stack([x_full])).float(),
        y=torch.from_numpy(np.stack([y_full])).float(),
    )
    pre_train_idx = int(round(batch_obj.xc.shape[1] * pre_train_prop)) # Updates pre training index

    # Attach Stats
    batch_obj.pre_pad_dx = dx
    batch_obj.x_mean = torch.from_numpy(np.stack([x_center])).float().to(device)
    batch_obj.x_std = torch.from_numpy(np.stack([x_scale])).float().to(device)
    batch_obj.y_mean = (
        torch.from_numpy(np.stack([y_center])).float().to(device).unsqueeze(-1)
    )
    batch_obj.y_std = (
        torch.from_numpy(np.stack([y_scale])).float().to(device).unsqueeze(-1)
    )
    batch_obj.pre_train_items = pre_train_idx
    return batch_obj


def measure_perf(callable_function, measure_time: bool):
    runtimes = []
    runtime_ms = -1
    if measure_time:
        starter, ender = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        torch.cuda.synchronize()
        starter.record()
    ret = callable_function()
    if measure_time:
        ender.record()
        torch.cuda.synchronize()
        runtime_ms = starter.elapsed_time(ender)
    runtimes.append(runtime_ms)
    runtime_ms_mean = np.mean(np.array(runtimes))
    return runtime_ms_mean, ret


# Streams updates for the given model, recording the key information to be plotted in the provided arrays
def stream_trial(
    model,
    batch,
    nlls,
    rmses,
    query_times,
    condition_times,
    eval_mode,
    device,
    model_name,
    use_flash,
    cache_mhca,
    denormalise,
):
    assert eval_mode in ("normal", "normal_cast"), "invalid eval mode"
    i = 0
    lower, upper = 0, 1
    if model_name == "WISKI":
        upper = batch.pre_train_items
    m, nc, dx = batch.xc.shape
    _, nt, dy = batch.yt.shape
    grad_manager = torch.no_grad() if model_name != "WISKI" else nullcontext()
    cast_manager = (
        torch.autocast(device_type=device, dtype=torch.bfloat16)
        if eval_mode == "normal_cast" and model_name != "WISKI"
        else nullcontext()
    )
    xt, yt = batch.xt.to(device=device), batch.yt.to(device=device)
    if model_name == "WISKI":
        model.pre_pad_dx = (
            batch.pre_pad_dx
        )  # Denotes the size of the features actually needed to be learned
    if denormalise:
        yt_denorm = denormalize_targets(yt, batch)
    else:
        yt_denorm = yt
    while upper <= nc:
        with grad_manager, cast_manager:
            if isinstance(model, IncUpdateEff):
                if lower == 0:
                    model.init_inc_structs(
                        m=m,
                        max_nc=nc + nt,
                        device=device,
                        use_flash=use_flash,
                        cache_mhca=cache_mhca,
                    )
                xc, yc = (
                    batch.xc[:, lower:upper, :].to(device=device),
                    batch.yc[:, lower:upper, :].to(device=device),
                )
                condition_t, _ = measure_perf(
                    lambda: model.update_ctx(
                        xc=xc, yc=yc, use_flash=use_flash, cache_mhca=cache_mhca
                    ),
                    measure_time=True,
                )
                query_t, pred_dist = measure_perf(
                    lambda: model.query(
                        xt=xt, dy=dy, use_flash=use_flash, cache_mhca=cache_mhca
                    ),
                    measure_time=True,
                )
            else:
                condition_t = float("nan")  # Signals no conditioning step possible
                xc, yc = (
                    batch.xc[:, :upper, :].to(device=device),
                    batch.yc[:, :upper, :].to(device=device),
                )
                x, y = (
                    torch.cat((xc, xt), dim=1).to(device=device),
                    torch.cat((yc, yt), dim=1).to(device=device),
                )
                batch_eval = Batch(x=x, y=y, xc=xc, yc=yc, xt=xt, yt=yt)
                query_t, pred_dist = measure_perf(
                    lambda: np_pred_fn(model, batch_eval), measure_time=True
                )
        if denormalise:
            pred_dist = denormalize_predictions(pred_dist, batch)
        nll_t = -(
            pred_dist.log_prob(yt_denorm).sum() / yt_denorm[..., 0].numel()
        ).item()
        rmse_t = (
            nn.functional.mse_loss(pred_dist.mean, yt_denorm).sqrt().cpu().mean()
        ).item()
        # Writes results to np arrays
        i = upper - 1
        nlls[i], rmses[i], query_times[i], condition_times[i] = (
            nll_t,
            rmse_t,
            query_t,
            condition_t,
        )
        # Loop increment
        lower = upper
        upper += 1


# Rolling median for timings to remove some outliers, using nanmedian to handle missing start data
def rolling_median(x, window):
    if window % 2 == 0:
        window += 1

    pad_width = window // 2
    # mode='constant', constant_values=np.nan is safer than reflect for arrays starting with NaNs
    x_padded = np.pad(x, pad_width, mode="constant", constant_values=np.nan)
    shape = x.shape[:-1] + (x.shape[-1], window)
    strides = x_padded.strides + (x_padded.strides[-1],)
    strided_view = np.lib.stride_tricks.as_strided(
        x_padded, shape=shape, strides=strides
    )
    return np.nanmedian(strided_view, axis=-1)


# Reads streaming logged results and produces plots
def plot_tabular_streaming_results(results_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    show_all_legends = True

    # Define stats file path and initialize it (clear previous content)
    stats_file_path = os.path.join(out_dir, "pretrain_costs.txt")
    with open(stats_file_path, "w") as f:
        f.write("Pre-training / First Batch Computational Costs\n")
        f.write("==============================================\n\n")

    if not os.path.exists(results_dir):
        print(f"Results directory {results_dir} does not exist.")
        return

    dataset_dirs = [
        d
        for d in os.listdir(results_dir)
        if os.path.isdir(os.path.join(results_dir, d))
    ]

    if not dataset_dirs:
        print("No dataset directories found.")
        return

    metrics = ["NLL", "RMSE", "Time_Total", "Time_Detailed"]

    for dataset_name in dataset_dirs:
        dataset_path = os.path.join(results_dir, dataset_name)
        model_files = [f for f in os.listdir(dataset_path) if f.endswith(".npz")]

        if not model_files:
            continue

        print(f"Processing {dataset_name}...")

        sorted_model_files = sorted(model_files)
        prop_cycle = plt.rcParams["axes.prop_cycle"]
        colors = prop_cycle.by_key()["color"]

        for metric in metrics:
            fig, ax = plt.subplots(figsize=(2.5, 2.5), constrained_layout=True)

            # Track min and max x values to tighten the axis later
            min_step_start = None
            max_step_end = None

            for idx, model_file in enumerate(sorted_model_files):
                model_name = os.path.splitext(model_file)[0]
                filepath = os.path.join(dataset_path, model_file)
                color = colors[idx % len(colors)]

                try:
                    data = np.load(filepath)

                    # Compute Means/Stds (using nan-safe functions)
                    # axis=0 is across trials
                    mean_rmse = np.nanmean(data["rmses"], axis=0)
                    std_rmse = np.nanstd(data["rmses"], axis=0, ddof=1)

                    mean_nll = np.nanmean(data["nlls"], axis=0)
                    std_nll = np.nanstd(data["nlls"], axis=0, ddof=1)

                    mean_query_time = np.nanmean(data["query_times"], axis=0)
                    mean_condition_time = np.nanmean(data["condition_times"], axis=0)

                    # Generate X-axis
                    # Since arrays are aligned such that Index 0 = 1 Context Point:
                    n_steps = len(mean_nll)
                    steps = np.arange(1, n_steps + 1)

                    # Identify valid data range (for axis scaling)
                    valid_indices = np.where(~np.isnan(mean_nll))[0]
                    if len(valid_indices) > 0:
                        first_valid = steps[valid_indices[0]]
                        last_valid = steps[valid_indices[-1]]

                        if min_step_start is None or first_valid < min_step_start:
                            min_step_start = first_valid
                        if max_step_end is None or last_valid > max_step_end:
                            max_step_end = last_valid

                    # Write Stats to File & Console
                    if metric == metrics[0]:
                        # Find the first index where data actually exists
                        valid_time_indices = np.where(~np.isnan(mean_query_time))[0]

                        if len(valid_time_indices) > 0:
                            first_idx = valid_time_indices[0]

                            pt_cond = data["condition_times"][:, first_idx]
                            pt_query = data["query_times"][:, first_idx]

                            # Calculate Condition Stats
                            # Filter NaNs for calculation (in case some specific trials failed)
                            pt_cond_clean = pt_cond[~np.isnan(pt_cond)]
                            if len(pt_cond_clean) == 0:
                                cond_str = "N/A"
                            else:
                                c_mean = np.mean(pt_cond_clean)
                                c_std = np.std(pt_cond_clean, ddof=1)
                                cond_str = f"{c_mean:.2f} ± {c_std:.2f} ms"

                            # Calculate Query Stats
                            pt_query_clean = pt_query[~np.isnan(pt_query)]
                            q_mean = np.mean(pt_query_clean)
                            q_std = np.std(pt_query_clean, ddof=1)

                            stats_output = (
                                f"[{dataset_name}] {model_name} First Valid Batch (N_c={first_idx + 1}):\n"
                                f"  Condition Time: {cond_str}\n"
                                f"  Query Time:     {q_mean:.2f} ± {q_std:.2f} ms\n"
                                f"{'-' * 30}\n"
                            )
                            print(stats_output, end="")
                            with open(stats_file_path, "a") as f:
                                f.write(stats_output)

                    if metric == "NLL":
                        ll = -mean_nll
                        ax.plot(steps, ll, label=model_name, color=color)
                        ax.fill_between(
                            steps,
                            ll - 2 * std_nll,
                            ll + 2 * std_nll,
                            color=color,
                            alpha=0.2,
                            linewidth=0,
                        )
                        ax.set_ylabel("LL")

                        if dataset_name == "Skillcraft": ax.set_ylim(bottom=-1.10, top=-0.90)
                        elif dataset_name == "Powerplant": ax.set_ylim(bottom=-0.1, top=0.0)
                        elif dataset_name == "Elevators": ax.set_ylim(bottom=-0.8, top=-0.3)
                        elif dataset_name == "Protein": ax.set_ylim(bottom=-1.3, top=-1.1)
                        elif dataset_name == "Synthetic": ax.set_ylim(bottom=-0.5, top=-0.1)



                    elif metric == "RMSE":
                        ax.plot(steps, mean_rmse, label=model_name, color=color)
                        ax.fill_between(
                            steps,
                            mean_rmse - 2 * std_rmse,
                            mean_rmse + 2 * std_rmse,
                            color=color,
                            alpha=0.2,
                            linewidth=0,
                        )
                        ax.set_ylabel("RMSE")
                        #ax.set_ylim(bottom=0.8, top=1.0)

                    elif "Time" in metric:
                        data_mask = ~np.isnan(mean_query_time)

                        safe_cond = np.nan_to_num(mean_condition_time, nan=0.0)
                        safe_query = np.nan_to_num(mean_query_time, nan=0.0)

                        raw_total = safe_cond + safe_query
                        raw_total[~data_mask] = np.nan
                        filtered_total = rolling_median(raw_total, window=7)
                        filtered_total[~data_mask] = np.nan

                        ax.plot(
                            steps,
                            filtered_total,
                            label=model_name,
                            color=color,
                            linestyle="-",
                        )

                        if metric == "Time_Detailed":
                            has_real_cond = (
                                np.nanmax(mean_condition_time) > 0
                                if not np.all(np.isnan(mean_condition_time))
                                else False
                            )

                            if has_real_cond:
                                filtered_cond = rolling_median(
                                    mean_condition_time, window=7
                                )
                                filtered_cond[~data_mask] = np.nan
                                ax.plot(
                                    steps,
                                    filtered_cond,
                                    color=color,
                                    linestyle="--",
                                    alpha=0.4,
                                    linewidth=1.0,
                                )

                        ax.set_ylabel("Time (ms)")

                except Exception as e:
                    print(
                        f"Error plotting {model_name} in {dataset_name} ({metric}): {e}"
                    )
                    continue

            # --- Formatting ---
            ax.set_xlabel(r"$N_s$")
            ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
            ax.tick_params(axis="both", which="both", length=0)

            # Set X-Axis Limits Exactly based on data range
            if min_step_start is not None and max_step_end is not None:
                ax.set_xlim(left=min_step_start, right=max_step_end)

            if (dataset_name == "Skillcraft" and metric == "NLL") or show_all_legends:
                ax.legend(loc="upper right", frameon=True, fontsize="xx-small")

            filename = f"{dataset_name}_{metric}.pdf"
            save_path = os.path.join(out_dir, filename)
            plt.savefig(save_path, bbox_inches="tight")
            print(f"Saved {save_path}")
            plt.close(fig)


# Loads the correct wiski model based on the dataset
def load_wiski_model(dataset_name):
     assert dataset_name in (
         "Skillcraft",
         "Powerplant",
         "Elevators",
         "Protein",
         "Synthetic",
     ), "invalid dataset specified"
     if dataset_name == "Skillcraft":
        return SkillcraftWiski()
     elif dataset_name == "Powerplant":
        return PowerplantWiski()
     elif dataset_name == "Elevators":
        return ElevatorsWiski()
     elif dataset_name == "Protein":
        return ProteinWiski()
     elif dataset_name == "Synthetic":
        return GenericWiski()


# Main function that coordinates the plot generation
def streamed_real_tab_main(
    dataset_to_run=None, no_trials=5, normalisation_mode="wiski_style_zscore_targets"
):
    # --- Hypers -------
    seed = 1
    no_trials = no_trials  # Number of trials to aggregate over to obtain standard dev
    train_split = 0.8  # Percent of overall data used to train model (that is the ctx). Remaining 1.0 - train_split used as targets
    pre_train_prop = 0.05  # Percentage of training data used to pretrain models. In case of WISKI this is the pretraining phase
    out_dir = "experiments/stream_tab/updated"  # Output folder
    max_features = 20
    device = "cuda"
    eval_mode = "normal_cast"  # Either eval normal or use autocast to float16 (helps with flash attention)
    use_flash = True
    cache_mhca = True
    denormalise = False
    run_experiments = False  # Allows to skip the experiments and just do the plotting
    # ----- end of hypers -----
    os.makedirs(out_dir, exist_ok=True)
    pl.seed_everything(seed)
    results_path = f"{out_dir}/results"
    os.makedirs(results_path, exist_ok=True)
    if run_experiments:
        model_list = get_model_list()
        all_uci_sets = get_uci_sets()

        if dataset_to_run:
            # Filter the list case-insensitively
            uci_sets = [
                s for s in all_uci_sets if s[1].lower() == dataset_to_run.lower()
            ]
            if not uci_sets:
                valid_names = [s[1] for s in all_uci_sets]
                raise ValueError(
                    f"Dataset '{dataset_to_run}' not found. Available: {valid_names}"
                )
        else:
            uci_sets = all_uci_sets
        for data_path, data_name in uci_sets:
            print(f"{data_name} dataset")
            data = load_uci_dataset(data_path, data_name, seed)
            for (model_yml, model_ckpt, model_name) in model_list:
                print(f"{model_name} on {data_name}")
                nlls, rmses, query_times, condition_times = None, None, None, None
                if model_name == "WISKI": model = load_wiski_model(data_name)
                else: model = get_model(model_yml, model_ckpt, seed=False, local_weights=True)
                if eval_mode == "normal_cast" and model_name != "WISKI": model.to(dtype=torch.bfloat16)
                for trial_i in range(no_trials):
                    pl.seed_everything(seed + trial_i)
                    batch = load_batch(
                        data,
                        train_split,
                        pre_train_prop,
                        max_features,
                        device,
                        normalisation_mode,
                        data_name,
                    )
                    if nlls is None:
                        # ts = batch.xc.shape[1] - batch.pre_train_items + 1
                        ts = batch.xc.shape[1]
                        nlls, rmses, query_times, condition_times = (
                            np.full((no_trials, ts), np.nan),
                            np.full((no_trials, ts), np.nan),
                            np.full((no_trials, ts), np.nan),
                            np.full((no_trials, ts), np.nan),
                        )
                    stream_trial(
                        model,
                        batch,
                        nlls[trial_i],
                        rmses[trial_i],
                        query_times[trial_i],
                        condition_times[trial_i],
                        eval_mode,
                        device,
                        model_name,
                        use_flash,
                        cache_mhca,
                        denormalise,
                    )
                # Writes result
                base_data_path = f"{results_path}/{data_name}"
                os.makedirs(base_data_path, exist_ok=True)
                np.savez(
                    f"{base_data_path}/{model_name}_{normalisation_mode}_V2.npz",
                    rmses=rmses,
                    nlls=nlls,
                    query_times=query_times,
                    condition_times=condition_times,
                    pre_train_items=np.array([batch.pre_train_items]),
                )
                del model
                gc.collect()
                torch.cuda.empty_cache()
            plot_tabular_streaming_results(results_dir=results_path, out_dir=f"{out_dir}/plot")
    plot_tabular_streaming_results(results_dir=results_path, out_dir=f"{out_dir}/plot")


if __name__ == "__main__":
    dataset = None
    trials = 10
    norm_mode = "online_zscore"

    streamed_real_tab_main(
        dataset_to_run=dataset,
        no_trials=trials,
        normalisation_mode=norm_mode,
    )
