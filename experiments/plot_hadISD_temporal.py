# HadISD plotting (temporal)
import argparse
import datetime
import os
import sys
import warnings
from pathlib import Path
from typing import Callable, Optional, List, Tuple
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
from torch import nn
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import wandb
from tnp.utils.experiment_utils import initialize_experiment
from tnp.utils.np_functions import np_pred_fn
from tnp.data.hadISDTemporal import TemporalHadISDBatch, normalise_time, scale_pred_temp_dist, get_true_temp
from data_temp.data_processing.elevations import get_cached_elevation_grid


matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"



def convert_time_to_str(unnorm_time: int):
    ZERO_TIME = datetime.datetime(1931, 1, 1)
    try:
        final_datetime = ZERO_TIME + datetime.timedelta(hours=float(unnorm_time))
        return final_datetime.strftime("%H:00 %d %B %Y")
    except Exception:
        return f"Hours={unnorm_time}"


def _context_block_time(t_star: float, block_index_0_is_oldest: int, delta_hours: int, H: int) -> float:
    k = int(block_index_0_is_oldest)
    lag_steps = H - k
    return float(t_star) - float(delta_hours * lag_steps)


def _unnormalise_lat_lon(batch: TemporalHadISDBatch, lat_norm: np.ndarray, lon_norm: np.ndarray):
    lat_lo, lat_hi = batch.lat_range
    lon_lo, lon_hi = batch.long_range
    lat_deg = ((lat_norm + 1.0) / 2.0) * (lat_hi - lat_lo) + lat_lo
    lon_deg = ((lon_norm + 1.0) / 2.0) * (lon_hi - lon_lo) + lon_lo
    return lat_deg, lon_deg


# Earth map to plot with
def init_earth_fig(title, figsize, proj, lat_range, long_range, height_data):
    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection=proj)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS)
    ax.set_extent([*long_range, *lat_range], crs=proj)
    if height_data is not None:
        lon_mesh, lat_mesh, elev_np = height_data
        ax.pcolormesh(lon_mesh, lat_mesh, elev_np, cmap="terrain", shading="auto")
    ax.set_title(title)
    return fig, ax


def save_plot(fig, name, i, panel, logging, savefig):
    tag = f"{name}/{i:03d}_{panel}"
    if wandb.run is not None and logging:
        wandb.log({tag: wandb.Image(fig)})
    elif savefig:
        base_folder = f"{name}"
        save_name = base_folder + f"/{i:03d}_{panel}.png"
        if not os.path.isdir(base_folder):
            os.makedirs(base_folder)
        fig.savefig(save_name, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)

# Logs some useful stats
def save_stats(stats_dict, name, i, panel, logging, savefig):
    if wandb.run is not None and logging:
        wandb_data = {f"{name}/{panel}_{key}": val for key, val in stats_dict.items()}
        wandb.log(wandb_data)
    elif savefig:
        base_folder = f"{name}"
        if not os.path.isdir(base_folder):
            os.makedirs(base_folder)
        file_path = f"{base_folder}/{i:03d}_{panel}.txt"       
        with open(file_path, "w") as f:
            f.write(f"--- Statistics for Step {i} ({panel}) ---\n")
            for key, val in stats_dict.items():
                f.write(f"{key}: {val:.6f}\n")
    else:
        print(f"--- Statistics for Step {i} ({panel}) ---\n")
        for key, val in stats_dict.items():
            print(f"{key}: {val:.6f}\n")

# Plots H sequences and target panel together
def _plot_temporal_sequence(
    *,
    panel: str,
    name: str,
    i: int,
    logging: bool,
    savefig: bool,
    proj,
    figsize: Tuple[float, float],
    batch: TemporalHadISDBatch,
    height_data,
    lon_ctx: np.ndarray,
    lat_ctx: np.ndarray,
    yc_true: np.ndarray,
    lon_tgt: np.ndarray,
    lat_tgt: np.ndarray,
    tgt_vals: np.ndarray,
    t_star_val: float,
    time_ctx: np.ndarray,
    time_tgt: np.ndarray,
    delta_hours: int,
    vmin: float,
    vmax: float,
    title_suffix: str,
    cmap: str,
    norm=None,
):

    Nc_total = int(lon_ctx.size)
    Nt_total = int(lon_tgt.size)
    if Nc_total == 0 and Nt_total == 0:
        return

    # Ensure 1D numpy arrays
    lon_ctx = np.asarray(lon_ctx).reshape(-1)
    lat_ctx = np.asarray(lat_ctx).reshape(-1)
    yc_true = np.asarray(yc_true).reshape(-1)

    lon_tgt = np.asarray(lon_tgt).reshape(-1)
    lat_tgt = np.asarray(lat_tgt).reshape(-1)
    tgt_vals = np.asarray(tgt_vals).reshape(-1)

    time_ctx = np.asarray(time_ctx).reshape(-1) if time_ctx is not None else np.array([], dtype=np.float64)
    time_tgt = np.asarray(time_tgt).reshape(-1) if time_tgt is not None else np.array([], dtype=np.float64)

    if Nc_total != len(time_ctx):
        # If the caller forgot to pass time_ctx correctly, bail safely.
        return
    if Nt_total != len(time_tgt):
        return

    # Round/convert times to int hours for grouping (your loader uses integer hours)
    t_ctx_int = np.round(time_ctx).astype(np.int64) if Nc_total > 0 else np.array([], dtype=np.int64)
    t_tgt_int = np.round(time_tgt).astype(np.int64) if Nt_total > 0 else np.array([], dtype=np.int64)

    # Preserve *stream order* of timesteps (not sorted), but ensure uniqueness
    def unique_in_order(arr: np.ndarray) -> List[int]:
        seen = set()
        out = []
        for x in arr.tolist():
            if x not in seen:
                seen.add(x)
                out.append(int(x))
        return out

    ctx_times = unique_in_order(t_ctx_int) if Nc_total > 0 else []
    tgt_times = unique_in_order(t_tgt_int) if Nt_total > 0 else []

    # If you want chronological within each section, uncomment:
    # ctx_times = sorted(ctx_times)
    # tgt_times = sorted(tgt_times)

    n_ctx_panels = len(ctx_times)
    n_tgt_panels = len(tgt_times)
    ncols = max(1, n_ctx_panels + n_tgt_panels)

    # Dynamic figure width
    fig = plt.figure(figsize=(3.0 * ncols, 4.0))
    gs = fig.add_gridspec(1, ncols, wspace=0.05)

    # Helper to init a subplot with the same basemap logic
    def _init_ax(cell):
        ax = fig.add_subplot(cell, projection=proj)
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS)
        ax.set_extent([*batch.long_range, *batch.lat_range], crs=proj)
        if height_data is not None:
            lon_mesh, lat_mesh, elev_np = height_data
            ax.pcolormesh(lon_mesh, lat_mesh, elev_np, cmap="terrain", shading="auto")
        return ax

    last_mappable = None

    col = 0
    for k, t_h in enumerate(ctx_times):
        idx = np.nonzero(t_ctx_int == t_h)[0]
        if idx.size == 0:
            continue

        ax = _init_ax(gs[0, col])

        sc = ax.scatter(
            lon_ctx[idx],
            lat_ctx[idx],
            c=yc_true[idx],
            cmap="coolwarm",
            s=15,
            vmin=vmin,
            vmax=vmax,
            edgecolors="k",
            linewidth=0.2,
        )
        last_mappable = sc

        t_str = convert_time_to_str(int(t_h))
        ax.set_title(f"Ctx {k+1}/{n_ctx_panels}\n{t_str}", fontsize=8)

        col += 1

    for k, t_h in enumerate(tgt_times):
        idx = np.nonzero(t_tgt_int == t_h)[0]
        if idx.size == 0:
            continue

        ax = _init_ax(gs[0, col])

        sc_t = ax.scatter(
            lon_tgt[idx],
            lat_tgt[idx],
            c=tgt_vals[idx],
            cmap=cmap,
            s=30,
            vmin=vmin if norm is None else None,
            vmax=vmax if norm is None else None,
            norm=norm,
            marker="x",
        )
        last_mappable = sc_t

        t_str = convert_time_to_str(int(t_h))
        ax.set_title(f"Tgt {k+1}/{n_tgt_panels}\n{t_str}", fontsize=8, fontweight="bold")

        col += 1

    if last_mappable is not None:
        cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
        fig.colorbar(
            last_mappable,
            cax=cbar_ax,
            label="Temp (°C)" if panel in ("F", "G") else "Value",
        )

    t_star_str = convert_time_to_str(int(round(t_star_val)))
    fig.suptitle(
        f"{panel}) Temporal Stream Sequence (t*={t_star_str}, Δ={delta_hours}h)",
        fontsize=12,
    )

    save_plot(fig, name, i, panel, logging, savefig)


# Main plotting function
# Produces plots:
# A) Context vs Target stations
# B) Predicted targets (with context shown)
# C) True targets (with context shown)
# D) Error targets (True - Pred) (with context shown)
# E) Gridded predictions (optional)
# F) Temporal sequence: context temps + true targets
# G) Temporal sequence: context temps + predicted target
# H) Temporal sequence: context temps + error target
@torch.no_grad()
def plot_hadISD_temporal(
    model: nn.Module,
    batches: List[TemporalHadISDBatch],
    lat_mesh: np.ndarray,
    lon_mesh: np.ndarray,
    elev_np: np.ndarray,
    num_fig: int = 5,
    figsize: Tuple[float, float] = (8.0, 6.0),
    name: str = "temporal_plot",
    savefig: bool = False,
    logging: bool = True,
    model_lbl: str = "Model",
    pred_fn: Callable = np_pred_fn,
    huge_grid_plots: bool = True,
    device=None,
    delta_hours: Optional[int] = None,
    H: Optional[int] = None,
):
    proj = ccrs.PlateCarree()
    height_data = (lon_mesh, lat_mesh, elev_np) if (lon_mesh is not None and lat_mesh is not None and elev_np is not None) else None

    for i in range(min(num_fig, len(batches))):
        batch = batches[i]
        BATCH_IDX = 0

        xc = batch.xc[BATCH_IDX:BATCH_IDX + 1]
        yc = batch.yc[BATCH_IDX:BATCH_IDX + 1]
        xt = batch.xt[BATCH_IDX:BATCH_IDX + 1]
        yt = batch.yt[BATCH_IDX:BATCH_IDX + 1]
        if device is not None:
            xc, yc, xt, yt = xc.to(device), yc.to(device), xt.to(device), yt.to(device)
        plot_device = xc.device

        unnorm_time = batch.unnormalised_time[BATCH_IDX]
        t_star_val = float(unnorm_time.detach().cpu().item())
        batch_time_str = convert_time_to_str(int(round(t_star_val)))

        batch_pred = TemporalHadISDBatch(
            x=torch.cat((xc, xt), dim=1),
            y=torch.cat((yc, yt), dim=1),
            xc=xc,
            yc=yc,
            xt=xt,
            yt=yt,
            mean_temp=batch.mean_temp,
            std_temp=batch.std_temp,
            mean_elev=batch.mean_elev,
            std_elev=batch.std_elev,
            lat_range=batch.lat_range,
            long_range=batch.long_range,
            unnormalised_time=unnorm_time.unsqueeze(0),
        )

        # Predict at targets
        yt_pred_dist = pred_fn(model, batch_pred)
        yt_pred_dist = scale_pred_temp_dist(batch_pred, yt_pred_dist)

        pred_mean = yt_pred_dist.mean.detach().cpu().squeeze(0).squeeze(-1).numpy()
        yt_true = get_true_temp(batch_pred, batch_pred.yt)
        true_vals = yt_true.detach().cpu().squeeze(0).squeeze(-1).numpy()

        yc_true = get_true_temp(batch_pred, batch_pred.yc).detach().cpu().squeeze(0).squeeze(-1).numpy()

        # Metrics
        rmse = float(np.sqrt(np.mean((pred_mean - true_vals) ** 2)))
        nll = float((-yt_pred_dist.log_prob(yt_true)).mean().detach().cpu().item())

        # Unnormalise station coords
        xc_np = xc.detach().cpu().squeeze(0).numpy()
        xt_np = xt.detach().cpu().squeeze(0).numpy()
        lat_ctx, lon_ctx = _unnormalise_lat_lon(batch, xc_np[:, 0], xc_np[:, 1])
        lat_tgt, lon_tgt = _unnormalise_lat_lon(batch, xt_np[:, 0], xt_np[:, 1])

        time_ctx = xc_np[:, 3]
        time_tgt = xt_np[:, 3]

        # Shared range for temperature-colored plots
        all_temps = np.concatenate([true_vals, pred_mean, yc_true])
        vmin = float(np.nanmin(all_temps))
        vmax = float(np.nanmax(all_temps))

        # Logs uncertainty stats for verification
        if False:
            sigma_flat = yt_pred_dist.stddev.detach().cpu().numpy().flatten()
            num_nans = np.isnan(sigma_flat).sum()
            num_infs = np.isinf(sigma_flat).sum()
            if num_nans > 0 or num_infs > 0:
                print(f"WARNING: Found {num_nans} NaNs and {num_infs} Infs in uncertainty map!")
            stats = {
                "sigma_min": float(sigma_flat.min()),
                "sigma_max": float(sigma_flat.max()),
                "sigma_mean": float(sigma_flat.mean()),
                "sigma_med": float(np.median(sigma_flat)),
                "sigma_spread": float(sigma_flat.std()), 
                "sigma_p05": float(np.percentile(sigma_flat, 5)),
                "sigma_p95": float(np.percentile(sigma_flat, 95)),
                "num_nans": int(num_nans),
                "num_infs": int(num_infs)
            }
            save_stats(stats, name, i, "UNC", logging, savefig)

        # A) Context vs Target stations
        title_a = f"NC={xc.shape[1]} NT={xt.shape[1]} - {batch_time_str}"
        fig_a, ax_a = init_earth_fig(title_a, figsize, proj, batch.lat_range, batch.long_range, height_data)
        ax_a.scatter(lon_ctx, lat_ctx, c="k", s=10, label="Context")
        ax_a.scatter(lon_tgt, lat_tgt, c="r", s=10, label="Target")
        ax_a.legend()
        save_plot(fig_a, name, i, "A", logging, savefig)

        # B) Predicted with context stations
        title_b = f"Predicted Temperature RMSE={rmse:.2f} NLL={nll:.3f} - {batch_time_str}"
        fig_b, ax_b = init_earth_fig(title_b, figsize, proj, batch.lat_range, batch.long_range, height_data)
        ax_b.scatter(lon_ctx, lat_ctx, c="k", s=10, label="Context")
        sc_b = ax_b.scatter(lon_tgt, lat_tgt, c=pred_mean, s=20, cmap="coolwarm", vmin=vmin, vmax=vmax)
        cbar = fig_b.colorbar(sc_b, ax=ax_b, orientation="vertical", pad=0.05)
        cbar.set_label("Predicted Temperature (°C)")
        ax_b.legend()
        save_plot(fig_b, name, i, "B", logging, savefig)

        # C) True temperatures
        title_c = f"Recorded Temperature - {batch_time_str}"
        fig_c, ax_c = init_earth_fig(title_c, figsize, proj, batch.lat_range, batch.long_range, height_data)
        ax_c.scatter(lon_ctx, lat_ctx, c="k", s=10, label="Context")
        sc_c = ax_c.scatter(lon_tgt, lat_tgt, c=true_vals, s=20, cmap="coolwarm", vmin=vmin, vmax=vmax)
        cbar = fig_c.colorbar(sc_c, ax=ax_c, orientation="vertical", pad=0.05)
        cbar.set_label("Measured Temperature (°C)")
        ax_c.legend()
        save_plot(fig_c, name, i, "C", logging, savefig)

       # D) Error
        error = (true_vals - pred_mean)
        title_d = f"Prediction Error (True - Pred) RMSE={rmse:.2f} - {batch_time_str}"
        fig_d, ax_d = init_earth_fig(title_d, figsize, proj, batch.lat_range, batch.long_range, height_data)
        ax_d.scatter(lon_ctx, lat_ctx, c="k", s=10, label="Context")
        max_abs_error = float(np.max(np.abs(error))) if error.size else 1.0
        err_norm = matplotlib.colors.TwoSlopeNorm(vcenter=0, vmin=-max_abs_error, vmax=max_abs_error)
        sc_d = ax_d.scatter(lon_tgt, lat_tgt, c=error, s=20, cmap="seismic", norm=err_norm)
        cbar = fig_d.colorbar(sc_d, ax=ax_d, orientation="vertical", pad=0.05)
        cbar.set_label("Prediction Error (°C)")
        ax_d.legend()
        save_plot(fig_d, name, i, "D", logging, savefig)

        # E) Gridded predictions (optional for compute reasons)
        if huge_grid_plots and (lat_mesh is not None) and (lon_mesh is not None) and (elev_np is not None):
            N_POINTS = int(lat_mesh.shape[0])

            # Normalises grid
            lat_norm = 2.0 * (lat_mesh - batch.lat_range[0]) / (batch.lat_range[1] - batch.lat_range[0]) - 1.0
            lon_norm = 2.0 * (lon_mesh - batch.long_range[0]) / (batch.long_range[1] - batch.long_range[0]) - 1.0
            elev_norm = (elev_np - batch.mean_elev) / batch.std_elev

            time_grid = np.full_like(lat_mesh, fill_value=normalise_time(np.array([t_star_val], dtype=np.int64))[0])

            xt_grid = torch.tensor(
                np.stack(
                    [lat_norm.flatten(), lon_norm.flatten(), elev_norm.flatten(), time_grid.flatten()],
                    axis=-1,
                ),
                dtype=torch.float32,
                device=plot_device,
            ).unsqueeze(0)

            batch_grid = TemporalHadISDBatch(
                x=None, y=None, xc=xc, yc=yc, xt=xt_grid, yt=None,
                mean_temp=batch.mean_temp, std_temp=batch.std_temp,
                mean_elev=batch.mean_elev, std_elev=batch.std_elev,
                lat_range=batch.lat_range, long_range=batch.long_range,
                unnormalised_time=unnorm_time.unsqueeze(0),
            )

            yg_dist = pred_fn(model, batch_grid, predict_without_yt_tnpa=True)
            yg_dist = scale_pred_temp_dist(batch_grid, yg_dist)
            predicted_grid_points = yg_dist.mean.squeeze(0).squeeze(-1).view(N_POINTS, N_POINTS).detach().cpu().numpy()

            title_e = f"Gridded Predictions P={N_POINTS*N_POINTS:,} - {batch_time_str}"
            fig_e, ax_e = init_earth_fig(title_e, figsize, proj, batch.lat_range, batch.long_range, height_data)
            pcm = ax_e.pcolormesh(lon_mesh, lat_mesh, predicted_grid_points, cmap="coolwarm", shading="auto", vmin=vmin, vmax=vmax)
            cbar = fig_e.colorbar(pcm, ax=ax_e, orientation="vertical", pad=0.05)
            cbar.set_label("Temperature (°C)")
            ax_e.scatter(lon_ctx, lat_ctx, c="k", s=10, label="Context")
            ax_e.legend()
            save_plot(fig_e, name, i, "E", logging, savefig)

            # Plots predictive uncertainty
            predicted_grid_stddev = yg_dist.stddev.squeeze(0).squeeze(-1).view(N_POINTS, N_POINTS).detach().cpu().numpy()
            title_e2 = f"Uncertainty P={N_POINTS*N_POINTS:,} - {batch_time_str}"
            fig_e2, ax_e2 = init_earth_fig(title_e2, figsize, proj, batch.lat_range, batch.long_range, height_data)
            pcm = ax_e2.pcolormesh(lon_mesh, lat_mesh, predicted_grid_stddev, cmap="viridis", shading="auto", vmin=0)
            cbar = fig_e2.colorbar(pcm, ax=ax_e2, orientation="vertical", pad=0.05)
            cbar.set_label("Standard Deviation")
            ax_e2.scatter(lon_ctx, lat_ctx, c="k", s=10, label="Context")
            ax_e2.legend()
            save_plot(fig_e2, name, i, "E2", logging, savefig)

        # F) context temps + true target
        _plot_temporal_sequence(
            panel="F",
            name=name, i=i, logging=logging, savefig=savefig,
            proj=proj, figsize=figsize,
            batch=batch, height_data=height_data,
            lon_ctx=lon_ctx, lat_ctx=lat_ctx, yc_true=yc_true,
            lon_tgt=lon_tgt, lat_tgt=lat_tgt,
            tgt_vals=true_vals,
            t_star_val=t_star_val, delta_hours=int(delta_hours),
            time_ctx=time_ctx, time_tgt=time_tgt,
            vmin=vmin, vmax=vmax,
            title_suffix=batch_time_str,
            cmap="coolwarm",
            norm=None,
        )
        # G) context temps + predicted targets
        _plot_temporal_sequence(
            panel="G",
            name=name, i=i, logging=logging, savefig=savefig,
            proj=proj, figsize=figsize,
            batch=batch, height_data=height_data,
            lon_ctx=lon_ctx, lat_ctx=lat_ctx, yc_true=yc_true,
            lon_tgt=lon_tgt, lat_tgt=lat_tgt,
            tgt_vals=pred_mean,
            t_star_val=t_star_val, delta_hours=int(delta_hours),
            time_ctx=time_ctx, time_tgt=time_tgt,
            vmin=vmin, vmax=vmax,
            title_suffix=batch_time_str,
            cmap="coolwarm",
            norm=None,
        )

        # H) error temporally
        err_norm = matplotlib.colors.TwoSlopeNorm(vcenter=0, vmin=-max_abs_error, vmax=max_abs_error)
        _plot_temporal_sequence(
            panel="H",
            name=name, i=i, logging=logging, savefig=savefig,
            proj=proj, figsize=figsize,
            batch=batch, height_data=height_data,
            lon_ctx=lon_ctx, lat_ctx=lat_ctx, yc_true=yc_true,
            lon_tgt=lon_tgt, lat_tgt=lat_tgt,
            tgt_vals=error,
            t_star_val=t_star_val, delta_hours=int(delta_hours),
            time_ctx=time_ctx, time_tgt=time_tgt,
            vmin=-max_abs_error, vmax=max_abs_error,
            title_suffix=batch_time_str,
            cmap="seismic",
            norm=err_norm,
        )

# Loads batch from a config and plots to visualise each set
def main(default_config, default_out_dir):
    default_config = ""
    default_out_dir = ""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        nargs="+",
        default=[default_config],
        help=f"Data loader configs. Default: {default_config}",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=default_out_dir,
        help=f"Output directory for plots. Default: {default_out_dir}",
    )
    parser.add_argument("--plots_per_split", type=int, default=10, help="Batches to process per split")
    parser.add_argument("--no_grid", action="store_true", help="Skip grid plots")
    args, _ = parser.parse_known_args()

    sys.argv = [sys.argv[0], "--config", *args.config]
    experiment = initialize_experiment()

    model = experiment.model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Loaded Experiment: {experiment.misc.name}")
    print(f"Outputting to: {out_root}")

    for split in ["train", "val", "test"]:
        if not hasattr(experiment.generators, split):
            continue

        print(f"\n--- Plotting {split.upper()} ---")
        gen = getattr(experiment.generators, split)

        delta_hours = int(gen.delta_hours)

        lat_mesh, lon_mesh, elev_np = None, None, None
        if not args.no_grid:
            try:
                cache_dir = Path(getattr(experiment.misc, "cache_dem_dir", "cache"))
                dem_path = getattr(experiment.misc, "dem_path", None)
                lat_mesh, lon_mesh, elev_np = get_cached_elevation_grid(
                    gen.lat_range,
                    gen.long_range,
                    n_points=experiment.misc.num_grid_points_plot,
                    cache_dir=cache_dir,
                    dem_path=Path(dem_path) if dem_path else None,
                )
            except Exception as e:
                print(f"Warning: Could not load DEM for grid plots: {e}")

        batches = [gen.generate_batch() for _ in range(args.plots_per_split)]

        plot_hadISD_temporal(
            model=model,
            batches=batches,
            lat_mesh=lat_mesh,
            lon_mesh=lon_mesh,
            elev_np=elev_np,
            num_fig=len(batches),
            figsize=(10, 6),
            name=str(out_root / split),
            savefig=True,
            logging=False,
            huge_grid_plots=(not args.no_grid),
            device=device,
            delta_hours=delta_hours,
        )
