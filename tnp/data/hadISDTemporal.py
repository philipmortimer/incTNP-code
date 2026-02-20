# Data loading for HadISD temporal task
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Literal, Optional, Sequence, Tuple
import collections
import json
import random
import warnings
import numpy as np
import pyarrow.parquet as pq
import torch
from .base import Batch, DataGenerator
import datetime


@dataclass
class TemporalHadISDBatch(Batch):
    mean_temp: float
    std_temp: float
    mean_elev: float
    std_elev: float
    unnormalised_time: torch.Tensor  # [M] t0 times
    lat_range: Tuple[float, float]
    long_range: Tuple[float, float]


def normalise_time(x: np.ndarray) -> np.ndarray:
    return x % (365 * 24)


def get_true_temp(batch: TemporalHadISDBatch, y_in: torch.Tensor) -> torch.Tensor:
    return y_in * batch.std_temp + batch.mean_temp


def scale_pred_temp_dist(
    batch: TemporalHadISDBatch,
    pred_dist: torch.distributions.Normal,
) -> torch.distributions.Normal:
    mean_scaled = get_true_temp(batch, pred_dist.mean)
    std_scaled = pred_dist.stddev * batch.std_temp
    return torch.distributions.Normal(loc=mean_scaled, scale=std_scaled)


def get_true_elev(batch: TemporalHadISDBatch, y_in: torch.Tensor) -> torch.Tensor:
    return y_in * batch.std_elev + batch.mean_elev


class TemporalHadISDDataGenerator(DataGenerator):
    def __init__(
        self,
        *,
        split: Literal["train", "val", "test"],
        data_root: str,
        N_t_min: int,
        N_t_max: int,
        N_c_min: int,
        N_c_max: int,
        delta_hours: int,
        h_window: int,
        ordering: Literal["random", "ctx_time", "full_time", "forecasting"],
        max_resamples: int = 10_000_000, # Maximum number of times to resample in gen batch when an error occurs
        **kwargs,
    ):
        super().__init__(**kwargs)

        assert split in ("train", "val", "test"), "Invalid split"
        assert N_t_min >= 1 and N_t_max >= N_t_min
        assert N_c_min >= 1 and N_c_max >= N_c_min
        assert ordering in ("random", "ctx_time", "full_time", "forecasting"), "Invalid time ordering"

        self.split = split
        self.data_root = Path(data_root).resolve()
        self.N_t_min = int(N_t_min)
        self.N_t_max = int(N_t_max)
        self.N_c_min = int(N_c_min)
        self.N_c_max = int(N_c_max)
        self.delta_hours = int(delta_hours)
        self.h_window = int(h_window)
        self.ordering = ordering
        self.max_resamples = int(max_resamples)

        # Loads norm stats
        norm_path = self.data_root / "config.json"
        if not norm_path.exists():
            raise FileNotFoundError(f"config.json not found at {norm_path}")
        with open(norm_path, "r") as f:
            norm_cfg = json.load(f)
        self.mean_temp = float(norm_cfg["normalization"]["temperature"]["mean"])
        self.std_temp = float(norm_cfg["normalization"]["temperature"]["std"])
        self.mean_elev = float(norm_cfg["normalization"]["elevation"]["mean"])
        self.std_elev = float(norm_cfg["normalization"]["elevation"]["std"])
        self.epoch_year = int(norm_cfg["epoch_start_year"])
        self.lat_range = tuple(norm_cfg["bbox"]["lat"])
        self.long_range = tuple(norm_cfg["bbox"]["lon"])
        self.min_lat, self.max_lat = self.lat_range
        self.min_long, self.max_long = self.long_range

        # Loads stations
        reg_path = self.data_root / "station_registry.parquet"
        if not reg_path.exists():
            raise FileNotFoundError(f"Missing station_registry.parquet at {reg_path}")
        reg_table = pq.read_table(
            reg_path, 
            columns=["station_id", "mapped_id", "latitude", "longitude", "elevation"]
        )
        self.registry = reg_table.to_pandas()
        self.registry.sort_values("mapped_id", inplace=True)
        mapped = self.registry["mapped_id"].to_numpy()
        assert mapped[0] == 0
        assert np.all(mapped == np.arange(len(mapped))), "Registry mapped_id must be 0..N-1 with no gaps"
        self.max_station_id = int(self.registry["mapped_id"].max())
        coords = np.stack([
            self.registry["latitude"].values,
            self.registry["longitude"].values,
            self.registry["elevation"].values
        ], axis=1)
        self.station_coords = torch.from_numpy(coords).float()

        # Load packed observations
        pt_path = self.data_root / "packed_observations.pt"
        if not pt_path.exists():
            raise FileNotFoundError(f"Missing packed_observations.pt at {pt_path}")
        packed_data = torch.load(pt_path, map_location="cpu")
        self.pointers = packed_data["pointers"].long()
        self.unique_times = packed_data["unique_times"].long()
        self.values = packed_data["values"].float()
        self.station_ids = packed_data["station_ids"].int()
        self.obs_counts = packed_data["obs_counts"].long()
        self.split_mask = packed_data[f"{self.split}_mask"].bool()

        # Sanity check
        T = self.unique_times.numel()
        assert self.pointers.numel() == T + 1, "pointers must be length len(unique_times)+1"
        assert self.obs_counts.numel() == T, "obs_counts must be length len(unique_times)"
        assert int(self.pointers[-1].item()) == self.values.numel() == self.station_ids.numel(), "pointers[-1] must equal number of packed observations"
        assert self.split_mask.numel() == T, "split_mask must align with unique_times"
        
        # Computes time sampling by normalising counts over windows and restricting to correct time split (e.g. time test val)
        # For simplicity we sample t_start -> the first time step (and the window is thus (t_start, t_start + 1, ..., t_start + H - 1))
        # Gets min [inclusive] and max hour [exclusive] 
        if split == "train": min_hour, max_hour = int(norm_cfg["splits_hour"]["train"][0]), int(norm_cfg["splits_hour"]["train"][1]) 
        elif split == "val": min_hour, max_hour = int(norm_cfg["splits_hour"]["val"][0]), int(norm_cfg["splits_hour"]["val"][1]) 
        elif split == "test": min_hour, max_hour = int(norm_cfg["splits_hour"]["test"][0]), int(norm_cfg["splits_hour"]["test"][1]) 
        else: raise RuntimeError("Invalid split somehow") 
        max_time_start_inclusive = max_hour - ((self.h_window - 1) * self.delta_hours) - 1 # Last valid time to sample for the split
        # A time is valid if all times in window exist in unique_times and they all fall inside the split_mask
        # Ensures within valid bounds
        cand_indices = torch.nonzero(self.split_mask, as_tuple=False).squeeze(1)
        cand_times = self.unique_times[cand_indices]
        valid_t0_mask = (cand_times >= min_hour) & (cand_times <= max_time_start_inclusive)
        cand_indices = cand_indices[valid_t0_mask]
        if cand_indices.numel() == 0:
             raise RuntimeError(f"No valid start times found in bounds for split={self.split}")
        # Vectorised checks to see that the window exists - code is quite ugly but just ensures that it falls within time bounds and that times exist for all points in window
        window_offsets = torch.arange(self.h_window, device=self.unique_times.device) * self.delta_hours
        target_times = self.unique_times[cand_indices].unsqueeze(1) + window_offsets.unsqueeze(0)
        flat_targets = target_times.view(-1)
        idx_flat = torch.searchsorted(self.unique_times, flat_targets)
        idx_flat_clamped = idx_flat.clamp(max=len(self.unique_times) - 1)
        found_vals = self.unique_times[idx_flat_clamped]
        exists_mask_flat = (found_vals == flat_targets)
        in_split_flat = self.split_mask[idx_flat_clamped]
        valid_step_mask = (exists_mask_flat & in_split_flat).view(len(cand_indices), self.h_window)
        window_valid_mask = valid_step_mask.all(dim=1)
        valid_indices = cand_indices[window_valid_mask]
        valid_step_indices = idx_flat_clamped.view(len(cand_indices), self.h_window)[window_valid_mask]
        counts_per_step = self.obs_counts[valid_step_indices]
        past_counts_all = counts_per_step[:, :-1].sum(dim=1) # At 1, ..., H-1 steps of window
        future_counts_all = counts_per_step[:, -1] # At last step of window
        window_weights = counts_per_step.sum(dim=1).float()
        final_mask = window_weights > 0
        self.valid_start_idx = valid_indices[final_mask]
        self.window_weights = window_weights[final_mask]
        self.past_counts = past_counts_all[final_mask]
        self.future_counts = future_counts_all[final_mask]
        if len(self.valid_start_idx) == 0:
            raise RuntimeError(f"No valid window starts for split={self.split} (H={self.h_window}, delta={self.delta_hours}).")
        print(f"[{self.split.upper()}] Initialized {len(self.window_weights)} valid windows.")
        # End of time weighting vectorised


    
    def _normalise_data(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.clone().detach()
        y = y.clone().detach()
        x[:, 0] = (2.0 * (x[:, 0] - self.min_lat) / (self.max_lat - self.min_lat)) - 1.0
        x[:, 1] = (2.0 * (x[:, 1] - self.min_long) / (self.max_long - self.min_long)) - 1.0
        x[:, 2] = (x[:, 2] - self.mean_elev) / self.std_elev
        x[:, 3] = normalise_time(x[:, 3].long())
        y = (y - self.mean_temp) / self.std_temp
        return x, y


    
    # Builds an episode
    def _sample_episode(self, N_c: int, N_t: int, eligible_indices, probs):
        # Sample time
        idx_in_eligible = np.random.choice(len(eligible_indices), p=probs)
        t_start_idx = eligible_indices[idx_in_eligible].item()
        t_0 = self.unique_times[t_start_idx].item()

        # Loads all the data within the window
        all_x, all_y = [], []
        window_times = [t_0 + i * self.delta_hours for i in range(self.h_window)]
        for t in window_times:
            t_idx = torch.searchsorted(self.unique_times, torch.tensor(t)).item() # can make faster using vectorised structs from init
            start_ptr = self.pointers[t_idx]
            end_ptr = self.pointers[t_idx + 1]
            sids = self.station_ids[start_ptr:end_ptr].long()
            vals = self.values[start_ptr:end_ptr].unsqueeze(-1) # Temps (unsqueeze to add dy=1 dim)
            coords = self.station_coords[sids] # Lat, lon, elev
            t_vec = torch.full((sids.shape[0], 1), float(t))
            x_raw = torch.cat([coords, t_vec], dim=1)
            all_x.append(x_raw)
            all_y.append(vals)

        if self.ordering == "forecasting": # Forecasting takes the first H-1 windows and gets N_c points (random spatially, but temporally ordered) and then N_t points from last window H
            x_past = torch.cat(all_x[:-1], dim=0)
            y_past = torch.cat(all_y[:-1], dim=0)
            x_future = all_x[-1]
            y_future = all_y[-1]
            c_perm = torch.randperm(x_past.shape[0])[:N_c]
            t_perm = torch.randperm(x_future.shape[0])[:N_t]
            xc = x_past[c_perm]
            yc = y_past[c_perm]
            xt = x_future[t_perm]
            yt = y_future[t_perm]
            # Sorts context chronologically - targets only from one time step
            ctx_sort = torch.argsort(xc[:, 3])
            xc, yc = xc[ctx_sort], yc[ctx_sort]
        else: # Otherwise is a pure spatiotemporal infilling task
            x_full = torch.cat(all_x, dim=0)
            y_full = torch.cat(all_y, dim=0)
            # Selects subset of data randomly
            perm = torch.randperm(x_full.shape[0])
            ctx_idx = perm[:N_c]
            tgt_idx = perm[N_c : N_c + N_t]
            sequence_idx = perm[:N_c + N_t]
            # Orders data if appropriate
            if self.ordering == "random":
                pass # Nothing to do its already randomly ordered
            elif self.ordering == "ctx_time": # Orders context set on time
                ctx_times = x_full[ctx_idx, 3]
                ctx_sort_map = torch.argsort(ctx_times)
                ctx_idx = ctx_idx[ctx_sort_map]
            elif self.ordering == "full_time": # Orders all points on time (so it becomes a nowcasting + forecasting task)
                total_times = x_full[sequence_idx, 3]
                full_sort_map = torch.argsort(total_times)
                sorted_subset = sequence_idx[full_sort_map]
                ctx_idx = sorted_subset[:N_c]
                tgt_idx = sorted_subset[N_c : N_c + N_t]
            else:
                assert False, "Invalid ordering somehow"
            # Constructs ctx and tgt
            xc = x_full[ctx_idx]
            yc = y_full[ctx_idx]
            xt = x_full[tgt_idx]
            yt = y_full[tgt_idx]
        return xc, yc, xt, yt, t_0


    def generate_batch(self) -> TemporalHadISDBatch:
        i = 0
        while i < self.max_resamples:
            # Samples N_t and N_c
            N_t = random.randint(self.N_t_min, self.N_t_max)
            N_c = random.randint(self.N_c_min, self.N_c_max)
            # Recomputes observation weighting by removing windows with too few observations
            if self.ordering == "forecasting":
                valid_mask = (self.past_counts >= N_c) & (self.future_counts >= N_t) # Forecasting must have at least N_t at final step and N_c at all preceding steps
            else:
                valid_mask = self.window_weights >= N_c + N_t
            if not torch.any(valid_mask):
                i += 1
                print(f"ERROR with N_c={N_c} and N_t={N_t}")
                continue
            eligible_indices = self.valid_start_idx[valid_mask]
            eligible_weights = self.window_weights[valid_mask]
            probs = (eligible_weights / eligible_weights.sum()).numpy() # Renorm
            # Generates episodes in batch
            xs_all, ys_all = [], []
            xcs_all, ycs_all = [], []
            xts_all, yts_all = [], []
            raw_times = []
            for _ in range(self.batch_size):
                x_ctx_raw, y_ctx_raw, x_tgt_raw, y_tgt_raw, t_star = self._sample_episode(N_c=N_c, N_t=N_t,eligible_indices=eligible_indices, probs=probs)
                # Builds batch
                x_raw = torch.cat([x_ctx_raw, x_tgt_raw], axis=0)
                y_raw = torch.cat([y_ctx_raw, y_tgt_raw], axis=0)

                x_norm, y_norm = self._normalise_data(x_raw, y_raw)

                xc_i = x_norm[:N_c, :]
                yc_i = y_norm[:N_c, :]
                xt_i = x_norm[N_c:, :]
                yt_i = y_norm[N_c:, :]

                xs_all.append(x_norm.detach().to(dtype=torch.float32))
                ys_all.append(y_norm.detach().to(dtype=torch.float32))
                xcs_all.append(xc_i.detach().to(dtype=torch.float32))
                ycs_all.append(yc_i.detach().to(dtype=torch.float32))
                xts_all.append(xt_i.detach().to(dtype=torch.float32))
                yts_all.append(yt_i.detach().to(dtype=torch.float32))
                raw_times.append(torch.tensor(t_star, dtype=torch.long))
            return TemporalHadISDBatch(
                x=torch.stack(xs_all, dim=0),
                y=torch.stack(ys_all, dim=0),
                xc=torch.stack(xcs_all, dim=0),
                yc=torch.stack(ycs_all, dim=0),
                xt=torch.stack(xts_all, dim=0),
                yt=torch.stack(yts_all, dim=0),
                mean_temp=self.mean_temp,
                std_temp=self.std_temp,
                mean_elev=self.mean_elev,
                std_elev=self.std_elev,
                lat_range=self.lat_range,
                long_range=self.long_range,
                unnormalised_time=torch.stack(raw_times, dim=0),
            )
        raise RuntimeError("Sampling exceed max retries")
