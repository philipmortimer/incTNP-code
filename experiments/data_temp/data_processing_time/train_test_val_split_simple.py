import os
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.compute as pc
import torch
from tqdm import tqdm

# --- GLOBAL CONSTANTS ---
START_TIME = 1931
EPOCH0 = np.datetime64(f"{START_TIME}-01-01T00:00:00")

def year_to_hour(year):
    """Converts a year to hours since EPOCH0 (1931)."""
    if year is None: return None
    dt = np.datetime64(f"{year}-01-01T00:00:00")
    return int((dt - EPOCH0) / np.timedelta64(1, "h"))

def run_preprocessor(npz_root, parquet_root, lat_bounds, lon_bounds, splits, output_path):
    npz_root, output_dir = Path(npz_root), Path(output_path)
    station_data_dir = npz_root / "station_data"

    if output_dir.exists():
        print(f"Cleaning existing directory: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    # 1. Geographic Filter
    print("Filtering stations by geography...")
    meta_ds = ds.dataset(Path(parquet_root) / "station_meta.parquet")
    
    selection = (pc.field("latitude") >= lat_bounds[0]) & \
                (pc.field("latitude") <= lat_bounds[1]) & \
                (pc.field("longitude") >= lon_bounds[0]) & \
                (pc.field("longitude") <= lon_bounds[1])
    
    meta_df = meta_ds.scanner(filter=selection).to_table().to_pandas()
    print(f"Region Filter: Found {len(meta_df)} candidates in metadata.")

    # 2. Setup Time Bounds
    split_bounds = {k: (year_to_hour(v[0]), year_to_hour(v[1]+1)) for k, v in splits.items()}
    train_start, train_end = split_bounds["train"]

    # 3. Load Data & Assign Dense IDs
    print("Loading valid station data...")
    
    all_times = []
    all_sids = []
    all_vals = []
    
    # Storage for Stats Calculation
    train_vals_for_stats = []     # Actual temperature readings in train split
    stations_seen_in_train = []   # List of mapped_ids that appeared in train split
    
    valid_registry_rows = []
    current_dense_id = 0

    # Iterate using itertuples for speed
    for row in tqdm(meta_df.itertuples(), total=len(meta_df)):
        sid = str(row.station_id)
        npz_path = station_data_dir / f"{sid}.npz"
        
        if not npz_path.exists():
            continue

        try:
            with np.load(npz_path) as data:
                t = data["time_hours"].astype(np.int32)
                v = data["temperature"].astype(np.float32)

                # Filter NaNs immediately
                valid_mask = ~np.isnan(v)
                t = t[valid_mask]
                v = v[valid_mask]

                if len(t) == 0: 
                    continue

                # --- 3a. Pack Global Data ---
                all_times.append(t)
                all_vals.append(v)
                all_sids.append(np.full(t.shape, current_dense_id, dtype=np.int32))

                # --- 3b. Accumulate Train Stats ---
                # Check which observations fall into the training window
                station_train_mask = (t >= train_start) & (t < train_end)
                
                if np.any(station_train_mask):
                    # Append valid training temperatures for Mean/Std calc
                    train_vals_for_stats.append(v[station_train_mask])
                    # Mark this station as "Seen in Train" for spatial stats
                    stations_seen_in_train.append(current_dense_id)
                
                # --- 3c. Build Registry ---
                valid_registry_rows.append({
                    "station_id": sid,
                    "mapped_id": current_dense_id,
                    "latitude": row.latitude,
                    "longitude": row.longitude,
                    "elevation": row.elevation
                })
                
                current_dense_id += 1
                
        except Exception as e:
            print(f"Error loading {sid}: {e}")
            continue

    # Safety Guard: Did we load anything?
    if not all_times:
        raise RuntimeError("No valid observations found! Check paths and bounds.")

    print(f"Final Count: {current_dense_id} stations contributed data.")

    # 4. Calculate Normalization Statistics (Strictly on Train Split)
    print("Calculating statistics...")

    # A. Temperature Statistics
    if train_vals_for_stats:
        concat_train = np.concatenate(train_vals_for_stats)
        temp_mean = float(np.mean(concat_train))
        temp_std = float(np.std(concat_train))
        temp_std = max(temp_std, 1e-6) # Prevent div-by-zero
    else:
        print("WARNING: No training data found. Defaulting Temp stats.")
        temp_mean, temp_std = 0.0, 1.0

    # B. Elevation Statistics (Using only stations seen in Train)
    final_reg_df = pd.DataFrame(valid_registry_rows)
    
    # Filter registry to stations active during training
    train_subset_df = final_reg_df[final_reg_df['mapped_id'].isin(stations_seen_in_train)]
    
    if not train_subset_df.empty:
        # Use ddof=0 so N=1 gives std=0.0 instead of NaN
        elev_values = train_subset_df['elevation'].dropna()
        if len(elev_values) > 0:
            elev_mean = float(elev_values.mean())
            elev_std = float(elev_values.std(ddof=0))
        else:
             # Fallback if stations exist but all elevations are NaN
             elev_mean, elev_std = 0.0, 1.0
    else:
        print("WARNING: No stations found in training split! Defaulting Elevation stats.")
        # Fallback if no stations in train split
        elev_mean, elev_std = 0.0, 1.0

    # Guard: If std is 0 (N=1 or all stations at same height) or infinite, force to 1.0
    if not np.isfinite(elev_std) or elev_std < 1e-6:
        print(f"  Note: Elevation std was {elev_std}, clamping to 1.0 to prevent div-by-zero.")
        elev_std = 1.0
    
    print(f"  Temp -> Mean: {temp_mean:.2f}, Std: {temp_std:.2f}")
    print(f"  Elev -> Mean: {elev_mean:.2f}, Std: {elev_std:.2f} (Computed on {len(train_subset_df)} stations)")

    # 5. Build Time-First CSPR
    print("Sorting and building index...")
    
    global_t = np.concatenate(all_times)
    global_s = np.concatenate(all_sids)
    global_v = np.concatenate(all_vals)

    # Sort stable by time
    sort_idx = np.argsort(global_t, kind='stable')
    sorted_t = global_t[sort_idx]
    sorted_s = global_s[sort_idx]
    sorted_v = global_v[sort_idx]

    unique_times, start_indices = np.unique(sorted_t, return_index=True)
    
    pointers = np.zeros(len(unique_times) + 1, dtype=np.int64)
    pointers[:-1] = start_indices
    pointers[-1] = len(sorted_t)

    # 6. Create Split Masks
    split_masks = {}
    for split_name, (start_h, end_h) in split_bounds.items():
        mask = (unique_times >= start_h) & (unique_times < end_h)
        split_masks[f"{split_name}_mask"] = torch.from_numpy(mask)

    obs_counts = torch.from_numpy(np.diff(pointers))

    # 7. Save Everything
    print(f"Saving to {output_dir}...")

    # Save Tensor Data
    torch.save({
        "unique_times": torch.from_numpy(unique_times),
        "pointers": torch.from_numpy(pointers),
        "station_ids": torch.from_numpy(sorted_s),
        "values": torch.from_numpy(sorted_v),
        "obs_counts": obs_counts, 
        **split_masks
    }, output_dir / "packed_observations.pt")

    # Save Registry
    final_reg_df.to_parquet(output_dir / "station_registry.parquet")

    # Save Config
    config = {
        "splits_year": splits,
        "splits_hour": split_bounds, 
        "epoch_start_year": START_TIME,
        "normalization": {
            "temperature": {
                "mean": temp_mean, 
                "std": temp_std
            },
            "elevation": {
                "mean": elev_mean,
                "std": elev_std
            }
        },
        "bbox": {"lat": lat_bounds, "lon": lon_bounds},
        "n_stations": current_dense_id
    }
    
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print("Done.")

if __name__ == "__main__":
    run_preprocessor(
        npz_root="REMOVED",
        parquet_root="REMOVED",
        lat_bounds=(-20.0, 60.0), 
        lon_bounds=(-10.0, 52.0),
        splits={"train": (1980, 2016), "val": (2017, 2017), "test": (2018, 2019)},
        output_path="REMOVED"
    )