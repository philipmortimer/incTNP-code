# 3. Builds train test and val sets with temporal splits and geographic splits
import os, math, json, time
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import pyarrow.compute as pc
import reverse_geocoder as rg
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

# --- GLOBAL CONSTANTS ---
START_TIME = 1931
EPOCH0 = np.datetime64(f"{START_TIME}-01-01T00:00:00")

def write_summary(path: Path, d: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for k, v in d.items():
            f.write(f"{k}: {v}\n")

def year_bounds_to_hour_range(y_min, y_max):
    if y_min is not None:
        start_dt = np.datetime64(f"{y_min}-01-01T00:00:00")
        start_h = int(((start_dt - EPOCH0) / np.timedelta64(1, "h")))
    else:
        start_h = None
    if y_max is not None:
        end_dt = np.datetime64(f"{y_max+1}-01-01T00:00:00")
        end_h = int(((end_dt - EPOCH0) / np.timedelta64(1, "h")))
    else:
        end_h = None
    return start_h, end_h

# Maps lat long to 3d sphere for better distance clustering
def latlon_to_3d(lat_deg, lon_deg):
    lats = np.radians(lat_deg)
    lons = np.radians(lon_deg)
    x = np.cos(lats) * np.cos(lons)
    y = np.cos(lats) * np.sin(lons)
    z = np.sin(lats)
    return np.stack([x, y, z], axis=1)

def plot_split_map(meta_pdf, registry_df, output_path, bounds):
    print("Generating split visualization map...")
    
    # Merge registry info (spatial_set_id) into meta for plotting
    df = meta_pdf.loc[registry_df['station_id']].copy()
    df['set_id'] = registry_df.set_index('station_id')['spatial_set_id']

    fig = plt.figure(figsize=(15, 12), dpi=150)
    proj = ccrs.Mercator()
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    
    lon_min, lon_max = bounds[0]
    lat_min, lat_max = bounds[1]
    extent = [lon_min-1, lon_max+1, lat_min-1, lat_max+1]
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.LAND, facecolor='#f0f0f0')
    ax.add_feature(cfeature.OCEAN, facecolor='#e0f7fa')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)

    # 0=Train, 1=Val, 2=Test
    cmap = ListedColormap(['#bbbbbb', '#1f77b4', '#d62728'])
    
    scatter = ax.scatter(
        df['longitude'], df['latitude'], 
        c=df['set_id'], cmap=cmap, 
        s=15, edgecolor='black', linewidth=0.3, alpha=0.9,
        transform=ccrs.PlateCarree(),
        vmin=0, vmax=2 # Ensure mapping is consistent even if set is missing
    )

    counts = df['set_id'].value_counts()
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#bbbbbb', label=f'Train/Context ({counts.get(0,0)})', markersize=10),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4', label=f'Val Holdout ({counts.get(1,0)})', markersize=10),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#d62728', label=f'Test Holdout ({counts.get(2,0)})', markersize=10)
    ]
    ax.legend(handles=legend_elements, loc='upper right', facecolor='white', framealpha=1)
    ax.set_title("Station Spatial Split Assignment", fontsize=16)
    
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Map saved to {output_path}")
    plt.close()

# K means splitting
def split_stations_kmeans(meta_pdf, station_ids, n_clusters=50, seed=1234, train_frac=0.80, val_frac=0.10):
    print(f"Running K-Means split with K={n_clusters}...")
    test_frac = 1.0 - train_frac - val_frac
    subset = meta_pdf.loc[station_ids]
    lats = subset["latitude"].values
    lons = subset["longitude"].values
    
    X_3d = latlon_to_3d(lats, lons).astype(np.float64)
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = kmeans.fit_predict(X_3d)
    
    unique_clusters = np.arange(n_clusters)
    rng = np.random.RandomState(seed)
    rng.shuffle(unique_clusters)
    
    n_val_blocks = max(1, int(val_frac * n_clusters))
    n_test_blocks = max(1, int(test_frac * n_clusters))
    
    val_clusters = set(unique_clusters[:n_val_blocks])
    test_clusters = set(unique_clusters[n_val_blocks : n_val_blocks + n_test_blocks])
    
    train_ids, val_ids, test_ids = [], [], []
    for i, sid in enumerate(station_ids):
        cid = labels[i]
        if cid in val_clusters: val_ids.append(sid)
        elif cid in test_clusters: test_ids.append(sid)
        else: train_ids.append(sid)
            
    return sorted(train_ids), sorted(val_ids), sorted(test_ids)

# Allows all stations to be in all splits (Pure Stream / Episode Disjoint)
def split_stations_all(station_ids):
    sids = sorted(list(station_ids))
    # NOTE: Returns empty lists for val/test HOLDOUTS. 
    # Because in this mode, Val and Test use the 'Train' stations (spatial_set=0)
    # just at different times.
    return sids, [], []

# Split randomly
def split_stations_random(station_ids, train_frac=0.7, val_frac=0.15, seed=1234):
    rng = np.random.RandomState(seed)
    station_ids = np.array(list(station_ids), dtype=object)
    rng.shuffle(station_ids)

    n_total = len(station_ids)
    if n_total < 3: raise SystemExit(f"Need at least 3 stations, got {n_total}")

    n_train = int(round(train_frac * n_total))
    n_val = int(round(val_frac * n_total))
    n_train = max(1, min(n_train, n_total - 2))
    n_val = max(1, min(n_val, n_total - n_train - 1))

    train_stations = sorted(station_ids[:n_train].tolist())
    val_stations = sorted(station_ids[n_train : n_train + n_val].tolist())
    test_stations = sorted(station_ids[n_train + n_val :].tolist())
    return train_stations, val_stations, test_stations

# Split by country
def _load_or_build_country_codes(meta_pdf, station_ids: List[str], cache_path: Path) -> Dict[str, str]:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists(): 
        tbl = pq.read_table(cache_path)
        df = tbl.to_pandas()
        return dict(zip(df["station_id"].astype(str), df["country_code"].astype(str).str.upper()))

    sub = meta_pdf.loc[station_ids, ["latitude", "longitude"]].copy()
    coords = list(zip(sub["latitude"].to_numpy().tolist(), sub["longitude"].to_numpy().tolist()))
    print(f"Reverse geocoding {len(coords)} stations...")
    results = rg.search(coords)
    codes = [r["cc"].upper() for r in results]

    out_tbl = pa.table({"station_id": list(sub.index.astype(str)), "country_code": codes})
    pq.write_table(out_tbl, cache_path, compression="snappy")
    return dict(zip(list(sub.index.astype(str)), codes))


def build_splits_and_sets(
    NPZ_ROOT: str, PARQUET_ROOT: str,
    LAT_BOUNDS: Tuple[float, float], LON_BOUNDS: Tuple[float, float],
    SPLITS: dict, *,
    split_mode: str = "random",
    val_countries: Optional[List[str]] = None,
    test_countries: Optional[List[str]] = None,
    n_clusters: int = 50,
    train_frac: float = 0.7, val_frac: float = 0.15,
    seed: int = 1234,
):
    start_t = time.time()
    npz_root = Path(NPZ_ROOT)
    parquet_root = Path(PARQUET_ROOT)
    station_data_dir = npz_root / "station_data"
    
    # 1. Filter stations by Geobounds
    meta_path = parquet_root / "station_meta.parquet"
    if not meta_path.exists(): raise SystemExit(f"Missing {meta_path}")

    meta_ds = ds.dataset(meta_path)
    lat_lo, lat_hi = LAT_BOUNDS
    lon_lo, lon_hi = LON_BOUNDS
    wanted_ids = (meta_ds.scanner(
            filter=((pc.field("latitude") >= lat_lo) & (pc.field("latitude") <= lat_hi) & 
                    (pc.field("longitude") >= lon_lo) & (pc.field("longitude") <= lon_hi)),
            columns=["station_id"]).to_table()["station_id"].to_pylist())
    wanted_ids = [str(s) for s in wanted_ids]
    print(f"Found {len(wanted_ids)} stations in lat/lon region")

    meta_tbl = pq.read_table(meta_path, columns=["station_id", "latitude", "longitude", "elevation"])
    meta_pdf = meta_tbl.to_pandas()
    meta_pdf["station_id"] = meta_pdf["station_id"].astype(str)
    meta_pdf = meta_pdf.set_index("station_id")

    split_hour_ranges = {k: year_bounds_to_hour_range(*v) for k, v in SPLITS.items()}

    # 2. Scan availability and map valid stations
    valid_station_ids = []
    # These lists store the *Temporal* availability index for every station
    split_indices = {k: [] for k in SPLITS.keys()}
    
    print("Scanning station data availability...")
    for i, sid_str in enumerate(wanted_ids):
        npz_path = station_data_dir / f"{sid_str}.npz"
        if not npz_path.exists(): continue

        # Lightweight header read if possible, but loading npz is fast enough usually
        with np.load(npz_path) as data:
            time_hours = data["time_hours"]
            n_points = int(time_hours.size)

        if n_points == 0 or sid_str not in meta_pdf.index: continue
        
        # Calculate start/end index for each temporal split (train/val/test)
        # This allows the loader to know WHICH PART of the array corresponds to "Train Time"
        has_any_data = False
        station_indices = {}
        
        for split_name, (start_h, end_h) in split_hour_ranges.items():
            mask = np.ones(n_points, dtype=bool)
            if start_h is not None: mask &= time_hours >= start_h
            if end_h is not None: mask &= time_hours < end_h
            
            idx = np.nonzero(mask)[0]
            if idx.size > 0:
                station_indices[split_name] = (int(idx[0]), int(idx[-1]) + 1)
                has_any_data = True
            else:
                station_indices[split_name] = (-1, -1)
        
        if has_any_data:
            valid_station_ids.append(sid_str)
            for k in SPLITS.keys():
                split_indices[k].append(station_indices[k])
        
        if i % 1000 == 0: print(f"  {i}/{len(wanted_ids)} scanned")

    if not valid_station_ids: raise SystemExit("No valid data found.")

    # 3. Determine Spatial Splits (The Sets)
    # Filter to only stations that have data in the TRAIN temporal window
    # (We assume context comes from 'Train Time', even if held out spatially)
    train_indices = split_indices["train"]
    stations_with_train_time = [
        sid for i, sid in enumerate(valid_station_ids) 
        if train_indices[i][0] != -1
    ]
    
    if split_mode == "random":
        s_train, s_val, s_test = split_stations_random(stations_with_train_time, train_frac, val_frac, seed)
    elif split_mode == "kmeans":
        s_train, s_val, s_test = split_stations_kmeans(meta_pdf, stations_with_train_time, n_clusters, seed, train_frac, val_frac)
    elif split_mode == "country":
        if not val_countries or not test_countries: raise ValueError("Countries required")
        cc_map = _load_or_build_country_codes(meta_pdf, valid_station_ids, npz_root / "station_country_codes.parquet")
        val_codes, test_codes = set(val_countries), set(test_countries)
        s_val = [s for s in stations_with_train_time if cc_map.get(s) in val_codes]
        s_test = [s for s in stations_with_train_time if cc_map.get(s) in test_codes]
        heldout = set(s_val) | set(s_test)
        s_train = [s for s in stations_with_train_time if s not in heldout]
    elif split_mode == "all":
        # Pure Stream Mode: Everyone is "Train" spatially.
        # Val and Test sets are empty lists of *held-out IDs*.
        s_train, s_val, s_test = split_stations_all(stations_with_train_time)
    else:
        raise ValueError(f"Unknown mode {split_mode}")

    # 4. Create the Master Registry
    # We assign integer IDs 0..N-1 to ALL valid stations
    # We also assign a 'spatial_set_id': 0=Train, 1=Val, 2=Test
    
    # Map spatial sets
    spatial_set_map = {sid: 0 for sid in s_train} # Default 0 (Context/Train)
    for sid in s_val: spatial_set_map[sid] = 1    # 1 (Val Holdout)
    for sid in s_test: spatial_set_map[sid] = 2   # 2 (Test Holdout)
    
    # Even stations not in 'stations_with_train_time' (but in valid_station_ids)
    # might exist (e.g. only data in 2018). We default them to -1 (Ignored) or 0? 
    # Safest is to only include stations we actually split.
    final_station_list = sorted(list(set(s_train) | set(s_val) | set(s_test)))
    
    registry_data = {
        "station_id": [],
        "mapped_id": [],
        "spatial_set_id": [],
        "latitude": [],
        "longitude": [],
        "elevation": []
    }
    
    # Add temporal index columns dynamically
    for split_key in SPLITS.keys():
        registry_data[f"{split_key}_start_idx"] = []
        registry_data[f"{split_key}_end_idx"] = []

    # Helper map for temporal indices
    # We need to look up the indices we computed in Step 2.
    # Since valid_station_ids and split_indices lists are aligned:
    sid_to_valid_idx = {sid: i for i, sid in enumerate(valid_station_ids)}

    for mapped_id, sid in enumerate(final_station_list):
        registry_data["station_id"].append(sid)
        registry_data["mapped_id"].append(mapped_id)
        registry_data["spatial_set_id"].append(spatial_set_map[sid])
        
        # Meta
        row = meta_pdf.loc[sid]
        registry_data["latitude"].append(float(row["latitude"]))
        registry_data["longitude"].append(float(row["longitude"]))
        registry_data["elevation"].append(float(row["elevation"]))
        
        # Temporal Indices
        valid_idx = sid_to_valid_idx[sid]
        for split_key in SPLITS.keys():
            start, end = split_indices[split_key][valid_idx]
            registry_data[f"{split_key}_start_idx"].append(start)
            registry_data[f"{split_key}_end_idx"].append(end)

    # Save Registry
    registry_table = pa.Table.from_pydict(registry_data)
    reg_path = npz_root / "station_registry.parquet"
    pq.write_table(registry_table, reg_path, compression="snappy")
    print(f"Wrote Station Registry ({len(final_station_list)} stations) to {reg_path}")

    # 5. Visualizations & JSONs
    registry_df = registry_table.to_pandas()
    plot_split_map(meta_pdf, registry_df, npz_root / "split_visualization.png", (LON_BOUNDS, LAT_BOUNDS))

    # Saves time splits
    station_sets = {
        "split_mode": split_mode,
        "epoch_start_year": START_TIME,
        "time_splits": SPLITS,
        "n_stations": len(final_station_list),
        "spatial_counts": {
            "train": len(s_train),
            "val_heldout": len(s_val),
            "test_heldout": len(s_test)
        }
    }
    with open(npz_root / "station_sets.json", "w") as f:
        json.dump(station_sets, f, indent=2)

    # 6. Normalization
    print("Computing Normalization Stats...")
    sum_temp, sumsq_temp, count = 0.0, 0.0, 0
    elevs_train = []
    
    for sid in s_train: # ONLY iterate Train Spatially
        npz_path = station_data_dir / f"{sid}.npz"
        # Get Train Time indices from registry
        reg_row = registry_df[registry_df["station_id"] == sid].iloc[0]
        t_start = reg_row["train_start_idx"]
        t_end = reg_row["train_end_idx"]
        
        if t_start == -1 or t_end <= t_start: continue
        
        with np.load(npz_path) as data:
            temps = data["temperature"][t_start:t_end].astype(np.float64)
            if temps.size > 0:
                sum_temp += float(temps.sum())
                sumsq_temp += float((temps**2).sum())
                count += int(temps.size)
                elevs_train.append(float(reg_row["elevation"]))

    if count == 0: raise SystemExit("No valid training samples found.")
    
    mean_temp = sum_temp / count
    std_temp = math.sqrt(max(sumsq_temp/count - mean_temp**2, 1e-12))
    
    elevs_arr = np.array(elevs_train)
    mean_elev = float(np.mean(elevs_arr)) if len(elevs_arr) > 0 else 0.0
    std_elev = float(np.std(elevs_arr)) if len(elevs_arr) > 0 else 1.0

    norm_info = {
        "temperature": {"mean": mean_temp, "std": std_temp},
        "elevation": {"mean": mean_elev, "std": std_elev},
        "lat_range": LAT_BOUNDS,
        "lon_range": LON_BOUNDS
    }
    with open(npz_root / "normalization.json", "w") as f:
        json.dump(norm_info, f, indent=2)
    
    print(f"Done. Normalization: Temp Mean={mean_temp:.2f}, Std={std_temp:.2f}")

if __name__ == "__main__":
    NPZ_ROOT = "REMOVED"
    PARQUET_ROOT = "REMOVED"

    LON_BOUNDS = (-10.0, 52.0)
    LAT_BOUNDS = (-20.0, 60.0)

    SPLITS = {
        "train": (START_TIME, 2016),
        "val":   (2017, 2017),
        "test":  (2018, 2019),
    }
    
    # CONFIG 1 & 2: Pure Stream / Episode Disjoint
    # split_mode = "all"
    
    # CONFIG 3: Spatial Holdout
    split_mode = "kmeans" 
    
    build_splits_and_sets(
        NPZ_ROOT, PARQUET_ROOT, LAT_BOUNDS, LON_BOUNDS, SPLITS,
        split_mode=split_mode,
        n_clusters=150,
        seed=1234,
        val_frac=0.10,
        train_frac=0.80,
    )