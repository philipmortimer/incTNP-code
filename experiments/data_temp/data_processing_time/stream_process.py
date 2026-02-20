# 4. Packs observations densely for data loader
import numpy as np
import pyarrow.parquet as pq
import torch
import time
import json
from pathlib import Path
from tqdm import tqdm

def pack_observations_csr(
    npz_root_str: str, 
    registry_path_str: str, 
    output_path_str: str
):
    # Time major CSR
    start_t = time.time()
    npz_root = Path(npz_root_str)
    station_data_dir = npz_root / "station_data"
    output_path = Path(output_path_str)
    
    # Loads stations
    print(f"Loading Registry from {registry_path_str}...")
    if not Path(registry_path_str).exists():
        raise FileNotFoundError(f"Registry not found at {registry_path_str}. Run Script 3 first.")
        
    registry = pq.read_table(registry_path_str).to_pandas()
    
    # Map String ID -> Mapped Integer (0..N)
    sid_to_int = dict(zip(registry["station_id"], registry["mapped_id"]))
    valid_sids = set(registry["station_id"])
    
    # Determine smallest safe integer type for Station IDs
    max_id = registry["mapped_id"].max()
    if max_id < 32767:
        id_dtype = np.int16
        print(f"Optimization: Using int16 for Station IDs (Max ID: {max_id})")
    else:
        id_dtype = np.int32
        print(f"Using int32 for Station IDs (Max ID: {max_id})")

    # -------------------------------------------------------------
    # 2. Scan & Accumulate Data (Station-Major)
    # -------------------------------------------------------------
    all_times = []
    all_sids = []
    all_vals = []
    
    print("Scanning station files...")
    # Iterate only stations present in the Registry
    for sid in tqdm(registry["station_id"].values):
        npz_path = station_data_dir / f"{sid}.npz"
        if not npz_path.exists(): 
            continue
        
        mapped_id = sid_to_int[sid]
        
        with np.load(npz_path) as data:
            # Load raw data
            # Assumes 'time_hours' is int32 (hours since epoch)
            # Assumes 'temperature' is float
            t_hours = data["time_hours"].astype(np.int32) 
            temps = data["temperature"].astype(np.float32)
            
            # correctness check: Filter NaNs
            valid_mask = ~np.isnan(temps)
            
            if np.any(valid_mask):
                t_clean = t_hours[valid_mask]
                v_clean = temps[valid_mask]
                
                # Create ID array for this block
                s_clean = np.full(t_clean.shape, mapped_id, dtype=id_dtype)
                
                all_times.append(t_clean)
                all_vals.append(v_clean)
                all_sids.append(s_clean)

    print("Concatenating into global arrays...")
    if not all_times:
        raise ValueError("No valid data found in any station files!")

    # Combine into massive 1D arrays
    global_time = np.concatenate(all_times)
    global_sid  = np.concatenate(all_sids)
    global_val  = np.concatenate(all_vals)
    
    num_obs = len(global_time)
    print(f"Total Observations: {num_obs:,}")
    
    # Free memory
    del all_times, all_sids, all_vals

    # Sort by time
    print("Sorting by Time (This may take a moment)...")
    sort_idx = np.argsort(global_time, kind='stable') 
    
    # Apply sort
    sorted_time = global_time[sort_idx]
    sorted_sid  = global_sid[sort_idx]
    sorted_val  = global_val[sort_idx]
    
    del global_time, global_sid, global_val, sort_idx

    # Builds CSR
    print("Computing CSR Pointers...")
    
    # np.unique gives us the sorted unique timestamps and the *start index* of each block
    unique_times, start_indices = np.unique(sorted_time, return_index=True)
    
    # Build pointers: [start_0, start_1, ..., start_N, total_obs]
    # The block for unique_times[i] is at pointers[i] : pointers[i+1]
    pointers = np.zeros(len(unique_times) + 1, dtype=np.int32) # int32 is fine for < 2B indices
    pointers[:-1] = start_indices
    pointers[-1] = num_obs
    
    # Build the Hash Map (Time -> Pointer Index)
    # This enables O(1) random access: lookup(t) -> idx -> pointers[idx]
    # We use a standard Python Dict. It is highly optimized and handles sparse keys perfectly.
    print("Building Time Index Map...")
    time_to_pointer_idx = {int(t): int(i) for i, t in enumerate(unique_times)}

    # Computes weighjting
    print("Computing Sampling Weights...")
    # Weight = Count / Total. 
    # Used to sample t* heavily from times with many active stations.
    counts = np.diff(pointers) # Count of obs per timestamp
    weights = counts.astype(np.float32) / counts.sum()

    print(f"Saving to {output_path}...")
    
    data_dict = {
        # The Payload (Sorted by time)
        "payload_station_ids": torch.from_numpy(sorted_sid),   # int16 or int32
        "payload_temperatures": torch.from_numpy(sorted_val),  # float32
        
        # The Index Structure
        "pointers": torch.from_numpy(pointers),                # int32
        "unique_times": torch.from_numpy(unique_times),        # int32
        "time_to_pointer_idx": time_to_pointer_idx,            # Dict[int, int]
        
        # The Sampling Distribution
        "sampling_weights": torch.from_numpy(weights)          # float32
    }
    
    # Create parent dir if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(data_dict, output_path)
    
    file_size_mb = output_path.stat().st_size / 1e6
    print(f"Done.")
    print(f"  Output: {output_path}")
    print(f"  Size:   {file_size_mb:.2f} MB")
    print(f"  Time:   {time.time() - start_t:.2f} s")
    print("-" * 60)
    print(f"  Dtype Check:")
    print(f"    IDs:   {data_dict['payload_station_ids'].dtype}")
    print(f"    Temps: {data_dict['payload_temperatures'].dtype}")


if __name__ == "__main__":
    # Update these paths to match your environment
    NPZ_ROOT = "REMOVED"
    
    # Input
    REGISTRY_PATH = f"{NPZ_ROOT}/station_registry.parquet"
    
    # Output
    OUTPUT_PATH = f"{NPZ_ROOT}/packed_observations.pt"
    
    pack_observations_csr(NPZ_ROOT, REGISTRY_PATH, OUTPUT_PATH)