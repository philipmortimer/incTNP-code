# Main pre processing script to run all steps together
import time
import shutil
import os
from pathlib import Path
from train_test_val_split import build_splits_and_sets
from stream_process import pack_observations_csr

def run_full_pipeline(
    # Data path
    npz_root: str,           # Original Raw Data location (Read-Only)
    experiment_root: str,    # Output location for new artifacts
    parquet_root: str,       # Location of station_meta.parquet

    # Geographic filter
    lat_bounds: tuple,
    lon_bounds: tuple,
    
    # Time split
    start_time: int,
    train_end: int,
    val_year: int,
    test_start: int,
    test_end: int,
    
    # Train test val split strategy
    split_mode: str, 
    val_countries: list=[],
    test_countries: list=[],
    train_frac: float = 0.80,
    val_frac: float = 0.10,
    seed: int = 1234,
    n_clusters: int = 150,
):
    print("==================================================")
    print("STARTING PREPROCESSING PIPELINE")
    print(f"Source Raw Data: {npz_root}")
    print(f"Experiment Output: {experiment_root}")
    print(f"Split Mode: {split_mode}")
    print("==================================================\n")
    
    start_global = time.time()
    
    # Ensure experiment dir exists
    exp_path = Path(experiment_root)
    exp_path.mkdir(parents=True, exist_ok=True)
    
    # --- Step 1: Build Registry & Splits (Script 3) ---
    print(f"\n--- [Step 1/2] Building Registry & Splits ({split_mode}) ---")
    
    # LOGIC: Script 3 expects to find 'station_data' in the root folder provided 
    # and writes artifacts to that same folder. To keep the raw folder clean/safe, 
    # we temporarily symlink 'station_data' into experiment_root, run the script, 
    # then delete the link.
    
    raw_station_data = Path(npz_root) / "station_data"
    temp_symlink = exp_path / "station_data"
    
    # 1. Create Temporary Symlink
    if not temp_symlink.exists():
        os.symlink(raw_station_data, temp_symlink)
    
    try:
        # 2. Run Script 3 (Reads from symlink, writes Parquet/JSON to experiment_root)
        build_splits_and_sets(
            NPZ_ROOT=experiment_root,      
            PARQUET_ROOT=parquet_root,
            LAT_BOUNDS=lat_bounds,
            LON_BOUNDS=lon_bounds,
            SPLITS={
                "train": (start_time, train_end),
                "val":   (val_year, val_year),
                "test":  (test_start, test_end),
            },
            split_mode=split_mode,
            val_countries=val_countries,
            test_countries=test_countries,
            train_frac=train_frac,
            val_frac=val_frac,
            seed=seed,
            n_clusters=n_clusters,
        )
    finally:
        # 3. Cleanup: Remove the symlink immediately
        # The final folder will NOT contain station_data
        if temp_symlink.is_symlink():
            temp_symlink.unlink()
            print("  (Temporary symlink removed)")

    # --- Step 2: Pack Data into CSR Tensor (Script 4) ---
    print(f"\n--- [Step 2/2] Packing Data into Time-Major CSR ---")
    
    # We pass the RAW SOURCE as the input root (so it finds .npz files directly)
    # We pass the EXPERIMENT ROOT as the output path (for the .pt file)
    registry_path = exp_path / "station_registry.parquet"
    output_pt_path = exp_path / "packed_observations.pt"
    
    pack_observations_csr(
        npz_root_str=npz_root,               # Read from RAW source directly
        registry_path_str=str(registry_path),# Read registry from Experiment dir
        output_path_str=str(output_pt_path)  # Write .pt to Experiment dir
    )

    print("\n==================================================")
    print(f"PIPELINE COMPLETE in {time.time() - start_global:.1f} seconds")
    print(f"Artifacts ready in: {experiment_root}")
    print("  - station_registry.parquet")
    print("  - normalization.json")
    print("  - packed_observations.pt")
    print("  (Note: station_data folder is NOT present)")
    print("==================================================")


if __name__ == "__main__":
    # --- CONFIGURATION ---
    
    # MODE SELECTION
    # "all"    -> For Config 1 (Pure Stream) & Config 2 (Disjoint Episode)
    # "kmeans" -> For Config 3 (Spatial Holdout)
    SPLIT_MODE = "kmeans"
    N_CLUSTERS = 150
    
    # DATA PATHS
    RAW_SOURCE = "REMOVED" 
    PARQUET_ROOT = "REMOVED"
    
    # OUTPUT PATH
    EXPERIMENT_DIR = f"REMOVED_{SPLIT_MODE}_k{N_CLUSTERS}"
    
    # Run Pipeline
    run_full_pipeline(
        npz_root=RAW_SOURCE, 
        experiment_root=EXPERIMENT_DIR, 
        parquet_root=PARQUET_ROOT,
        
        # Region Filter
        lon_bounds=(-10.0, 52.0),
        lat_bounds=(-20.0, 60.0),
        
        # Time Splits
        start_time=1931,
        train_end=2016,
        val_year=2017,
        test_start=2018,
        test_end=2019,
        
        # Split Parameters
        split_mode=SPLIT_MODE,
        n_clusters=N_CLUSTERS,
        val_frac=0.10,
        train_frac=0.80,
        seed=1234
    )