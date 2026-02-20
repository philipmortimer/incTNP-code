# Allows for data to be moved entirely to RAM for HPC systems
import os
import shutil
import tempfile
import logging
from pathlib import Path
from contextlib import contextmanager
import time

log = logging.getLogger(__name__)

# If preload_ram is True in config file, copies data to dev/shm and cleans it at run end
@contextmanager
def RamDiskDataLoader(use_ram, original_data_root):
    #use_ram = getattr(config.misc, "preload_ram", False)
    
    # Checks for linux cluster
    ram_disk_root = "/dev/shm"
    if use_ram and not os.path.exists(ram_disk_root):
        log.warning(f"{ram_disk_root} not found. Ignoring preload_ram=True.")
        use_ram = False

    # No behaviour change if flag not defined
    if not use_ram:
        yield Path(original_data_root)
        return

    # Setup ram disk
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    temp_dir = tempfile.mkdtemp(prefix=f"job_{job_id}_", dir=ram_disk_root)
    temp_path = Path(temp_dir)
    original_path = Path(original_data_root)

    t0 = time.time()
    try:
        print(f"Preloading data to RAM: {temp_dir} ...")
        
        # Copies relevant files (based on file ending)
        extensions = {".npy", ".npz", ".parquet", ".json"}
        
        files_copied = 0
        for file_name in os.listdir(original_path):
            if any(file_name.endswith(ext) for ext in extensions):
                src = original_path / file_name
                dst = temp_path / file_name
                shutil.copy2(src, dst)
                files_copied += 1
        
        duration = time.time() - t0
        print(f"Copied {files_copied} files in {duration:.1f}s. Training starting...")
        
        # Yield the new path to the training script
        yield temp_path

    finally:
        # Clean up ram as best practice
        print(f"Cleaning up RAM disk: {temp_dir}")
        shutil.rmtree(temp_dir, ignore_errors=True)