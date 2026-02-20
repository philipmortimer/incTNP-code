# 1. Converts netCDF files to paraquet -> one per station
import concurrent.futures as cf
import glob, os, uuid
from pathlib import Path
import numpy as np
import xarray as xr
import pyarrow as pa
import pyarrow.parquet as pq

# Cleans strings
def char2str(arr):
    a = np.asarray(arr).astype("U")
    if a.ndim == 0:
        return a.item().strip()
    if a.ndim == 1:
        return "".join(a.tolist()).strip()
    return "".join(a.ravel().tolist()).strip()


def process_one_file(path: str, out_root: Path):
    ds = xr.open_dataset(path, engine="h5netcdf")

    sid = char2str(ds["input_station_id"][0].values) # Only works for one nc file per station - CHECK
    elev = float(ds["elevation"])
    lat = float(ds["latitude"])
    lon = float(ds["longitude"])
    
    # Temp
    tvar = ds["temperatures"]
    temps = tvar.values.astype("float32")
    times = xr.decode_cf(ds[["time"]])["time"].values.astype("datetime64[ns]")

    # Checks for invalid data - max filled, missing, flagged etc
    valid = np.isfinite(temps)

    for key in ("_FillValue", "missing_value", "flagged_value"):
        v = tvar.encoding.get(key) or tvar.attrs.get(key)
        if v is not None and np.isfinite(v):
            valid &= temps != v

    vmin = tvar.attrs.get("valid_min")
    vmax = tvar.attrs.get("valid_max")
    if vmin is not None:
        valid &= temps >= vmin
    if vmax is not None:
        valid &= temps <= vmax

    # If no valid items station metadata still written but early return
    if not valid.any():
        return sid, lat, lon, elev

    temps = temps[valid]
    times = times[valid]

    tbl = pa.table({
        "station_id": pa.array([sid] * len(temps)),
        "latitude": pa.array([lat] * len(temps), pa.float32()),
        "longitude": pa.array([lon] * len(temps), pa.float32()),
        "elevation": pa.array([elev] * len(temps), pa.float32()),
        "time": pa.array(times, pa.timestamp("ns")),
        "temperature": pa.array(temps, pa.float32()),
    })

    part_dir = out_root / f"station_id={sid}"
    part_dir.mkdir(parents=True, exist_ok=True)
    fn = part_dir / f"part_{uuid.uuid4().hex}.parquet"
    pq.write_table(tbl, fn, compression="snappy")

    return sid, lat, lon, elev


# Takes a folder of .nc files and writes to another folder
def convert_nc_folder_to_para(nc_dir, out_root, n_workers=os.cpu_count()):
    paths = sorted(glob.glob(f"{nc_dir}/*.nc*")) # Gets all nc files
    if not paths:
        raise SystemExit("No NetCDF files found")

    out_root.mkdir(parents=True, exist_ok=True) # Ensures out folder exists

    meta_rows = []
    with cf.ThreadPoolExecutor(max_workers=n_workers) as pool:
        futs = [pool.submit(process_one_file, p, out_root) for p in paths]
        for i, fut in enumerate(cf.as_completed(futs), 1):
            meta_rows.append(fut.result())
            if i % 100 == 0 or i == len(paths):
                print(f"{i}/{len(paths)} files done")

    # Meta data for each station
    meta_tbl = pa.table({
        "station_id": pa.array([r[0] for r in meta_rows]),
        "latitude": pa.array([r[1] for r in meta_rows], pa.float32()),
        "longitude": pa.array([r[2] for r in meta_rows], pa.float32()),
        "elevation": pa.array([r[3] for r in meta_rows], pa.float32()),
    })
    pq.write_table(meta_tbl, out_root / "station_meta.parquet", compression="snappy")

    print("Written all to folder")

if __name__ == "__main__":
    nc_dir = "REMOVED"
    out_root = Path("REMOVED")
    convert_nc_folder_to_para(nc_dir, out_root)