# 2 Converts per-stations Paraquet files to to NPZ per-station
import os
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds


EPOCH_YEAR = 1931

def process_one_station(station_id: str, parquet_root: Path, out_root: Path):
    station_dir = parquet_root / f"station_id={station_id}"
    out_root.mkdir(parents=True, exist_ok=True)
    out_path = out_root / f"{station_id}.npz"

    # Skips existing station
    if out_path.exists():
        print(f"[SKIP] {station_id} (already exists)")
        return

    if not station_dir.exists():
        print(f"[WARN] No Parquet directory for station {station_id}, writing empty arrays")
        np.savez(
            out_path,
            time_hours=np.array([], dtype=np.int64),
            temperature=np.array([], dtype=np.float32),
            valid_mask=np.array([], dtype=bool),
            latitude=np.float32(np.nan),
            longitude=np.float32(np.nan),
            elevation=np.float32(np.nan),
        )
        return

    files = sorted(station_dir.glob("*.parquet"))
    if not files:
        print(f"[WARN] No Parquet files for station {station_id}, writing empty arrays")
        np.savez(
            out_path,
            time_hours=np.array([], dtype=np.int64),
            temperature=np.array([], dtype=np.float32),
            valid_mask=np.array([], dtype=bool),
            latitude=np.float32(np.nan),
            longitude=np.float32(np.nan),
            elevation=np.float32(np.nan),
        )
        return

    # Gets station info and writes it to npz
    station_ds = ds.dataset(files, format="parquet")
    tbl = station_ds.to_table(columns=["time", "temperature", "latitude", "longitude", "elevation"])

    n_rows = tbl.num_rows
    if n_rows == 0:
        print(f"[WARN] Station {station_id} has 0 rows, writing empty arrays")
        np.savez(
            out_path,
            time_hours=np.array([], dtype=np.int64),
            temperature=np.array([], dtype=np.float32),
            valid_mask=np.array([], dtype=bool),
            latitude=np.float32(np.nan),
            longitude=np.float32(np.nan),
            elevation=np.float32(np.nan),
        )
        return

    time_arr = tbl["time"].to_pandas().values
    temp_arr = tbl["temperature"].to_numpy().astype("float32")
    lat = float(tbl["latitude"][0].as_py())
    lon = float(tbl["longitude"][0].as_py())
    elev = float(tbl["elevation"][0].as_py())

    # Temporal ordering
    order = np.argsort(time_arr)
    time_arr = time_arr[order]
    temp_arr = temp_arr[order]
    epoch0 = np.datetime64(f"{EPOCH_YEAR:04d}-01-01T00:00:00")
    time_deltas = (time_arr - epoch0).astype("timedelta64[h]")
    time_hours = time_deltas.astype("int64")
    valid_mask = np.isfinite(temp_arr)

    np.savez( # Saves sttations
        out_path,
        time_hours=time_hours,
        temperature=temp_arr,
        valid_mask=valid_mask.astype(bool),
        latitude=np.float32(lat),
        longitude=np.float32(lon),
        elevation=np.float32(elev),
    )
    print(f"[OK] {station_id} to {out_path} (N={len(time_hours)})")


def load_station_ids(meta_path: Path):
    tbl = pq.read_table(meta_path)
    return [str(sid) for sid in tbl["station_id"].to_pylist()]


def convert_parquet_folder_to_npz(parquet_root: Path, npz_root: Path):
    meta_path = parquet_root / "station_meta.parquet"
    if not meta_path.exists():
        raise SystemExit(f"No station_meta.parquet found in {parquet_root}")

    station_ids = load_station_ids(meta_path)
    if not station_ids:
        raise SystemExit("No stations found in station_meta.parquet")

    out_root = npz_root / "station_data"
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(station_ids)} stations")
    for i, sid in enumerate(station_ids, 1):
        process_one_station(sid, parquet_root, out_root)
        if i % 100 == 0 or i == len(station_ids):
            print(f"{i}/{len(station_ids)} stations done")

    print("Written all NPZ files to folder")


if __name__ == "__main__":
    # Defaults change as needed for filesystem
    parquet_root = Path("REMOVED")
    npz_root = Path("REMOVED")

    convert_parquet_folder_to_npz(parquet_root, npz_root)
