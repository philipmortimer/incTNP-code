# Script that precomputes the list of valid timestamps and the number of observations available at each time stamp
import os, sys, time
from pathlib import Path
import duckdb
import pyarrow.dataset as ds
import shutil

# Summary writer as before
def write_summary(path: Path, d: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for k, v in d.items():
            f.write(f"{k}: {v}\n")


# Checks to see if exact cached folder for path already exists
def is_file_cached(split_dir: str, min: int, max: int):
    split_dir_path = Path(split_dir)
    folder_nm = f"minn_{min}_maxn_{max}"
    for p in split_dir_path.iterdir():
        if p.is_dir and p.name == folder_nm: return True
    return False


# Used to visualise file to verify that results are sensible
def safe_head_tail(counts_file: Path, n=5):
    tbl = ds.dataset(counts_file).to_table()
    rows = tbl.num_rows
    head = tbl.slice(0, min(n, rows))
    tail = tbl.slice(max(rows - n, 0), min(n, rows))
    return head, tail

# Caches the obs per timestamps and returns the max practical number
def cache_n_rows(split_dir, min_needed, max_cap, show: bool=False):
    cache_dir = is_file_cached(split_dir, min_needed, max_cap)
    if cache_dir:
        print(f"Exact cache already present: {cache_dir}")
        return max_cap
    else: print("Cache file not found building from scratch")
    split_dir = Path(split_dir).resolve()
    data_file = split_dir / "data" / "data.parquet"
    if not data_file.exists():
        sys.exit(f"Data file not found exiting - {data_file}")


    # Builds cache
    cache_dir = split_dir / f"minn_{min_needed}_maxn_{max_cap}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    counts_file = cache_dir / "counts.parquet"

    t0 = time.time()
    con = duckdb.connect()
    con.execute("PRAGMA threads=%d" % os.cpu_count())
    # Filter bu min
    con.execute(f"""
        COPY (
        SELECT time AS hour_int,
                COUNT(*)::INT AS n_obs
        FROM read_parquet('{data_file.as_posix()}')
        GROUP BY time
        HAVING COUNT(*) >= {min_needed}
        ORDER BY time
        )
        TO '{counts_file.as_posix()}'
        (FORMAT PARQUET, COMPRESSION 'snappy')
    """)
    # Get number of readings at each time point
    n_rows, max_practical = con.execute(
        f"SELECT COUNT(*), max(n_obs) FROM read_parquet('{counts_file.as_posix()}')"
    ).fetchone()
    
    if n_rows == 0:
        counts_file.unlink(missing_ok=True)
        sys.exit(f"no times satisfying min_needed={min_needed} found.")

    elapsed = time.time() - t0
    print(f"built {n_rows:,} rows in {elapsed:.1f}s")

    # Adds a summary file to the file also
    parent_summary = {}
    sfile = split_dir / "summary.txt"
    summary = {
        "min_needed": min_needed,
        "max_cap_user": max_cap,
        "max_obs_practical": int(max_practical),
        "n_timestamps": n_rows,
        "source_file": data_file.as_posix(),
        "build_seconds": f"{elapsed:.2f}",
    }
    write_summary(cache_dir / "summary.txt", summary)

    # Optionally prints some file values
    if show:
        head, tail = safe_head_tail(counts_file)
        print("\nHEAD (first 5 rows)")
        print(head.to_pandas())
        print("\nTAIL (last 5 rows)")
        print(tail.to_pandas())

    return max_practical
