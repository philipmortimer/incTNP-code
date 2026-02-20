# Uses a DEM to get heights used for predicting on the grid - and caches where possible
from pathlib import Path
import hashlib
import numpy as np
import xarray as xr

# Creates a file name based on these fields using caching
def _cache_key(lat_range, lon_range, n_points):
    key_str = f"{lat_range}_{lon_range}_{n_points}"
    return hashlib.sha1(key_str.encode()).hexdigest()[:16] + ".npz"


# Loads saved grid if it exists or regens it otherwise
def get_cached_elevation_grid(lat_range, lon_range, n_points, cache_dir, dem_path):
    cache_file = Path(cache_dir) / _cache_key(lat_range, lon_range, n_points)
    if cache_file.exists():
        print("Using cached elevation")
        data = np.load(cache_file)
        return data["lat"], data["lon"], data["elev"]

    # Opens DEM and loads data
    ds = xr.open_dataset(dem_path)
    lats = np.linspace(lat_range[0], lat_range[1], n_points)
    lons = np.linspace(lon_range[0], lon_range[1], n_points)

    elev = ds["z"].interp(lat=("lat", lats), lon=("lon", lons), method="nearest")
    elev_np = elev.values.astype(np.float32)
    lon_mesh, lat_mesh = np.meshgrid(lons, lats)
    np.savez_compressed(cache_file, lat=lat_mesh, lon=lon_mesh, elev=elev_np)
    print("Cached new elevation data")
    return lat_mesh, lon_mesh, elev_np
