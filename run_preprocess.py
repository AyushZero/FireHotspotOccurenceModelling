#!/usr/bin/env python3
"""
Preprocessing pipeline - end-to-end.

Flow:
  1. Read raw data (ERA5, MODIS, SRTM, FIRMS)
  2. Grid & aggregate to daily 0.25 deg resolution
  3. Compute training-year min-max normalisation parameters
  4. Normalise all drivers
  5. Compute physics-guided features (F_avail, F_dry, G_spread, H_history, R_phys)
  6. Save feature dataset as NetCDF

Usage:
    python run_preprocess.py                 # from real data
"""

import argparse, os, sys, yaml
import numpy as np
import xarray as xr

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.grid import grid_info
from src.preprocess import process_era5, process_modis, process_srtm, process_firms
from src.normalize import compute_norm_params, normalize, save_params
from src.features import build_all_features


def main():
    # Abort before writing features if any observed source month is absent.
    import audit_real_inputs  # noqa: F401
    # Load config
    with open(os.path.join(ROOT, "config.yaml")) as f:
        cfg = yaml.safe_load(f)

    raw_dir       = os.path.join(ROOT, cfg["paths"]["raw_data"])
    processed_dir = os.path.join(ROOT, cfg["paths"]["processed_data"])
    features_dir  = os.path.join(ROOT, cfg["paths"]["features"])
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(features_dir, exist_ok=True)

    gi = grid_info(cfg)
    print(f"Grid: {gi['n_lat']} lat x {gi['n_lon']} lon = "
          f"{gi['n_lat'] * gi['n_lon']} cells\n")

    # Step 1: Process raw data
    print("=" * 60)
    print("Step 1 - Processing raw data...")
    print("=" * 60)

    print("\n[ERA5]")
    era5 = process_era5(raw_dir, cfg)

    print("\n[MODIS]")
    modis = process_modis(raw_dir, cfg)

    print("\n[SRTM]")
    slope = process_srtm(raw_dir, cfg)

    print("\n[FIRMS]")
    firms = process_firms(raw_dir, cfg)

    # Step 2: Align temporal dimensions
    print("\n" + "=" * 60)
    print("Step 2 - Aligning time dimensions...")
    print("=" * 60)

    # Use FIRMS time as reference (daily, full period)
    common_times = firms.time.values

    era5  = era5.reindex(time=common_times, method="nearest")
    modis = modis.reindex(time=common_times, method="ffill")

    # Broadcast static slope to all time steps
    slope_3d = slope.broadcast_like(era5["t_max"])

    print(f"  Aligned {len(common_times)} days")

    # Step 3: Compute normalisation parameters
    print("\n" + "=" * 60)
    print("Step 3 - Computing normalisation parameters...")
    print("=" * 60)

    raw_drivers = {
        "t_max":      era5["t_max"],
        "rh_min":     era5["rh_min"],
        "u10_max":    era5["u10_max"],
        "sm_top":     era5["sm_top"],
        "ndvi":       modis["ndvi"],
        "ndwi":       modis["ndwi"],
        "slope":      slope,
        "frp_hist":   firms["frp_hist"],
        "count_hist": firms["count_hist"],
    }

    variables = list(raw_drivers.keys())
    # Fit every continuous-feature transform on the training period only.
    # The same frozen limits are then applied to validation and test years.
    train_years = cfg["time"]["train_years"]
    train_mask = era5["time"].dt.year.isin(train_years)
    if cfg["region"].get("california_mask", False):
        import geopandas as gpd
        from shapely.geometry import Point
        boundary = gpd.read_file("zip://" + os.path.join(ROOT, "outputs", "cb_2024_us_state_500k.zip"))
        ca = boundary.loc[boundary.STUSPS == "CA", "geometry"].iloc[0]
        inside = np.array([[ca.covers(Point(float(lon), float(lat))) for lon in gi["lons"]]
                           for lat in gi["lats"]])
        ca_mask = xr.DataArray(inside, dims=["latitude", "longitude"],
                               coords={"latitude": gi["lats"], "longitude": gi["lons"]})
    else:
        ca_mask = None
    train_drivers = {}
    for name, value in raw_drivers.items():
        train_value = value.sel(time=train_mask) if "time" in value.dims else value
        train_drivers[name] = train_value.where(ca_mask) if ca_mask is not None else train_value
    norm_params = compute_norm_params(train_drivers, variables)
    save_params(norm_params, os.path.join(processed_dir, "norm_params.json"))

    # Step 4: Normalise
    print("\n" + "=" * 60)
    print("Step 4 - Normalising drivers...")
    print("=" * 60)

    normed = {}
    for var in variables:
        data = raw_drivers[var]
        normed[var] = normalize(data, norm_params[var])
        print(f"  {var:>12s} -> [{float(np.nanmin(normed[var])):.3f}, "
              f"{float(np.nanmax(normed[var])):.3f}]")

    # Step 5: Compute physics features
    print("\n" + "=" * 60)
    print("Step 5 - Computing physics-guided features...")
    print("=" * 60)

    features = build_all_features(normed, cfg)
    for name, arr in features.items():
        arr_np = np.asarray(arr)
        print(f"  {name:>12s}  range=[{float(np.nanmin(arr_np)):.4f}, "
              f"{float(np.nanmax(arr_np)):.4f}]")

    # Step 6: Save feature dataset
    print("\n" + "=" * 60)
    print("Step 6 - Saving feature dataset...")
    print("=" * 60)

    # Build combined Dataset
    ds = xr.Dataset(
        coords={
            "time":      common_times,
            "latitude":  gi["lats"],
            "longitude": gi["lons"],
        }
    )

    # Add normalised raw drivers
    for var in variables:
        if hasattr(normed[var], "dims") and "time" in normed[var].dims:
            ds[var] = normed[var]
        else:
            # Static variable (slope) - broadcast
            ds[var] = (["latitude", "longitude"], np.asarray(normed[var]))

    # Add physics features
    for name, arr in features.items():
        if hasattr(arr, "dims") and "time" in arr.dims:
            ds[name] = arr
        else:
            ds[name] = (["latitude", "longitude"], np.asarray(arr))

    # Add ignition label
    ds["ignition"] = firms["ignition"]

    outpath = os.path.join(features_dir, "features.nc")
    ds.to_netcdf(outpath)
    print(f"  [OK] Features saved -> {outpath}")
    print(f"  Dataset dimensions: {dict(ds.dims)}")
    print(f"  Variables: {list(ds.data_vars)}")

    print("\n[DONE] Preprocessing complete!\n")


if __name__ == "__main__":
    main()
