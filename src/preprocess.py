"""
Preprocessing - daily aggregation of raw datasets to gridded daily values.

Handles ERA5, MODIS, SRTM, and FIRMS data.
"""

import os, glob, yaml
import numpy as np
import pandas as pd
import xarray as xr
from src.grid import build_grid, assign_to_grid, grid_info


# ══════════════════════════════════════════════════════════════════
# ERA5 -> daily aggregates
# ══════════════════════════════════════════════════════════════════

def _relative_humidity(t2m_K, d2m_K):
    """Magnus formula: compute RH (%) from T and dewpoint (both in K)."""
    t = t2m_K - 273.15
    td = d2m_K - 273.15
    # August-Roche-Magnus approximation
    rh = 100 * np.exp(17.625 * td / (243.04 + td)) / \
               np.exp(17.625 * t  / (243.04 + t))
    return np.clip(rh, 0, 100)


def process_era5(raw_dir, cfg):
    """
    Process ERA5 hourly NetCDFs -> daily grid-level aggregates.

    Output variables per cell per day:
      t_max   - daily max 2 m temperature ( degC)
      rh_min  - relative humidity at the hour of daily Tmax (%)
      u10_max - daily max 10 m wind speed (m/s)
      p_tot   - daily total precipitation (mm)
      sm_top  - daily mean top-layer soil moisture (m³/m³)

    Returns xr.Dataset with dims (time, latitude, longitude).
    """
    gi = grid_info(cfg)
    era5_dir = os.path.join(raw_dir, "era5")
    files = sorted(glob.glob(os.path.join(era5_dir, "era5_*.nc")))
    if not files:
        raise FileNotFoundError(f"No ERA5 files in {era5_dir}")

    all_years = []
    for f in files:
        print(f"  Processing {os.path.basename(f)}...")
        ds = xr.open_dataset(f)

        if "valid_time" in ds.dims or "valid_time" in ds.coords:
            ds = ds.rename({"valid_time": "time"})

        # Rename coords if needed
        if "lat" in ds.dims:
            ds = ds.rename({"lat": "latitude", "lon": "longitude"})

        # Compute wind speed
        if "v10" in ds:
            ws = np.sqrt(ds["u10"]**2 + ds["v10"]**2)
        else:
            ws = np.abs(ds["u10"])

        ds["ws"] = (ds["u10"].dims, ws.values)

        # Daily aggregation
        daily = xr.Dataset()
        daily["t_max"]   = (ds["t2m"] - 273.15).resample(time="1D").max()
        # Calculate RH with T and Td sampled at the hour of each cell's Tmax.
        # ERA5 supplies both inputs in kelvin; _relative_humidity converts both
        # to degrees Celsius before applying the Magnus relation.
        def rh_at_tmax(day):
            idx = day["t2m"].argmax(dim="time")
            t_at_max = day["t2m"].isel(time=idx)
            td_at_max = day["d2m"].isel(time=idx)
            return xr.DataArray(
                _relative_humidity(t_at_max.values, td_at_max.values),
                coords=t_at_max.coords, dims=t_at_max.dims,
            )
        rh_at_max = ds[["t2m", "d2m"]].groupby("time.date").map(rh_at_tmax)
        rh_at_max = rh_at_max.drop_vars("time").rename({"date": "time"})
        rh_at_max = rh_at_max.assign_coords(time=daily.time)
        daily["rh_min"] = rh_at_max
        daily["u10_max"] = ds["ws"].resample(time="1D").max()
        # Precipitation is not a model input. The empirical ERA5 acquisition
        # requests only the five instantaneous fields used by this paper.
        if "tp" in ds:
            daily["p_tot"] = ds["tp"].resample(time="1D").sum() * 1000
        daily["sm_top"]  = ds["swvl1"].resample(time="1D").mean()

        # Regrid to target lats/lons (nearest-neighbour interpolation)
        daily = daily.interp(latitude=gi["lats"], longitude=gi["lons"],
                             method="nearest")
        # CDS and the public ERA5 mirror attach different auxiliary scalar
        # coordinates (e.g. number/expver versus lsm). They are not inputs.
        daily = daily.reset_coords(drop=True).transpose("time", "latitude", "longitude")
        day_coords = np.unique(ds.time.values.astype("datetime64[D]")).astype("datetime64[ns]")
        assert daily.sizes["time"] == len(day_coords)
        daily = daily.assign_coords(time=day_coords)
        all_years.append(daily)
        ds.close()

    result = xr.concat(all_years, dim="time")
    # Sort by time, handling potential calendar mismatches
    result = result.assign_coords(time=result.time.values.astype("datetime64[ns]"))
    result = result.sortby("time")
    # Remove duplicates, keeping first occurrence
    _, unique_indices = np.unique(result.time.values, return_index=True)
    result = result.isel(time=unique_indices)
    return result


# ══════════════════════════════════════════════════════════════════
# MODIS -> daily NDVI, NDWI (interpolated)
# ══════════════════════════════════════════════════════════════════

def process_modis(raw_dir, cfg):
    """
    Process MODIS 16-day composites -> daily NDVI & NDWI via
    last-observation-carried-forward interpolation.

    Returns xr.Dataset with daily (time, latitude, longitude).
    """
    gi = grid_info(cfg)
    modis_dir = os.path.join(raw_dir, "modis")
    files = sorted(glob.glob(os.path.join(modis_dir, "*.nc")))
    # Do not mix the real AppEEARS extraction with synthetic demonstration files.
    appeears_files = [f for f in files if "MOD13A2" in os.path.basename(f)]
    if appeears_files:
        files = appeears_files
    if not files:
        raise FileNotFoundError(f"No MODIS files in {modis_dir}")

    all_years = []
    for f in files:
        print(f"  Processing {os.path.basename(f)}...")
        ds = xr.open_dataset(f)

        # Standardize coordinates
        if "lat" in ds.dims:
            ds = ds.rename({"lat": "latitude", "lon": "longitude"})

        # Handle NASA AppEEARS variable names (xarray auto-applies NetCDF scale_factor)
        if "_1_km_16_days_NDVI" in ds:
            ds["ndvi"] = ds["_1_km_16_days_NDVI"]
            nir = ds["_1_km_16_days_NIR_reflectance"]
            mir = ds["_1_km_16_days_MIR_reflectance"]
            ds["ndwi"] = (nir - mir) / (nir + mir + 1e-6)
        elif "NDVI" in ds:
            ds["ndvi"] = ds["NDVI"]
            if "NDWI" not in ds:
                raise ValueError("MODIS input has NDVI but no bands or layer for NDWI")
            ds["ndwi"] = ds["NDWI"]

        # Ensure ndvi and ndwi are within valid physical ranges
        if "ndvi" in ds:
            ds["ndvi"] = xr.where(ds["ndvi"] > 1.0, 1.0, ds["ndvi"])
            ds["ndvi"] = xr.where(ds["ndvi"] < -1.0, -1.0, ds["ndvi"])
        if "ndwi" in ds:
            ds["ndwi"] = xr.where(ds["ndwi"] > 1.0, 1.0, ds["ndwi"])
            ds["ndwi"] = xr.where(ds["ndwi"] < -1.0, -1.0, ds["ndwi"])

        # Regrid to target bounding box
        ds = ds.interp(latitude=gi["lats"], longitude=gi["lons"],
                       method="nearest")

        # Keep only the standardized variables we need
        ds_clean = ds[["ndvi", "ndwi"]]

        # The timestamp marks the beginning of a 16-day composite. Treat the
        # composite as available only after its window has finished.
        available = np.array([str(t)[:10] for t in ds_clean.time.values], dtype="datetime64[D]") + np.timedelta64(16, "D")
        ds_clean = ds_clean.assign_coords(time=available)
        # Resample to daily using only an already-completed composite.
        daily = ds_clean.resample(time="1D").ffill()
        all_years.append(daily)
        ds.close()

    result = xr.concat(all_years, dim="time")
    # Sort by time, handling potential calendar mismatches
    result = result.assign_coords(time=result.time.values.astype("datetime64[ns]"))
    result = result.sortby("time")
    # Remove duplicates, keeping first occurrence
    _, unique_indices = np.unique(result.time.values, return_index=True)
    result = result.isel(time=unique_indices)
    return result


# ══════════════════════════════════════════════════════════════════
# SRTM -> slope per grid cell (static)
# ══════════════════════════════════════════════════════════════════

def process_srtm(raw_dir, cfg):
    """
    Load precomputed slope grid.

    Returns xr.DataArray with dims (latitude, longitude).
    """
    srtm_file = os.path.join(raw_dir, "srtm", "srtm_slope.nc")
    if not os.path.exists(srtm_file):
        raise FileNotFoundError(f"SRTM slope file not found: {srtm_file}")

    gi = grid_info(cfg)
    ds = xr.open_dataset(srtm_file)
    slope = ds["slope"].interp(latitude=gi["lats"], longitude=gi["lons"],
                               method="nearest")
    return slope


# ══════════════════════════════════════════════════════════════════
# FIRMS -> ignition labels + fire history features
# ══════════════════════════════════════════════════════════════════

def process_firms(raw_dir, cfg):
    """
    Process FIRMS CSV -> per-cell per-day:
      y(t)         : binary ignition label (1 if any fire, else 0)
      frp_hist(t)  : exponentially decayed sum of FRP over sliding window
      count_hist(t): exponentially decayed count of hotspots over window

    Returns xr.Dataset with dims (time, latitude, longitude).
    """
    gi = grid_info(cfg)
    firms_dir = os.path.join(raw_dir, "firms")
    files = sorted(glob.glob(os.path.join(firms_dir, "firms_*.csv")))
    if not files:
        raise FileNotFoundError(f"No FIRMS files in {firms_dir}")

    window = cfg["firms"]["sliding_window_days"]
    decay  = cfg["firms"]["decay_factor"]
    conf_filter = cfg["firms"]["confidence_filter"]
    type_filter = cfg["firms"]["type_filter"]

    # Read and concatenate all FIRMS CSVs
    dfs = []
    for f in files:
        print(f"  Processing {os.path.basename(f)}...")
        df = pd.read_csv(f)
        dfs.append(df)
    firms = pd.concat(dfs, ignore_index=True)

    # Filter
    firms = firms[firms["confidence"].isin(conf_filter)]
    firms = firms[firms["type"] == type_filter]
    firms["acq_date"] = pd.to_datetime(firms["acq_date"])

    # Assign to grid cells
    lat_idx, lon_idx = assign_to_grid(
        firms["latitude"].values, firms["longitude"].values,
        gi["lats"][0] - gi["resolution"] / 2,
        gi["lons"][0] - gi["resolution"] / 2,
        gi["resolution"],
    )
    firms["lat_idx"] = lat_idx
    firms["lon_idx"] = lon_idx

    # Build daily time index
    start = pd.Timestamp(cfg["time"]["start"])
    end   = pd.Timestamp(cfg["time"]["end"])
    days  = pd.date_range(start, end, freq="D")

    n_lat = gi["n_lat"]
    n_lon = gi["n_lon"]
    n_days = len(days)

    # Allocate arrays
    y          = np.zeros((n_days, n_lat, n_lon), dtype=np.float32)
    daily_count = np.zeros_like(y)
    daily_frp = np.zeros_like(y)
    frp_hist   = np.zeros((n_days, n_lat, n_lon), dtype=np.float32)
    count_hist = np.zeros((n_days, n_lat, n_lon), dtype=np.float32)

    # -- Vectorized fire assignment --------------------------------
    # Filter to valid grid cells and time range
    valid_mask = (
        (firms["lat_idx"] >= 0) & (firms["lat_idx"] < n_lat) &
        (firms["lon_idx"] >= 0) & (firms["lon_idx"] < n_lon) &
        (firms["acq_date"] >= start) & (firms["acq_date"] <= end)
    )
    valid_firms = firms[valid_mask].copy()
    valid_firms["tidx"] = (valid_firms["acq_date"] - start).dt.days

    # Use np.add.at for vectorized accumulation (handles duplicate indices)
    t_idx = valid_firms["tidx"].values.astype(np.int32)
    lat_indices = valid_firms["lat_idx"].values.astype(np.int32)
    lon_indices = valid_firms["lon_idx"].values.astype(np.int32)
    np.add.at(daily_count, (t_idx, lat_indices, lon_indices), 1.0)
    np.add.at(daily_frp, (t_idx, lat_indices, lon_indices), valid_firms["frp"].fillna(0).to_numpy(np.float32))
    y = (daily_count > 0).astype(np.float32)

    print(f"  Assigned {len(valid_firms)} fire detections to grid")

    # -- Vectorized sliding-window history (10-100x faster) --------
    # Build decay weights: [decay^1, decay^2, ..., decay^window]
    decay_weights = decay ** np.arange(1, window + 1)

    # Apply each lag efficiently using array slicing
    for k in range(1, window + 1):
        w = decay ** k
        frp_hist[k:]   += w * daily_frp[:-k]
        count_hist[k:] += w * daily_count[:-k]

    print(f"  Computed {window}-day sliding window history")

    result = xr.Dataset(
        {
            "ignition":   (["time", "latitude", "longitude"], y),
            "frp_hist":   (["time", "latitude", "longitude"], frp_hist),
            "count_hist": (["time", "latitude", "longitude"], count_hist),
        },
        coords={
            "time":      days,
            "latitude":  gi["lats"],
            "longitude": gi["lons"],
        },
    )
    return result


# -- Self-test -----------------------------------------------------
if __name__ == "__main__":
    import yaml
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    raw_dir = os.path.join(root, cfg["paths"]["raw_data"])

    print("Testing ERA5 processing ...")
    try:
        era5 = process_era5(raw_dir, cfg)
        print(f"  ERA5: {dict(era5.dims)}")
    except FileNotFoundError as e:
        print(f"  Skipped: {e}")
