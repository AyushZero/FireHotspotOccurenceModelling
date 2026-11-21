"""Fetch real hourly ERA5 California tiles from the public Earthmover Icechunk archive.

Run with the isolated Python 3.12 .venv_arco2 environment. Each whole-year
variable is loaded once because the source is chunked by year and 12x12 tiles.
Monthly NetCDFs are validated before promotion to the empirical raw archive.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import calendar
import hashlib
import json
import time

import icechunk
import pcodec  # noqa: F401
import xarray as xr
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data/raw/era5"
OUT.mkdir(parents=True, exist_ok=True)
VARS = ["t2m", "d2m", "u10", "v10", "swvl1"]
parser = argparse.ArgumentParser()
parser.add_argument("--year", type=int, required=True, choices=[2021, 2022, 2023, 2024])
args = parser.parse_args()
year = args.year

storage = icechunk.s3_storage(bucket="earthmover-icechunk-era5", prefix="icechunkV2",
                              region="us-east-1", anonymous=True)
repo = icechunk.Repository.open(storage)
session = repo.readonly_session("main")
ds = xr.open_zarr(session.store, group="single/temporal", consolidated=False, chunks=None)
assert set(VARS).issubset(ds.data_vars)
region = ds[VARS].sel(valid_time=slice(f"{year}-01-01", f"{year}-12-31T23:00"),
                       latitude=slice(42, 32), longitude=slice(236, 246))
assert region.sizes["valid_time"] == (8784 if calendar.isleap(year) else 8760)
assert region.sizes["latitude"] == region.sizes["longitude"] == 41
print(year, "selected", dict(region.sizes), flush=True)

arrays = {}
for name in VARS:
    for attempt in range(1, 5):
        try:
            print(year, name, "loading", "attempt", attempt, flush=True)
            arrays[name] = region[name].load()
            print(year, name, "loaded", flush=True)
            break
        except Exception as exc:
            print(year, name, type(exc).__name__, str(exc)[:200], flush=True)
            if attempt == 4:
                raise
            time.sleep(10 * attempt)

full = xr.Dataset(arrays).rename({"valid_time": "time"})
full = full.assign_coords(longitude=full.longitude - 360)
assert np.isfinite(full["t2m"].values).any()
assert 190 < float(full["t2m"].mean()) < 330

for month in range(1, 13):
    target = OUT / f"era5_{year}_{month:02d}.nc"
    tmp = OUT / f"era5_{year}_{month:02d}.earthmover.download"
    subset = full.sel(time=slice(f"{year}-{month:02d}-01",
                                 f"{year}-{month:02d}-{calendar.monthrange(year, month)[1]:02d}T23:00"))
    assert subset.sizes["time"] == calendar.monthrange(year, month)[1] * 24
    if target.exists():
        ref = xr.open_dataset(target)
        ref_t = "valid_time" if "valid_time" in ref else "time"
        if set(VARS).issubset(ref.data_vars) and ref.sizes[ref_t] == subset.sizes["time"]:
            one = {v: float(abs(ref[v].isel({ref_t: 0, "latitude": 20, "longitude": 20}).values -
                                 subset[v].isel(time=0, latitude=20, longitude=20).values)) for v in VARS}
            print(year, month, "existing CDS max sample delta", max(one.values()), flush=True)
            ref.close()
            continue
        ref.close()
        raise RuntimeError(f"Existing {target} failed validation")
    subset.to_netcdf(tmp, engine="netcdf4",
                     encoding={v: {"zlib": True, "complevel": 2} for v in VARS})
    check = xr.open_dataset(tmp)
    assert set(VARS).issubset(check.data_vars)
    assert check.sizes["time"] == subset.sizes["time"]
    assert check.sizes["latitude"] == check.sizes["longitude"] == 41
    check.close()
    tmp.rename(target)
    print(year, month, "saved", target.stat().st_size, flush=True)

manifest = {"source": "Earthmover public ERA5 Icechunk v2 archive, reanalysis-era5-single-levels",
            "url": "s3://earthmover-icechunk-era5/icechunkV2, group single/temporal",
            "year": year, "variables": VARS, "source_chunks": [8736, 12, 12],
            "files": {}}
for month in range(1, 13):
    target = OUT / f"era5_{year}_{month:02d}.nc"
    manifest["files"][target.name] = {"bytes": target.stat().st_size,
                                      "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
(OUT / f"earthmover_manifest_{year}.json").write_text(json.dumps(manifest, indent=2))
print(year, "complete", flush=True)
