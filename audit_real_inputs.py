"""Validate observed source coverage before fitting any empirical model."""
from __future__ import annotations

from pathlib import Path
import calendar
import hashlib
import json

import numpy as np
import pandas as pd
import xarray as xr
import yaml

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data/raw"
cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
record = {"weather_source": "ERA5 hourly single levels: public Earthmover Icechunk mirror; one sampled grid-point hour checked against CDS in January 2021 and December 2023",
          "firms_source": "NASA FIRMS VIIRS_SNPP_SP area API",
          "modis_source": "NASA LP DAAC AppEEARS MOD13A2.061",
          "srtm_source": "USGS SRTMGL1 tiles", "era5": {}, "firms": {}}
needed = {"t2m", "d2m", "u10", "v10", "swvl1"}
for year in (2021, 2022, 2023):
    for month in range(1, 13):
        name = f"era5_{year}_{month:02d}.nc"
        path = RAW / "era5" / name
        if not path.exists():
            raise FileNotFoundError(path)
        ds = xr.open_dataset(path)
        time_name = "time" if "time" in ds else "valid_time"
        n_hours = calendar.monthrange(year, month)[1] * 24
        assert needed.issubset(ds.data_vars), name
        assert ds.sizes[time_name] == n_hours, (name, ds.sizes[time_name])
        dates = pd.to_datetime(ds[time_name].values)
        assert dates[0] == pd.Timestamp(year, month, 1)
        assert dates[-1] == pd.Timestamp(year, month, calendar.monthrange(year, month)[1], 23)
        assert ds.sizes["latitude"] >= 40 and ds.sizes["longitude"] >= 40
        sample = ds["t2m"].isel({time_name: 0}).values
        assert np.isfinite(sample).any() and 190 < np.nanmean(sample) < 330
        record["era5"][name] = {"hours": n_hours, "bytes": path.stat().st_size,
                                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        ds.close()
    path = RAW / "firms" / f"firms_{year}.csv"
    df = pd.read_csv(path)
    assert {"latitude", "longitude", "acq_date", "confidence", "type", "frp"}.issubset(df)
    assert df.acq_date.between(f"{year}-01-01", f"{year}-12-31").all()
    assert set(df.confidence.unique()).issubset({"l", "n", "h"})
    assert set(df.type.unique()).issubset({0, 1, 2, 3})
    record["firms"][str(year)] = {"rows": len(df),
                                    "presumed_vegetation_nominal_high": int(((df.type == 0) & df.confidence.isin(["n", "h"])).sum()),
                                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

modis = xr.open_dataset(RAW / "modis/MOD13A2.061_1km_aid0001.nc")
assert modis.sizes["time"] >= 60
assert str(modis.time.values[0])[:10] <= "2021-01-01"
assert str(modis.time.values[-1])[:10] >= "2023-12-01"
record["modis"] = {"composites": modis.sizes["time"],
                    "first_start": str(modis.time.values[0])[:10],
                    "last_start": str(modis.time.values[-1])[:10]}
modis.close()
for path in sorted((RAW / "srtm").glob("*.tif")):
    record.setdefault("srtm", []).append({"file": path.name, "bytes": path.stat().st_size})
assert len(record["srtm"]) == 3
(ROOT / "outputs/real_input_audit.json").write_text(json.dumps(record, indent=2))
print("All 36 ERA5 months and 3 FIRMS years validated", flush=True)
