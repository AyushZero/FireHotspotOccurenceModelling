"""Check observed MODIS and SRTM availability inside the California mask."""
from pathlib import Path
import json

import geopandas as gpd
import numpy as np
from shapely.geometry import Point
import yaml

from src.preprocess import process_modis, process_srtm

ROOT = Path(__file__).resolve().parent
cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
modis = process_modis(ROOT / "data/raw", cfg)
slope = process_srtm(ROOT / "data/raw", cfg)
states = gpd.read_file("zip://" + str(ROOT / "outputs/cb_2024_us_state_500k.zip"))
ca = states.loc[states.STUSPS == "CA", "geometry"].iloc[0]
mask = np.array([[ca.covers(Point(float(lon), float(lat))) for lon in slope.longitude.values]
                 for lat in slope.latitude.values])
ndvi = modis.ndvi.values
ndwi = modis.ndwi.values
valid = mask[None, :, :] & np.isfinite(ndvi) & np.isfinite(ndwi) & np.isfinite(slope.values)[None, :, :]
report = {"california_cells": int(mask.sum()),
          "california_slope_finite_cells": int(np.isfinite(slope.values[mask]).sum()),
          "california_cell_days": int(mask.sum() * modis.sizes["time"]),
          "model_ready_modis_slope_cell_days": int(valid.sum()),
          "model_ready_fraction": float(valid.sum() / (mask.sum() * modis.sizes["time"]))}
(ROOT / "outputs/spatial_coverage_audit.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
