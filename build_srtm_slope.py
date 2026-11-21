"""Compute mean native-pixel SRTM slope within each 0.25-degree grid cell.

Reads the three source GeoTIFF chunks by window so the full DEM is never
assembled in memory. Slope uses central finite differences in metres.
"""
from pathlib import Path
import glob
import numpy as np
import rasterio
from rasterio.windows import Window
import xarray as xr
import yaml

ROOT = Path(__file__).resolve().parent
cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
region = cfg["region"]
step = cfg["grid"]["resolution"]
nlat = round((region["lat_max"] - region["lat_min"]) / step)
nlon = round((region["lon_max"] - region["lon_min"]) / step)
lat_centres = region["lat_min"] + (np.arange(nlat) + .5) * step
lon_centres = region["lon_min"] + (np.arange(nlon) + .5) * step
raw_dir = (ROOT / cfg["paths"]["raw_data"] / "srtm").resolve()
files = sorted(glob.glob(str(raw_dir / "*.tif")))
if not files:
    raise FileNotFoundError(f"No SRTM GeoTIFFs in {raw_dir}")

total = np.zeros(nlat * nlon, dtype=np.float64)
count = np.zeros(nlat * nlon, dtype=np.int64)
for path in files:
    with rasterio.open(path) as src:
        if src.crs.to_epsg() != 4326:
            raise ValueError(f"Expected EPSG:4326: {path}")
        x_step, y_step = abs(src.transform.a), abs(src.transform.e)
        for row in range(0, src.height, 1024):
            height = min(1024, src.height - row)
            for col in range(0, src.width, 1024):
                width = min(1024, src.width - col)
                r0, c0 = max(0, row - 1), max(0, col - 1)
                r1, c1 = min(src.height, row + height + 1), min(src.width, col + width + 1)
                elevation = src.read(1, window=Window(c0, r0, c1 - c0, r1 - r0), masked=True).astype(np.float32).filled(np.nan)
                gy, gx = np.gradient(elevation)
                rs, cs = slice(row-r0, row-r0+height), slice(col-c0, col-c0+width)
                lat = src.transform.f + (row + np.arange(height) + .5) * src.transform.e
                lon = src.transform.c + (col + np.arange(width) + .5) * src.transform.a
                metres_y = 111_195.0 * y_step
                metres_x = 111_195.0 * x_step * np.cos(np.deg2rad(lat))
                slope = np.degrees(np.arctan(np.hypot(gx[rs, cs] / metres_x[:, None], gy[rs, cs] / metres_y)))
                ilat = np.floor((lat - region["lat_min"]) / step).astype(np.int32)
                ilon = np.floor((lon - region["lon_min"]) / step).astype(np.int32)
                ids = ilat[:, None] * nlon + ilon[None, :]
                valid = np.isfinite(slope) & (ilat[:, None] >= 0) & (ilat[:, None] < nlat) & (ilon[None, :] >= 0) & (ilon[None, :] < nlon)
                total += np.bincount(ids[valid], weights=slope[valid], minlength=nlat*nlon)
                count += np.bincount(ids[valid], minlength=nlat*nlon)
    print(f"Processed {Path(path).name}", flush=True)

mean = np.divide(total, count, out=np.full_like(total, np.nan), where=count > 0).reshape(nlat, nlon).astype(np.float32)
out = raw_dir / "srtm_slope.nc"
xr.Dataset({"slope": (("latitude", "longitude"), mean)}, coords={"latitude": lat_centres, "longitude": lon_centres}).to_netcdf(out)
print(f"Saved {out}; covered cells={np.isfinite(mean).sum()}/{nlat*nlon}; range={np.nanmin(mean):.2f}..{np.nanmax(mean):.2f} degrees", flush=True)
