"""Download a requested year of VIIRS/SNPP standard-processing data, resumably.

The MAP_KEY is read from FIRMS_MAP_KEY (or this folder's config.yaml) and is never
written to logs or the empirical handoff. Each five-day response is validated
and saved before annual CSVs are assembled.
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
import hashlib
import io
import json
import os
import argparse
import time

import pandas as pd
import requests
import yaml

HERE = Path(__file__).resolve().parent
KEY_CONFIG = HERE / "config.yaml"
OUT = HERE / "data/raw/firms"
CHUNKS = OUT / "chunks"
OUT.mkdir(parents=True, exist_ok=True)
CHUNKS.mkdir(exist_ok=True)
key = os.environ.get("FIRMS_MAP_KEY") or yaml.safe_load(KEY_CONFIG.read_text())["firms"].get("map_key", "")
if not key:
    raise RuntimeError("Set FIRMS_MAP_KEY or firms.map_key in config.yaml")

session = requests.Session()
manifest = {"source": "NASA FIRMS area API, VIIRS_SNPP_SP", "bbox": [-124, 32, -114, 42], "years": {}}
parser=argparse.ArgumentParser(); parser.add_argument('--year',type=int,default=2024); args=parser.parse_args()
for year in (args.year,):
    parts = []
    day = date(year, 1, 1)
    last = date(year, 12, 31)
    while day <= last:
        end = min(day + timedelta(days=4), last)
        n = (end - day).days + 1
        path = CHUNKS / f"{day.isoformat()}_{end.isoformat()}.csv"
        if not path.exists():
            url = ("https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
                   + key + f"/VIIRS_SNPP_SP/-124,32,-114,42/{n}/{day.isoformat()}")
            for attempt in range(6):
                try:
                    response = session.get(url, timeout=120)
                    if response.status_code == 429 or response.status_code >= 500:
                        raise RuntimeError(f"HTTP {response.status_code}")
                    if response.status_code != 200:
                        raise RuntimeError(f"HTTP {response.status_code}")
                    frame = pd.read_csv(io.StringIO(response.text))
                    if not {"latitude", "longitude", "acq_date", "confidence", "type", "frp"}.issubset(frame):
                        raise RuntimeError("Unexpected FIRMS response columns")
                    if len(frame) and not frame.acq_date.between(day.isoformat(), end.isoformat()).all():
                        raise RuntimeError("Unexpected FIRMS response dates")
                    path.write_bytes(response.content)
                    print(f"FIRMS {day}..{end}: {len(frame)} rows", flush=True)
                    break
                except Exception as exc:
                    # Never print the URL, which contains the private MAP_KEY.
                    print(f"FIRMS {day} attempt {attempt+1} failed: {type(exc).__name__}", flush=True)
                    if attempt == 5:
                        raise RuntimeError(f"FIRMS chunk {day} could not be fetched") from None
                    time.sleep(min(60, 5 * 2**attempt))
        parts.append(path)
        day = end + timedelta(days=1)

    frames = [pd.read_csv(path) for path in parts]
    annual = pd.concat(frames, ignore_index=True)
    annual = annual.drop_duplicates()
    assert annual.acq_date.between(f"{year}-01-01", f"{year}-12-31").all()
    target = OUT / f"firms_{year}.csv"
    annual.to_csv(target, index=False)
    manifest["years"][str(year)] = {
        "rows": len(annual), "chunks": len(parts),
        "first_date": str(annual.acq_date.min()) if len(annual) else None,
        "last_date": str(annual.acq_date.max()) if len(annual) else None,
        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "type_counts": {str(k): int(v) for k, v in annual.type.value_counts().items()},
        "confidence_counts": {str(k): int(v) for k, v in annual.confidence.value_counts().items()},
    }
    (OUT / f"manifest_{year}.json").write_text(json.dumps(manifest, indent=2))
    print(f"FIRMS {year} complete: {len(annual)} rows", flush=True)

print("FIRMS archive complete", flush=True)
