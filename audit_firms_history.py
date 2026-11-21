"""Observed-label sanity check independent of weather inputs."""
from pathlib import Path
import json
import numpy as np
import yaml
import geopandas as gpd
from shapely.geometry import Point
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from src.preprocess import process_firms

ROOT = Path(__file__).resolve().parent
cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
ds = process_firms(ROOT / "data/raw", cfg)
states = gpd.read_file("zip://" + str(ROOT / "outputs/cb_2024_us_state_500k.zip"))
ca = states.loc[states.STUSPS == "CA", "geometry"].iloc[0]
mask = np.array([[ca.covers(Point(float(lon), float(lat)))
                  for lon in ds.longitude.values] for lat in ds.latitude.values])
y = ds.ignition.values[:, mask].astype(int)
lag1 = np.zeros_like(y)
lag1[1:] = y[:-1]
prev7 = np.zeros_like(y)
for k in range(1, 8):
    prev7[k:] = np.maximum(prev7[k:], y[:-k])
years = np.array([int(str(t)[:4]) for t in ds.time.values])
x = np.stack([lag1, prev7], axis=-1)
tr, va, te = (years == yr for yr in (2021, 2022, 2023))
model = LogisticRegression(max_iter=1000).fit(x[tr].reshape(-1, 2), y[tr].ravel())
pred = model.predict_proba(x[te].reshape(-1, 2))[:, 1]
result = {
    "california_cell_centers": int(mask.sum()),
    "positive_cell_days": {str(yr): int(y[years == yr].sum()) for yr in (2021, 2022, 2023)},
    "base_rate": {str(yr): float(y[years == yr].mean()) for yr in (2021, 2022, 2023)},
    "p_event_given_yesterday": {
        str(yr): float(y[years == yr][lag1[years == yr] == 1].mean())
        for yr in (2021, 2022, 2023)
    },
    "persistence_test_auc": float(roc_auc_score(y[te].ravel(), pred)),
}
(ROOT / "outputs/real_firms_history_audit.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
