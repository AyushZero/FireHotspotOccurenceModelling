"""Ranking and calibration audit for the validation-selected additive fit."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss
from src.dataset import build_dataset, split_by_year

ROOT = Path(__file__).resolve().parent
out = ROOT / "outputs/review_followup.json"
d = json.loads(out.read_text())
primary = d["selection"]["primary"]
saved = np.load(ROOT / "outputs/model_predictions_test.npz")
p = saved[primary]
yte = saved["label"]
frame = build_dataset(ROOT / "data/features")
_, _, te = split_by_year(frame, [2021], [2022], [2023])
assert len(p) == len(te) == len(yte)
assert abs(brier_score_loss(yte, p) - d["models"][primary]["metrics"]["test"]["brier"]) < 1e-9
threshold = float(np.quantile(p, .9))
top = p >= threshold
selection_fraction = float(top.mean())
edges = np.unique(np.quantile(p, np.linspace(0, 1, 11)))
indices = np.clip(np.digitize(p, edges[1:-1]), 0, len(edges) - 2)
reliability = [
    {"n": int((indices == i).sum()),
     "mean_predicted": float(p[indices == i].mean()),
     "observed_rate": float(yte[indices == i].mean())}
    for i in range(len(edges) - 1) if (indices == i).any()
]
train_rate = float(d["train_base_rate"])
test_rate = float(yte.mean())
brier = float(brier_score_loss(yte, p))
brier_constant_train = float(brier_score_loss(yte, np.full(len(yte), train_rate)))
hit = float(yte[top].sum() / yte.sum())
d["ranking_calibration"] = {
    "primary_model": primary,
    "top_10_percent_threshold": threshold,
    "selected_fraction": selection_fraction,
    "top_10_percent_hit_rate": hit,
    "hit_rate_lift": hit / selection_fraction,
    "mean_prediction": float(p.mean()),
    "observed_rate": test_rate,
    "mean_to_observed_ratio": float(p.mean() / test_rate),
    "min_prediction": float(p.min()),
    "max_prediction": float(p.max()),
    "brier_primary": brier,
    "brier_constant_train_rate": brier_constant_train,
    "brier_constant_test_rate": float(brier_score_loss(yte, np.full(len(yte), test_rate))),
    "brier_skill_vs_train_rate": 1 - brier / brier_constant_train,
    "reliability_bins": reliability,
}
for variable in ("slope", "ndvi"):
    quartile = np.asarray(pd.qcut(te[variable], 4, labels=False, duplicates="drop"))
    d["ranking_calibration"][variable + "_quartile_hit_rates"] = [
        float(yte[(quartile == i) & top].sum() / yte[quartile == i].sum())
        if yte[quartile == i].sum() else None
        for i in range(int(np.nanmax(quartile)) + 1)
    ]
np.savez_compressed(ROOT / "outputs/review_primary_predictions.npz", probability=p, label=yte)
out.write_text(json.dumps(d, indent=2))
print("Saved empirical primary test predictions and calibration audit")
