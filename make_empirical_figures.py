"""Draw publication figures only from the finished empirical model outputs."""
from pathlib import Path
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
d = json.loads((OUT / "review_followup.json").read_text())
m = json.loads((OUT / "multiplicative_followup.json").read_text())
p = np.load(OUT / "model_predictions_test.npz")
y = p["label"]
primary = d["selection"]["primary"]
names = [(primary, "Selected additive"), ("multiplicative", "Multiplicative"),
         ("simplified_weather_index", "Weather index"),
         ("climatology", "Climatology"), ("persistence", "Persistence"),
         ("random_forest", "Random forest"), ("xgboost", "XGBoost"),
         ("lightgbm", "LightGBM")]
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
                     "legend.fontsize": 7, "savefig.dpi": 300})

fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))
colors = plt.get_cmap("tab10").colors
for i, (key, label) in enumerate(names):
    prob = p[key]
    fpr, tpr, _ = roc_curve(y, prob)
    precision, recall, _ = precision_recall_curve(y, prob)
    axes[0].plot(fpr, tpr, color=colors[i], lw=1.35,
                 label=f"{label}: {roc_auc_score(y, prob):.3f}")
    axes[1].plot(recall, precision, color=colors[i], lw=1.35,
                 label=f"{label}: {average_precision_score(y, prob):.3f}")
axes[0].plot([0, 1], [0, 1], color="0.45", ls=":", lw=.8)
axes[1].axhline(y.mean(), color="0.45", ls=":", lw=.8, label=f"Event rate: {y.mean():.3f}")
axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="2023 ROC")
axes[1].set(xlabel="Recall", ylabel="Precision", title="2023 precision–recall")
axes[0].legend(loc="lower right", framealpha=.95)
axes[1].legend(loc="upper right", framealpha=.95)
for ax in axes:
    ax.grid(alpha=.15)
fig.tight_layout()
fig.savefig(OUT / "empirical_discrimination.png", bbox_inches="tight")
plt.close(fig)

def ablation(path, rows, title, full):
    items = [(key, value["delta"]) for key, value in rows.items() if key != "full"]
    items.sort(key=lambda kv: kv[1])
    labels = [key.replace("_hist", " history").replace("_", " ") for key, _ in items]
    delta = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(4.5, max(2.8, .35 * len(items) + .9)))
    ax.barh(labels, delta, color=["#b35b45" if x >= 0 else "#467aa0" for x in delta])
    ax.axvline(0, color="0.2", lw=.8)
    ax.set_xlabel(f"Test AUC drop from full ({full:.3f})")
    ax.set_title(title)
    margin = max(.015, max(abs(x) for x in delta) * .25)
    ax.set_xlim(min(0, min(delta)) - margin, max(0, max(delta)) + margin)
    for i, v in enumerate(delta):
        ax.text(v + (margin * .08 if v >= 0 else -margin * .08), i,
                f"{v:+.4f}", ha="left" if v >= 0 else "right", va="center", fontsize=8)
    ax.grid(axis="x", alpha=.15)
    fig.tight_layout()
    fig.savefig(OUT / path, bbox_inches="tight")
    plt.close(fig)

ablation("empirical_additive_ablation.png", d["ablation"][primary],
         "Additive drop-one ablation", d["models"][primary]["metrics"]["test"]["auc_roc"])
ablation("empirical_multiplicative_ablation.png", m["factor_ablation"],
         "Multiplicative factor ablation", m["test_auc"])

prob = p[primary]
bins = np.quantile(prob, np.linspace(0, 1, 11))
bins = np.unique(bins)
ix = np.clip(np.searchsorted(bins[1:-1], prob, side="right"), 0, len(bins) - 2)
mean_p = np.array([prob[ix == j].mean() for j in range(len(bins) - 1)])
obs_p = np.array([y[ix == j].mean() for j in range(len(bins) - 1)])
count = np.array([(ix == j).sum() for j in range(len(bins) - 1)])
fig, ax = plt.subplots(figsize=(4.4, 3.7))
ax.plot([0, max(mean_p.max(), obs_p.max()) * 1.05],
        [0, max(mean_p.max(), obs_p.max()) * 1.05], ":", color="0.5", label="Perfect calibration")
ax.plot(mean_p, obs_p, "o-", color="#27637d", label="Equal-count bins")
ax.set(xlabel="Mean predicted probability", ylabel="Observed hotspot frequency",
       title="2023 reliability, additive model")
ax.legend(loc="upper left")
ax.grid(alpha=.15)
fig.tight_layout()
fig.savefig(OUT / "empirical_reliability.png", bbox_inches="tight")
plt.close(fig)

# Tree attribution uses the fitted empirical XGBoost model and model-ready 2023 cells.
import shap
from src.dataset import build_dataset, split_by_year
frame = build_dataset(ROOT / "data/features")
_, _, test = split_by_year(frame, [2021], [2022], [2023])
cols = ["t_max", "rh_min", "u10_max", "sm_top", "ndvi", "ndwi", "slope", "frp_hist", "count_hist"]
rng = np.random.default_rng(42)
sample = test.iloc[rng.choice(len(test), size=min(3000, len(test)), replace=False)]
model = joblib.load(ROOT / "models/xgboost_empirical.joblib")
sv = shap.TreeExplainer(model).shap_values(sample[cols].to_numpy())
if isinstance(sv, list):
    sv = sv[1]
importance = np.mean(np.abs(np.asarray(sv)), axis=0)
order = np.argsort(importance)
fig, ax = plt.subplots(figsize=(4.6, 3.5))
ax.barh(np.asarray(cols)[order], importance[order], color="#467aa0")
ax.set(xlabel="Mean absolute SHAP value (log odds)", title="XGBoost attribution, 2023 sample")
ax.grid(axis="x", alpha=.15)
fig.tight_layout()
fig.savefig(OUT / "empirical_shap.png", bbox_inches="tight")
plt.close(fig)

report = {"primary": primary, "figure_auc": {key: float(roc_auc_score(y, p[key])) for key, _ in names},
          "figure_ap": {key: float(average_precision_score(y, p[key])) for key, _ in names},
          "reliability_bins": [{"mean_predicted": float(a), "observed": float(b), "n": int(c)}
                               for a, b, c in zip(mean_p, obs_p, count)],
          "shap_mean_abs": dict(zip(cols, map(float, importance)))}
(OUT / "empirical_figure_metrics.json").write_text(json.dumps(report, indent=2))
print("Wrote five empirical figures", flush=True)
