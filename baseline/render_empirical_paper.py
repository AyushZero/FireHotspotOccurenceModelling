"""Fill the empirical IEEE paper exclusively from saved audited results."""
from pathlib import Path
import json

from jinja2 import Environment, FileSystemLoader
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
d = json.loads((OUT / "review_followup.json").read_text())
h = json.loads((OUT / "real_firms_history_audit.json").read_text())
c = json.loads((OUT / "spatial_coverage_audit.json").read_text())
f = json.loads((OUT / "empirical_figure_metrics.json").read_text())
m = json.loads((OUT / "multiplicative_followup.json").read_text())
p = np.load(OUT / "model_predictions_test.npz")
primary = d["selection"]["primary"]
assert f["primary"] == primary
test = d["models"][primary]["metrics"]["test"]
cal = d["ranking_calibration"]
assert abs(test["auc_roc"] - f["figure_auc"][primary]) < 1e-9
assert abs(test["auc_pr"] - f["figure_ap"][primary]) < 1e-9
assert abs(test["brier"] - cal["brier_primary"]) < 1e-9
assert len(p["label"]) == sum(p["label"] == 0) + sum(p["label"] == 1)

order = [("constrained_additive", "Constrained additive"),
         ("unconstrained_additive", "Unconstrained additive"),
         ("raw_logistic", "Raw standardized logistic"),
         ("climatology", "Cell and season climatology"),
         ("persistence", "Hotspot persistence"),
         ("multiplicative", "Multiplicative composite"),
         ("simplified_weather_index", "Simplified weather index"),
         ("random_forest", "Random Forest"),
         ("xgboost", "XGBoost"),
         ("lightgbm", "LightGBM")]
model_rows = []
for key, name in order:
    v = d["models"][key]["metrics"]["validation"]
    t = d["models"][key]["metrics"]["test"]
    if key == primary:
        name = "\\textbf{" + name + "}"
    model_rows.append(f"{name} & {v['auc_roc']:.3f} & {v['auc_pr']:.4f} & {v['brier']:.5f} & "
                      f"{t['auc_roc']:.3f} & {t['auc_pr']:.4f} & {t['brier']:.5f} \\\\")

coefficient_names = [("T", "$\\tilde T$"), ("NDVI", "$\\widetilde{\\mathrm{NDVI}}$"),
                     ("1-NDWI", "$1-\\widetilde{\\mathrm{NDWI}}$"),
                     ("1-RH", "$1-\\widetilde{\\mathrm{RH}}$"),
                     ("1-SM", "$1-\\widetilde{\\mathrm{SM}}$"),
                     ("U", "$\\tilde U$"), ("slope", "$\\tilde\\theta$"),
                     ("FRP_hist", "FRP history"), ("Count_hist", "Count history")]
fit = d["models"][primary]
coef_rows = []
for key, name in coefficient_names:
    a = fit["coefficients_minmax"][key]
    b = fit["coefficients_standardized"][key]
    coef_rows.append(f"{name} & {a:+.3f} & {b:+.3f} \\\\")
coef_rows.append(f"Intercept & {fit['intercept']:+.3f} & --- \\\\")

ab = [(key, row["delta"]) for key, row in d["ablation"][primary].items() if key != "full"]
positive = [kv for kv in ab if kv[1] > 0]
if positive:
    biggest = max(positive, key=lambda kv: kv[1])
    largest_drop = f"removing {biggest[0].replace('_', ' ')} (drop {biggest[1]:.3f})"
else:
    largest_drop = "none of the nine variables (every drop-one fit matched or improved test AUC)"

bins = cal["reliability_bins"]
max_gap = max(abs(row["mean_predicted"] - row["observed_rate"]) for row in bins)
reliability_sentence = f"The largest absolute observed--predicted difference across the bins is {max_gap:.4f}."
shap_top = max(f["shap_mean_abs"], key=f["shap_mean_abs"].get).replace("_", r"\_")
ctx = {
    "test_n": f"{len(p['label']):,}", "test_events": f"{int(p['label'].sum()):,}",
    "rate_pct": f"{100 * p['label'].mean():.2f}",
    "auc": f"{test['auc_roc']:.3f}", "ap": f"{test['auc_pr']:.4f}",
    "brier": f"{test['brier']:.5f}",
    "persistence_auc": f"{d['models']['persistence']['metrics']['test']['auc_roc']:.3f}",
    "persistence_ap": f"{d['models']['persistence']['metrics']['test']['auc_pr']:.4f}",
    "other_additive_auc": f"{d['models']['constrained_additive' if primary == 'unconstrained_additive' else 'unconstrained_additive']['metrics']['test']['auc_roc']:.3f}",
    "other_additive_ap": f"{d['models']['constrained_additive' if primary == 'unconstrained_additive' else 'unconstrained_additive']['metrics']['test']['auc_pr']:.4f}",
    "hit_pct": f"{100 * cal['top_10_percent_hit_rate']:.1f}",
    "cells": c["california_cells"],
    "n2021": f"{h['positive_cell_days']['2021']:,}",
    "n2022": f"{h['positive_cell_days']['2022']:,}",
    "n2023": f"{h['positive_cell_days']['2023']:,}",
    "rate2021": f"{100*h['base_rate']['2021']:.2f}",
    "rate2022": f"{100*h['base_rate']['2022']:.2f}",
    "rate2023": f"{100*h['base_rate']['2023']:.2f}",
    "persist_cond_pct": f"{100*h['p_event_given_yesterday']['2023']:.1f}",
    "selected_name": "the sign-constrained fit" if primary == "constrained_additive" else "the unconstrained fit",
    "test_rate": f"{p['label'].mean():.4f}",
    "model_rows": "\n".join(model_rows),
    "coef_rows": "\n".join(coef_rows),
    "coefficient_caution": (
        "The selected unconstrained fit assigns negative coefficients to "
        + ", ".join(key.replace("_", r"\_") for key, value in fit["coefficients_minmax"].items() if value < -0.001)
        + "; these conflict with the imposed directions in the constrained alternative and should not be read as physical effects."
        if primary == "unconstrained_additive" else
        "All selected fitted slopes obey the imposed nonnegative restriction; that alone does not establish a physical effect."
    ),
    "weights": ", ".join(f"{x:.2f}" for x in m["winner"]),
    "mult_val_auc": f"{m['winner_val_auc']:.3f}",
    "mult_test_auc": f"{m['test_auc']:.3f}",
    "largest_drop": largest_drop.replace("_", r"\_"),
    "mean_prob": f"{cal['mean_prediction']:.4f}",
    "mean_ratio": f"{cal['mean_to_observed_ratio']:.2f}",
    "brier_train": f"{cal['brier_constant_train_rate']:.5f}",
    "brier_test": f"{cal['brier_constant_test_rate']:.5f}",
    "bss": f"{cal['brier_skill_vs_train_rate']:.4f}",
    "reliability_sentence": reliability_sentence,
    "threshold": f"{cal['top_10_percent_threshold']:.4f}",
    "selected_pct": f"{100*cal['selected_fraction']:.1f}",
    "lift": f"{cal['hit_rate_lift']:.2f}",
    "shap_top": shap_top,
    "persistence_comparison": (f"{abs(test['auc_roc']-d['models']['persistence']['metrics']['test']['auc_roc']):.3f} "
                               + ("higher than" if test['auc_roc'] > d['models']['persistence']['metrics']['test']['auc_roc'] else "lower than")),
    "weather_auc": f"{d['models']['weather_only']['metrics']['test']['auc_roc']:.3f}",
    "weather_sm_auc": f"{d['models']['weather_plus_sm']['metrics']['test']['auc_roc']:.3f}",
    "slope_quartile_hits": ", ".join(f"{100*x:.1f}" for x in cal["slope_quartile_hit_rates"]),
    "ndvi_quartile_hits": ", ".join(f"{100*x:.1f}" for x in cal["ndvi_quartile_hit_rates"]),
    "rf_ci": " to ".join(f"{x:+.3f}" for x in d["paired_bootstrap"]["difference_primary_minus_comparator"]["random_forest"]["ci95"]),
}
best_key, best_name = max(order, key=lambda kv: d["models"][kv[0]]["metrics"]["test"]["auc_roc"])
ctx["best_model_name"] = best_name
ctx["best_model_auc"] = f"{d['models'][best_key]['metrics']['test']['auc_roc']:.3f}"
ctx["best_model_ap"] = f"{d['models'][best_key]['metrics']['test']['auc_pr']:.4f}"
env = Environment(loader=FileSystemLoader(str(ROOT)),
                  variable_start_string="<<", variable_end_string=">>", autoescape=False)
tex = env.get_template("paper_empirical_template.tex.j2").render(**ctx)
assert "<<" not in tex and "synthetic" not in tex.lower()
(ROOT / "18-11_empirical.tex").write_text(tex, encoding="utf-8")
print("Wrote 18-11_empirical.tex from audited empirical outputs")
