"""Recompute pre-submission comparisons on the corrected 2021/2022/2023 split."""
from pathlib import Path
import json
import warnings
import joblib
import numpy as np
import pandas as pd
import yaml
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from src.dataset import build_dataset, split_by_year
from src.models import train_random_forest, train_xgboost, train_lightgbm
from src.preprocess import process_firms

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
df = build_dataset(ROOT / "data/features")
tr, va, te = split_by_year(df, [2021], [2022], [2023])
splits = {"train": tr, "validation": va, "test": te}
y = {k: v.ignition.to_numpy(int) for k,v in splits.items()}
names = ["T", "NDVI", "1-NDWI", "1-RH", "1-SM", "U", "slope", "FRP_hist", "Count_hist"]
raw_names = ["t_max", "ndvi", "ndwi", "rh_min", "sm_top", "u10_max", "slope", "frp_hist", "count_hist"]

def phys(frame):
    return np.column_stack([frame.t_max, frame.ndvi, 1-frame.ndwi,
                            1-frame.rh_min, 1-frame.sm_top, frame.u10_max,
                            frame.slope, frame.frp_hist, frame.count_hist]).astype(np.float64)
X = {k: phys(v) for k,v in splits.items()}
R = {k: v[raw_names].to_numpy(np.float64) for k,v in splits.items()}
assert all(np.isfinite(a).all() for a in [*X.values(), *R.values()]), "Nonfinite model inputs"
result = {"source_script": "review_followup.py", "data": "data/features/features.nc", "models": {}, "ablation": {}}

def scores(probs, key):
    return {split: {"auc_roc": float(roc_auc_score(y[split], p)),
                    "auc_pr": float(average_precision_score(y[split], p)),
                    "brier": float(brier_score_loss(y[split], p)),
                    "mean_probability": float(np.mean(p)), "min_probability": float(np.min(p)),
                    "max_probability": float(np.max(p))}
            for split,p in probs.items()}

def fit_constrained(Xtrain, ytrain):
    n = Xtrain.shape[1]
    def fun(p):
        z = Xtrain @ p[:-1] + p[-1]
        return float(np.mean(np.logaddexp(0, z) - ytrain*z))
    def jac(p):
        err = expit(Xtrain @ p[:-1] + p[-1]) - ytrain
        return np.r_[Xtrain.T @ err / len(ytrain), np.mean(err)]
    fit = minimize(fun, np.r_[np.zeros(n), -5.8], jac=jac,
                   method="L-BFGS-B", bounds=[(0,None)]*n+[(None,None)],
                   options={"maxiter":5000,"ftol":1e-12})
    assert fit.success or fit.status == 2, fit.message
    return fit.x

print("Fitting additive variants", flush=True)
con = fit_constrained(X["train"], y["train"])
unc = LogisticRegression(C=1, solver="lbfgs", max_iter=2000).fit(X["train"], y["train"])
models = {"constrained_additive": con, "unconstrained_additive": unc}
pred = {}
pred["constrained_additive"] = {k: expit(a @ con[:-1] + con[-1]) for k,a in X.items() if k != "train"}
pred["unconstrained_additive"] = {k: unc.predict_proba(a)[:,1] for k,a in X.items() if k != "train"}
for key in models:
    coefs = con[:-1] if key == "constrained_additive" else unc.coef_[0]
    bias = con[-1] if key == "constrained_additive" else unc.intercept_[0]
    result["models"][key] = {"metrics": scores(pred[key], key),
                              "coefficients_minmax": dict(zip(names, map(float, coefs))),
                              "coefficients_standardized": dict(zip(names, map(float, coefs*X["train"].std(axis=0)))),
                              "intercept": float(bias)}
primary = max(models, key=lambda k: result["models"][k]["metrics"]["validation"]["auc_roc"])
result["selection"] = {"criterion": "higher 2022 validation AUC-ROC", "primary": primary}
print("Selected", primary, flush=True)

# Raw standardized features and standardized physics features are affine
# recodings. Their predictions must agree to numerical optimization tolerance.
scaler = StandardScaler().fit(R["train"])
raw_std = {k: scaler.transform(a) for k,a in R.items()}
raw_lr = LogisticRegression(C=1, solver="lbfgs", max_iter=2000).fit(raw_std["train"], y["train"])
pred["raw_logistic"] = {k: raw_lr.predict_proba(raw_std[k])[:,1] for k in ("validation","test")}
result["models"]["raw_logistic"] = {"metrics":scores(pred["raw_logistic"], "raw_logistic"), "features":"nine raw drivers, training-year standardization"}
phys_scaler = StandardScaler().fit(X["train"])
phys_std = {k: phys_scaler.transform(a) for k,a in X.items()}
phys_lr = LogisticRegression(C=1, solver="lbfgs", max_iter=2000).fit(phys_std["train"], y["train"])
result["affine_equivalence_max_abs_probability_difference"] = float(np.max(np.abs(pred["raw_logistic"]["test"] - phys_lr.predict_proba(phys_std["test"])[:,1])))

# Training-only cell climatology with annual harmonic terms.
cell_rate = tr.groupby(["latitude","longitude"]).ignition.mean()
def climatology_features(frame):
    doy = pd.to_datetime(frame.time).dt.dayofyear.to_numpy()
    keys = pd.MultiIndex.from_frame(frame[["latitude","longitude"]])
    rate = cell_rate.reindex(keys).fillna(y["train"].mean()).to_numpy()
    return np.column_stack([np.sin(2*np.pi*doy/365.25), np.cos(2*np.pi*doy/365.25), rate])
C = {k: climatology_features(v) for k,v in splits.items()}
clim = LogisticRegression(C=1,solver="lbfgs",max_iter=2000).fit(C["train"],y["train"])
pred["climatology"] = {k: clim.predict_proba(C[k])[:,1] for k in ("validation","test")}
result["models"]["climatology"] = {"metrics":scores(pred["climatology"],"climatology"),"features":"sin/cos day of year and training-only per-cell hotspot rate"}

print("Building lagged persistence", flush=True)
firms = process_firms((ROOT / cfg["paths"]["raw_data"]).resolve(), cfg)
truth = firms.ignition.to_numpy().astype(np.int8)
lag1 = np.zeros_like(truth)
lag1[1:] = truth[:-1]
prev7 = np.zeros_like(truth)
for k in range(1,8):
    prev7[k:] = np.maximum(prev7[k:],truth[:-k])
def persistence_features(frame):
    day = (pd.to_datetime(frame.time)-pd.Timestamp(cfg["time"]["start"])).dt.days.to_numpy()
    ilat = np.searchsorted(firms.latitude.values,frame.latitude.to_numpy())
    ilon = np.searchsorted(firms.longitude.values,frame.longitude.to_numpy())
    assert np.array_equal(firms.latitude.values[ilat],frame.latitude.to_numpy())
    assert np.array_equal(firms.longitude.values[ilon],frame.longitude.to_numpy())
    return np.column_stack([lag1[day,ilat,ilon],prev7[day,ilat,ilon]])
P = {k:persistence_features(v) for k,v in splits.items()}
pers = LogisticRegression(C=1,solver="lbfgs",max_iter=2000).fit(P["train"],y["train"])
pred["persistence"] = {k:pers.predict_proba(P[k])[:,1] for k in ("validation","test")}
result["models"]["persistence"] = {"metrics":scores(pred["persistence"],"persistence"),"features":"same-cell prior-day and any prior-seven-day hotspot"}
result["history_check"] = {}
for k in ("train","test"):
    previous = P[k][:,0] == 1
    result["history_check"][k] = {"base_rate":float(y[k].mean()),
                                    "p_event_given_yesterday":float(y[k][previous].mean()),
                                    "n_yesterday_positive":int(previous.sum()),
                                    "frp_count_feature_correlation":float(np.corrcoef(R[k][:,-2],R[k][:,-1])[0,1]),
                                    "frp_count_identical":bool(np.array_equal(R[k][:,-2],R[k][:,-1]))}

# Six validation-selected multiplicative configurations, with a fresh 2021 fit.
def risk(frame, weights):
    w1,w2,w3,b1,b2 = weights
    dry = w1*(1-frame.ndwi.to_numpy())+w2*(1-frame.rh_min.to_numpy())+w3*(1-frame.sm_top.to_numpy())
    return frame.t_max.to_numpy()*frame.ndvi.to_numpy()*dry*(1+b1*frame.u10_max.to_numpy()+b2*frame.slope.to_numpy())+.6*frame.frp_hist.to_numpy()+.4*frame.count_hist.to_numpy()
choices = [(.33,.33,.33,.5,.5),(.5,.25,.25,.7,.3),(.4,.3,.3,.5,.5),(.4,.3,.3,.6,.4),(.3,.4,.3,.6,.4),(.4,.2,.4,.6,.4)]
mult_fits=[]
for weights in choices:
    a = risk(tr,weights)
    mu,sd = a.mean(),a.std()
    m = LogisticRegression(penalty=None,solver="lbfgs",max_iter=5000).fit(((a-mu)/sd).reshape(-1,1),y["train"])
    pv = m.predict_proba(((risk(va,weights)-mu)/sd).reshape(-1,1))[:,1]
    mult_fits.append((roc_auc_score(y["validation"],pv),weights,m,mu,sd,pv))
best = max(mult_fits,key=lambda row:row[0])
pred["multiplicative"] = {"validation":best[5],"test":best[2].predict_proba(((risk(te,best[1])-best[3])/best[4]).reshape(-1,1))[:,1]}
result["models"]["multiplicative"] = {"metrics":scores(pred["multiplicative"],"multiplicative"),"weights":best[1],"selection":"highest validation AUC of six configurations"}

def weather_score(frame):
    return frame.t_max.to_numpy()*(1-frame.rh_min.to_numpy())*(1+frame.u10_max.to_numpy())
a = weather_score(tr)
lo,hi = a.min(),a.max()
simple = {k:np.clip((weather_score(v)-lo)/(hi-lo+1e-8),0,1).reshape(-1,1) for k,v in splits.items()}
simple_lr = LogisticRegression(max_iter=2000).fit(simple["train"],y["train"])
pred["simplified_weather_index"] = {k:simple_lr.predict_proba(simple[k])[:,1] for k in ("validation","test")}
result["models"]["simplified_weather_index"] = {"metrics":scores(pred["simplified_weather_index"],"simplified_weather_index")}

print("Fitting resampled trees", flush=True)
tree_columns = ["t_max","rh_min","u10_max","sm_top","ndvi","ndwi","slope","frp_hist","count_hist"]
T = {k:v[tree_columns].to_numpy(np.float64) for k,v in splits.items()}
sampler = ImbPipeline([("smote",SMOTE(sampling_strategy=.10,random_state=42,k_neighbors=3)),
                       ("under",RandomUnderSampler(sampling_strategy=.3,random_state=42))])
T_res,y_res = sampler.fit_resample(T["train"],y["train"])
for key,fn in (("random_forest",train_random_forest),("xgboost",train_xgboost),("lightgbm",train_lightgbm)):
    model = fn(T_res,y_res)
    validation_raw = model.predict_proba(T["validation"])[:,1]
    test_raw = model.predict_proba(T["test"])[:,1]
    platt = LogisticRegression(max_iter=2000).fit(validation_raw.reshape(-1,1),y["validation"])
    model_dir = ROOT / "models"
    model_dir.mkdir(exist_ok=True)
    joblib.dump(model, model_dir / f"{key}_empirical.joblib")
    joblib.dump(platt, model_dir / f"{key}_platt_empirical.joblib")
    pred[key] = {"validation":platt.predict_proba(validation_raw.reshape(-1,1))[:,1],
                 "test":platt.predict_proba(test_raw.reshape(-1,1))[:,1]}
    result["models"][key] = {"metrics":scores(pred[key],key),"calibration":"Platt logistic fit on 2022 scores/labels; validation Brier in-sample","n_estimators":200}
    print("Finished",key,flush=True)

# Refit each drop-one model on 2021. The constrained model has nonnegative
# coefficients; the unconstrained model uses the same sklearn settings as above.
print("Running drop-one ablations",flush=True)
for key in ("constrained_additive","unconstrained_additive"):
    full_auc = result["models"][key]["metrics"]["test"]["auc_roc"]
    rows = {"full":{"auc_roc":full_auc,"delta":0.0}}
    for j,name in enumerate(names):
        subset = [i for i in range(len(names)) if i != j]
        if key == "constrained_additive":
            pars = fit_constrained(X["train"][:,subset],y["train"])
            p = expit(X["test"][:,subset] @ pars[:-1] + pars[-1])
        else:
            fit = LogisticRegression(C=1,solver="lbfgs",max_iter=2000).fit(X["train"][:,subset],y["train"])
            p = fit.predict_proba(X["test"][:,subset])[:,1]
        auc = float(roc_auc_score(y["test"],p))
        rows[name] = {"auc_roc":auc,"delta":full_auc-auc}
    result["ablation"][key] = rows
    print("Finished ablation",key,flush=True)

# Day-block paired bootstrap: all 2023 events and a fixed 5% non-event sample.
print("Running paired bootstrap",flush=True)
base_y = y["test"]
rng = np.random.default_rng(20261120)
event = np.flatnonzero(base_y == 1)
nonevent = np.flatnonzero(base_y == 0)
rows = np.sort(np.r_[event,rng.choice(nonevent,size=round(.05*len(nonevent)),replace=False)])
days = pd.factorize(te.iloc[rows].time)[0]
day_rows = [np.flatnonzero(days==i) for i in range(days.max()+1)]
rng = np.random.default_rng(42)
comparators = ["raw_logistic","climatology","persistence","random_forest","xgboost","lightgbm"]
diffs = {key:[] for key in comparators}
for b in range(500):
    picked = rng.integers(0,len(day_rows),len(day_rows))
    ix = np.concatenate([day_rows[i] for i in picked])
    yy = base_y[rows[ix]]
    if len(np.unique(yy)) < 2:
        continue
    primary_auc = roc_auc_score(yy,pred[primary]["test"][rows[ix]])
    for key in comparators:
        diffs[key].append(primary_auc-roc_auc_score(yy,pred[key]["test"][rows[ix]]))
result["paired_bootstrap"] = {"settings":{"days":365,"resamples":500,"non_event_fraction":.05,"all_events":True,"seed_non_event":20261120,"seed_blocks":42},
                              "difference_primary_minus_comparator":{key:{"point":result["models"][primary]["metrics"]["test"]["auc_roc"]-result["models"][key]["metrics"]["test"]["auc_roc"],
                                                                               "ci95":list(map(float,np.quantile(values,[.025,.975])))} for key,values in diffs.items()}}
result["test_base_rate"] = float(y["test"].mean())
result["train_base_rate"] = float(y["train"].mean())
(OUT/"review_followup.json").write_text(json.dumps(result,indent=2))
np.savez_compressed(OUT/"model_predictions_test.npz", label=y["test"],
                    **{key: values["test"] for key, values in pred.items()})
print("Wrote outputs/review_followup.json",flush=True)
