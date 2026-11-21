import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from src.dataset import build_dataset, split_by_year

ROOT=Path(__file__).resolve().parent; OUT=ROOT/'outputs'
df=build_dataset(ROOT/'data/features'); tr,va,te=split_by_year(df,[2021],[2022],[2023])
def risk(d,w,b,g,off=None):
 t=d.t_max.to_numpy(); fa=d.ndvi.to_numpy(); fd=w[0]*(1-d.ndwi.to_numpy())+w[1]*(1-d.rh_min.to_numpy())+w[2]*(1-d.sm_top.to_numpy())
 ws=1+b[0]*d.u10_max.to_numpy()+b[1]*d.slope.to_numpy(); h=g[0]*d.frp_hist.to_numpy()+g[1]*d.count_hist.to_numpy()
 if off=='T': t=np.ones_like(t)
 if off=='F_avail': fa=np.ones_like(fa)
 if off=='F_dry': fd=np.ones_like(fd)
 if off=='W_slope': ws=np.ones_like(ws)
 if off=='A_recent': h=np.zeros_like(h)
 return (t*fa*fd*ws+h).reshape(-1,1)
cfgs=[(.33,.33,.33,.5,.5),(.5,.25,.25,.7,.3),(.4,.3,.3,.5,.5),(.4,.3,.3,.6,.4),(.3,.4,.3,.6,.4),(.4,.2,.4,.6,.4)]
ytr=tr.ignition.to_numpy(int); yva=va.ignition.to_numpy(int); yte=te.ignition.to_numpy(int)
def score(c,split='val',off=None):
 w=c[:3]; b=c[3:]; g=(.6,.4); rtr=risk(tr,w,b,g,off).astype(np.float64); d=va if split=='val' else te; y=yva if split=='val' else yte; rtest=risk(d,w,b,g,off).astype(np.float64)
 mean,sd=rtr.mean(),rtr.std(); model=LogisticRegression(penalty=None,solver='lbfgs',max_iter=5000).fit((rtr-mean)/sd,ytr)
 prob=model.predict_proba((rtest-mean)/sd)[:,1]; raw=roc_auc_score(y,rtest); logistic=roc_auc_score(y,prob)
 return raw,logistic,float(model.coef_[0,0])
vals=[score(c)[1] for c in cfgs]; winner=cfgs[int(np.argmax(vals))]
raw_full,full,slope=score(winner,'test')
ab={k:score(winner,'test',k)[1] for k in ['T','F_avail','F_dry','W_slope','A_recent']}
r={'six_configurations':[{'weights':c,'val_auc':v} for c,v in zip(cfgs,vals)],'winner':winner,'winner_val_auc':max(vals),'test_auc':full,'test_raw_score_auc':raw_full,'logistic_slope':slope,'factor_ablation':{k:{'auc':v,'delta':full-v} for k,v in ab.items()}}
(OUT/'multiplicative_followup.json').write_text(json.dumps(r,indent=2)); print(json.dumps(r,indent=2))
