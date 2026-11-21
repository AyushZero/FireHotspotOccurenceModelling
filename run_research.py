"""Frozen candidate search. Never scores 2023 while selecting candidates."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
from pathlib import Path
import json, time, joblib
import numpy as np
from threadpoolctl import threadpool_limits
from sklearn.metrics import roc_auc_score, average_precision_score,brier_score_loss
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from src.dataset import build_dataset,split_by_year
from research_models import drivers,Features,PenalizedLogit

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs/research'; OUT.mkdir(exist_ok=True)
MODELS=ROOT/'models/research'; MODELS.mkdir(exist_ok=True)

def metrics(y,p):
    k=int(np.ceil(.1*len(y))); ix=np.argsort(-p,kind='stable')[:k]
    return {'auc':float(roc_auc_score(y,p)),'ap':float(average_precision_score(y,p)),'brier':float(brier_score_loss(y,p)),'precision_top10':float(y[ix].mean()),'recall_top10':float(y[ix].sum()/y.sum()),'mean':float(p.mean())}

def main():
    df=build_dataset(ROOT/'data/features'); tr,va,_=split_by_year(df,[2021],[2022],[2023])
    sel=va[va.time.dt.day<=14]; cal=va[va.time.dt.day>=22]
    xt,xv,xc=map(drivers,[tr,sel,cal]); yt,yv,yc=[f.ignition.to_numpy(int) for f in (tr,sel,cal)]
    specs=[]
    for kind in ['minmax','standard','loghistory']:
        for C in [.01,.1,1,10,100]:
            for positive in [False,True]:specs.append(dict(family='logistic',kind=kind,C=C,positive=positive))
    for kind in ['spline','interactions']:
        for C in [.01,.1,1,10]:specs.append(dict(family='logistic',kind=kind,C=C,positive=False))
    for epsilon in [0,.05,.2,.5,1]:specs.append(dict(family='logistic',kind='multiplicative',C=100,positive=False,epsilon=epsilon))
    for family in ['rf','xgb','lgbm']:
        for handling in ['none','weighted','smote']:
            for depth in [4,8]:specs.append(dict(family=family,kind='loghistory',handling=handling,depth=depth))
    (OUT/'candidate_specifications.json').write_text(json.dumps(specs,indent=2))
    records=[]
    for i,spec in enumerate(specs):
        name=f'candidate_{i:03d}'; cache=OUT/f'{name}.json'
        if cache.exists() and (MODELS/f'{name}.joblib').exists() and (OUT/f'{name}_validation.npz').exists():
            record=json.loads(cache.read_text())
            if record['spec']!=spec:raise RuntimeError('Cached specification differs; use a new experiment folder')
            records.append(record); print('Resume',name,flush=True);continue
        start=time.time(); tf=Features(spec['kind'],epsilon=spec.get('epsilon',0)).fit(xt)
        a,b,c=[tf.transform(x) for x in [xt,xv,xc]]; yfit=yt
        family=spec['family']
        if family=='logistic':model=PenalizedLogit(spec['C'],spec['positive'])
        else:
            handling=spec['handling']; weight=(len(yt)-yt.sum())/yt.sum() if handling=='weighted' else 1
            if handling=='smote':
                a,yfit=SMOTE(sampling_strategy=.1,random_state=42,k_neighbors=3).fit_resample(a,yt)
                a,yfit=RandomUnderSampler(sampling_strategy=.3,random_state=42).fit_resample(a,yfit)
            if family=='rf':model=RandomForestClassifier(n_estimators=250,max_depth=spec['depth'],min_samples_leaf=10,class_weight='balanced' if handling=='weighted' else None,n_jobs=4,random_state=42)
            elif family=='xgb':model=XGBClassifier(n_estimators=300,max_depth=spec['depth'],learning_rate=.05,min_child_weight=10,subsample=.8,colsample_bytree=.9,reg_lambda=5,scale_pos_weight=weight,tree_method='hist',n_jobs=4,random_state=42,eval_metric='logloss')
            else:model=LGBMClassifier(n_estimators=300,max_depth=spec['depth'],num_leaves=min(31,2**spec['depth']-1),learning_rate=.05,min_child_samples=40,reg_lambda=5,scale_pos_weight=weight,n_jobs=4,random_state=42,verbosity=-1)
        with threadpool_limits(limits=4):model.fit(a,yfit)
        pv=model.predict_proba(b)[:,1]; pc=model.predict_proba(c)[:,1]
        record={'id':name,'spec':spec,'selection':metrics(yv,pv),'seconds':time.time()-start}
        joblib.dump({'transform':tf,'model':model},MODELS/f'{name}.joblib')
        np.savez_compressed(OUT/f'{name}_validation.npz',selection=pv,calibration=pc)
        cache.write_text(json.dumps(record,indent=2)); records.append(record)
        (OUT/'leaderboard.json').write_text(json.dumps(records,indent=2))
        print(name,spec,'AP',round(record['selection']['ap'],5),'AUC',round(record['selection']['auc'],5),'seconds',round(record['seconds']),flush=True)
    best=max(records,key=lambda r:(r['selection']['ap'],r['selection']['auc']))
    groups={}
    for r in records:
        group=r['spec']['family'] if r['spec']['family']!='logistic' else r['spec']['kind']+('_positive' if r['spec']['positive'] else '_free')
        if group not in groups or r['selection']['ap']>groups[group]['selection']['ap']:groups[group]=r
    selected={'primary':best,'family_winners':groups,'n_candidates':len(records),'selection_n':len(sel),'selection_events':int(yv.sum()),'calibration_n':len(cal),'calibration_events':int(yc.sum())}
    (OUT/'leaderboard.json').write_text(json.dumps(records,indent=2))
    (OUT/'frozen_selection.json').write_text(json.dumps(selected,indent=2))
    print('SELECTION FROZEN',best['id'],best['spec'],flush=True)

if __name__=='__main__':main()
