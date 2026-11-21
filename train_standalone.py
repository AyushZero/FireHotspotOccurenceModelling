"""Fit declared candidates on 2021--2022; select only on 2023."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
from pathlib import Path
import json,time,joblib,hashlib,argparse
import numpy as np
from threadpoolctl import threadpool_limits
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from research_models import Features,PenalizedLogit,drivers
from standalone_models import JointIndex,SmoothLogit,spline_penalty,Ensemble,IdentityFeatures
from build_standalone_data import load_frames
from evaluation_utils import metrics

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'outputs/standalone';MODELS=ROOT/'models/standalone'

def specifications():
    specs=[]
    for kind in ['standard','loghistory']:
        for C in [.1,1,10]:
            for positive in [False,True]:specs.append(dict(family='linear',kind=kind,C=C,positive=positive))
    for knots in [3,5,7]:
        for C in [.1,1,10]:
            for alpha in [0,.01,.1]:specs.append(dict(family='smooth_additive',kind='spline',knots=knots,C=C,alpha=alpha))
    rng=np.random.default_rng(20261120)
    for i in range(96):
        params=dict(weights=rng.dirichlet([1,1,1]).tolist(),wind=float(rng.uniform(0,2)),slope=float(rng.uniform(0,2)),offset=float(rng.choice([0,.03,.1,.3,.7])),history=float(rng.choice([.1,.3,1,3,10])),mix=float(rng.uniform(0,1)),power=float(rng.choice([.5,1,2])))
        if i==0:params=dict(weights=[.3,.4,.3],wind=.6,slope=.4,offset=0,history=1,mix=.6,power=1)
        specs.append(dict(family='joint_index',parameters=params))
    for depth in [3,5,7]:
        for trees in [200,500,800]:
            for regularization in [1,10]:specs.append(dict(family='xgb',depth=depth,trees=trees,regularization=regularization,handling='none'))
    for leaves in [15,31,63]:
        for trees in [300,600]:
            for regularization in [1,10]:specs.append(dict(family='lgbm',leaves=leaves,trees=trees,regularization=regularization,handling='none'))
    for depth in [8,12,None]:
        for leaf in [5,20]:specs.append(dict(family='rf',depth=depth,leaf=leaf,trees=300,handling='none'))
    for family in ['rf','xgb','lgbm']:
        for handling in ['none','weighted','smote']:specs.append(dict(family=family,depth=5,leaves=31,leaf=10,trees=300,regularization=5,handling=handling))
    # Matched history-transform comparison for every joint-index configuration.
    specs.extend([dict(family='joint_index',parameters={**s['parameters'],'history_log':True}) for s in list(specs) if s['family']=='joint_index'])
    return specs

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--start',type=int,default=0);parser.add_argument('--stop',type=int);parser.add_argument('--fit-only',action='store_true');args=parser.parse_args()
    OUT.mkdir(exist_ok=True);MODELS.mkdir(exist_ok=True)
    train,val,_=load_frames();selection=val[val.time.dt.day<=14];calibration=val[val.time.dt.day>=22]
    x,xv,xc=map(drivers,[train,selection,calibration]);y=train.ignition.to_numpy(int);yv=selection.ignition.to_numpy(int)
    specs=specifications();protocol={'train_years':[2021,2022],'selection_year':2023,'calibration_year':2023,'test_year':2024,'selection_days':'1-14 monthly','calibration_days':'22-end monthly','criterion':'pooled selection AP; AUC tie break','candidate_specs':specs,'ensemble':'128 Dirichlet convex blends of four family winners, plus four vertices','seed':20261120,'features_sha256':hashlib.sha256((ROOT/'data/standalone/features.nc').read_bytes()).hexdigest()}
    ppath=OUT/'protocol.json'
    if ppath.exists():assert json.loads(ppath.read_text())==protocol,'Changed protocol/data: use a fresh experiment'
    ppath.write_text(json.dumps(protocol,indent=2));records=[]
    for index,spec in enumerate(specs):
        if index<args.start or (args.stop is not None and index>=args.stop):continue
        key=f'model_{index:03d}';cache=OUT/f'{key}.json'
        if cache.exists() and (MODELS/f'{key}.joblib').exists():
            record=json.loads(cache.read_text());assert record['spec']==spec,'Cached candidate differs'
            records.append(record);print('Resume',key,flush=True);continue
        start=time.time();family=spec['family']
        if family=='joint_index':tf=JointIndex(**spec['parameters']).fit(x)
        else:tf=Features(spec.get('kind','loghistory'),knots=spec.get('knots',4)).fit(x)
        a,b,c=map(tf.transform,[x,xv,xc]);yf=y
        if family=='linear':model=PenalizedLogit(spec['C'],spec['positive'])
        elif family=='smooth_additive':model=SmoothLogit(spec['C'],spec['alpha'],spline_penalty(tf),tf.scaler.scale_)
        elif family=='joint_index':model=LogisticRegression(C=100,max_iter=2000,tol=1e-7)
        else:
            handling=spec['handling'];weight=(len(y)-y.sum())/y.sum() if handling=='weighted' else 1
            if handling=='smote':
                a,yf=SMOTE(sampling_strategy=.1,k_neighbors=3,random_state=42).fit_resample(a,y)
                a,yf=RandomUnderSampler(sampling_strategy=.3,random_state=42).fit_resample(a,yf)
            if family=='rf':model=RandomForestClassifier(n_estimators=spec['trees'],max_depth=spec['depth'],min_samples_leaf=spec['leaf'],class_weight='balanced' if handling=='weighted' else None,n_jobs=4,random_state=42)
            elif family=='xgb':model=XGBClassifier(n_estimators=spec['trees'],max_depth=spec['depth'],learning_rate=.04,min_child_weight=10,reg_lambda=spec['regularization'],subsample=.8,colsample_bytree=.9,scale_pos_weight=weight,tree_method='hist',n_jobs=4,random_state=42,eval_metric='logloss')
            else:model=LGBMClassifier(n_estimators=spec['trees'],num_leaves=spec['leaves'],learning_rate=.04,min_child_samples=40,reg_lambda=spec['regularization'],scale_pos_weight=weight,n_jobs=4,random_state=42,verbosity=-1)
        with threadpool_limits(limits=4):model.fit(a,yf)
        pv=model.predict_proba(b)[:,1];pc=model.predict_proba(c)[:,1]
        quarter={str(q):metrics(yv[selection.time.dt.quarter.to_numpy()==q],pv[selection.time.dt.quarter.to_numpy()==q]) for q in range(1,5)}
        record={'id':key,'spec':spec,'selection':metrics(yv,pv),'selection_quarters':quarter,'seconds':time.time()-start}
        joblib.dump({'transform':tf,'model':model},MODELS/f'{key}.joblib');np.savez_compressed(OUT/f'{key}_validation.npz',selection=pv,calibration=pc)
        cache.write_text(json.dumps(record,indent=2));records.append(record);(OUT/'leaderboard.json').write_text(json.dumps(records,indent=2))
        print(key,family,'AP',round(record['selection']['ap'],5),'AUC',round(record['selection']['auc'],5),'seconds',round(record['seconds']),flush=True)
    if args.fit_only:
        (OUT/f'part_{args.start}_{args.stop}.done').write_text('complete')
        print('Candidate partition complete',args.start,args.stop,flush=True);return
    assert args.start==0 and args.stop is None,'Selection requires all candidates'
    winners={}
    for r in records:
        family=r['spec']['family']
        if family not in winners or (r['selection']['ap'],r['selection']['auc'])>(winners[family]['selection']['ap'],winners[family]['selection']['auc']):winners[family]=r
    families=['smooth_additive','rf','xgb','lgbm'];members=[winners[k]['id'] for k in families]
    vp=np.column_stack([np.load(OUT/f'{k}_validation.npz')['selection'] for k in members]);cp=np.column_stack([np.load(OUT/f'{k}_validation.npz')['calibration'] for k in members])
    rng=np.random.default_rng(20261120);weights=np.vstack([np.eye(4),rng.dirichlet(np.ones(4),size=128)])
    blends=[]
    for i,w in enumerate(weights):blends.append({'index':i,'weights':w.tolist(),'selection':metrics(yv,vp@w)})
    best=max(blends,key=lambda r:(r['selection']['ap'],r['selection']['auc']));w=np.array(best['weights'])
    record={'id':'ensemble','spec':{'family':'ensemble','members':members,'weights':w.tolist()},'selection':best['selection']}
    joblib.dump({'transform':IdentityFeatures(),'model':Ensemble([joblib.load(MODELS/f'{k}.joblib') for k in members],w)},MODELS/'ensemble.joblib')
    np.savez_compressed(OUT/'ensemble_validation.npz',selection=vp@w,calibration=cp@w);winners['ensemble']=record
    (OUT/'ensemble_search.json').write_text(json.dumps(blends,indent=2))
    primary=max([*records,record],key=lambda r:(r['selection']['ap'],r['selection']['auc']))
    frozen={'primary':primary,'family_winners':winners,'n_candidates':len(records),'n_blends':len(blends),'selection_n':len(selection),'calibration_n':len(calibration)}
    (OUT/'leaderboard.json').write_text(json.dumps(records,indent=2));(OUT/'frozen_selection.json').write_text(json.dumps(frozen,indent=2))
    print('SELECTION FROZEN',json.dumps(primary),flush=True)

if __name__=='__main__':main()
