"""Freeze calibration on 2023, then evaluate the unseen 2024 labels once."""
from pathlib import Path
import json,joblib
import numpy as np,xarray as xr,pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss
from research_models import drivers
from build_standalone_data import load_frames
from evaluation_utils import fit_mapping,mapped,bootstrap
from evaluation_utils import metrics

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'outputs/standalone';MODELS=ROOT/'models/standalone'

def baseline_features(train,frames):
    raw=xr.open_dataset(ROOT/'data/standalone/raw_cell_days.nc');truth=raw.ignition.values
    lag=np.zeros_like(truth);lag[1:]=truth[:-1];week=np.zeros_like(truth)
    for k in range(1,8):week[k:]=np.maximum(week[k:],truth[:-k])
    counts=train.groupby('cell').ignition.agg(['sum','count']);prior=train.ignition.mean();result={}
    for name,f in frames.items():
        days=(f.time-pd.Timestamp('2021-01-01')).dt.days.to_numpy();cell=f.cell.to_numpy(int);doy=f.time.dt.dayofyear.to_numpy()
        stats=counts.reindex(cell);sums=stats['sum'].to_numpy();n=stats['count'].to_numpy()
        if name=='train':rate=(sums-f.ignition.to_numpy()+10*prior)/(n-1+10)
        else:rate=(sums+10*prior)/(n+10)
        result[name]={'persistence':np.column_stack([lag[days,cell],week[days,cell]]),'climatology':np.column_stack([rate,np.sin(2*np.pi*doy/365.25),np.cos(2*np.pi*doy/365.25)]),'weather':drivers(f)[:,[0,3,5]]}
    raw.close();return result

def main():
    frozen=json.loads((OUT/'frozen_selection.json').read_text());primary=frozen['primary']['id']
    train,val,test=load_frames();cal=val[val.time.dt.day>=22];ycal=cal.ignition.to_numpy(int);fit=cal.time.dt.day.to_numpy()<=25;assess=~fit
    chosen={r['id']:r for r in frozen['family_winners'].values()};decisions={}
    # Baselines have fixed specifications and are fit before benchmark scoring.
    frames={'train':train,'cal':cal,'test':test};base=baseline_features(train,frames);base_models={};base_cal={}
    for name in ['persistence','climatology','weather']:
        scaler=StandardScaler().fit(base['train'][name]);model=LogisticRegression(C=1,max_iter=2000).fit(scaler.transform(base['train'][name]),train.ignition)
        base_models[name]=(scaler,model);base_cal[name]=model.predict_proba(scaler.transform(base['cal'][name]))[:,1]
        joblib.dump({'scaler':scaler,'model':model},MODELS/f'baseline_{name}.joblib')
    calibration_predictions={key:np.load(OUT/f'{key}_validation.npz')['calibration'] for key in chosen}
    calibration_predictions.update({'baseline_'+k:v for k,v in base_cal.items()})
    for key,pc in calibration_predictions.items():
        scores={}
        for kind in ['identity','sigmoid','isotonic']:
            model=fit_mapping(kind,pc[fit],ycal[fit])
            if kind=='sigmoid' and model.coef_[0,0]<=0:continue
            scores[kind]=float(brier_score_loss(ycal[assess],mapped(kind,model,pc[assess])))
        kind=min(scores,key=scores.get);model=fit_mapping(kind,pc,ycal)
        decisions[key]={'kind':kind,'assessment_brier':scores}
        joblib.dump({'kind':kind,'model':model},MODELS/f'{key}_calibration.joblib')
    (OUT/'frozen_calibration.json').write_text(json.dumps(decisions,indent=2))
    # All model and probability-mapping decisions are now fixed.
    x=drivers(test);y=test.ignition.to_numpy(int);pred={};raw_predictions={};results={}
    for key,r in chosen.items():
        bundle=joblib.load(MODELS/f'{key}.joblib');p=bundle['model'].predict_proba(bundle['transform'].transform(x))[:,1]
        cm=joblib.load(MODELS/f'{key}_calibration.joblib');q=mapped(cm['kind'],cm['model'],p)
        raw_predictions[key]=p;pred[key]=q;results[key]={'spec':r['spec'],'selection':r['selection'],'raw_test':metrics(y,p),'test':metrics(y,q),'calibration':decisions[key]}
    for name,(scaler,model) in base_models.items():
        key='baseline_'+name;p=model.predict_proba(scaler.transform(base['test'][name]))[:,1];cm=joblib.load(MODELS/f'{key}_calibration.joblib');q=mapped(cm['kind'],cm['model'],p)
        pred[key]=q;raw_predictions[key]=p;results[key]={'spec':{'family':name},'raw_test':metrics(y,p),'test':metrics(y,q),'calibration':decisions[key]}
    pred['constant']=np.full(len(y),train.ignition.mean());results['constant']={'spec':{'family':'constant'},'test':metrics(y,pred['constant'])}
    np.savez_compressed(OUT/'test_predictions.npz',label=y,**pred);np.savez_compressed(OUT/'test_raw_predictions.npz',**raw_predictions)
    pd.DataFrame({'time':test.time,'latitude':test.latitude,'longitude':test.longitude,'label':y,'probability':pred[primary]}).to_csv(OUT/'selected_test_predictions.csv',index=False)
    report={'primary':primary,'test_year':2024,'train_years':[2021,2022],'validation_year':2023,'n':len(y),'events':int(y.sum()),'base_rate':float(y.mean()),'train_rate':float(train.ignition.mean()),'models':results,'brier_skill':1-results[primary]['test']['brier']/results['constant']['test']['brier']}
    (OUT/'results.json').write_text(json.dumps(report,indent=2));print('TEST',primary,results[primary],flush=True)
    days=(test.time-pd.Timestamp('2024-01-01')).dt.days.to_numpy();comparisons={k:pred[k] for k in pred if k!=primary and k!='constant'};comparisons[primary]=pred[primary]
    intervals={}
    for length in [7,30]:
        print('Bootstrap',length,flush=True);aucs,aps=bootstrap(y,comparisons,days,length,n=300)
        intervals[str(length)]={k:{'auc_difference':results[primary]['test']['auc']-results[k]['test']['auc'],'auc_ci':np.quantile(aucs[primary]-aucs[k],[.025,.975]).tolist(),'ap_difference':results[primary]['test']['ap']-results[k]['test']['ap'],'ap_ci':np.quantile(aps[primary]-aps[k],[.025,.975]).tolist()} for k in comparisons if k!=primary}
    report['bootstrap']={'replicates':300,'blocks':intervals};(OUT/'results.json').write_text(json.dumps(report,indent=2));print('Evaluation complete',flush=True)

if __name__=='__main__':main()
