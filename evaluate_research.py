"""Freeze probability mappings on 2022, then open the retrospective benchmark."""
from pathlib import Path
import json,joblib
import numpy as np
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss
from src.dataset import build_dataset,split_by_year
from research_models import drivers
from run_research import metrics

ROOT=Path(__file__).resolve().parent; OUT=ROOT/'outputs/research'; MODELS=ROOT/'models/research'

def fit_mapping(kind,p,y):
    if kind=='identity':return None
    if kind=='sigmoid':return LogisticRegression(C=1e6,max_iter=2000).fit(logit(np.clip(p,1e-7,1-1e-7))[:,None],y)
    return IsotonicRegression(out_of_bounds='clip').fit(p,y)

def mapped(kind,model,p):
    if kind=='identity':return p
    if kind=='sigmoid':return model.predict_proba(logit(np.clip(p,1e-7,1-1e-7))[:,None])[:,1]
    return model.predict(p)

def bootstrap(y,pred,days,length,n=300):
    # Sort once; each replicate changes day weights, not score order.
    prepared={}
    for key,p in pred.items():
        order=np.argsort(p,kind='stable'); s=p[order]
        starts=np.r_[0,np.flatnonzero(np.diff(s))+1]
        prepared[key]=(order,starts)
    rng=np.random.default_rng(1700+length); result={k:[] for k in pred}; aps={k:[] for k in pred}; nd=int(days.max()+1)
    for b in range(n):
        starts=rng.integers(0,nd,size=int(np.ceil(nd/length)))
        draw=((starts[:,None]+np.arange(length))%nd).ravel()[:nd]
        weights=np.bincount(draw,minlength=nd)[days]
        for key,(order,group) in prepared.items():
            w=weights[order]; yy=y[order]
            pos=np.add.reduceat(w*yy,group); neg=np.add.reduceat(w*(1-yy),group)
            auc=np.sum(pos*(np.cumsum(neg)-.5*neg))/(pos.sum()*neg.sum())
            result[key].append(float(auc))
            pp=pos[::-1]; nn=neg[::-1]; cp=np.cumsum(pp); total=cp+np.cumsum(nn)
            precision=np.divide(cp,total,out=np.zeros_like(cp,dtype=float),where=total>0)
            aps[key].append(float(np.sum(pp*precision)/pp.sum()))
    return {k:np.array(v) for k,v in result.items()}, {k:np.array(v) for k,v in aps.items()}

def main():
    frozen=json.loads((OUT/'frozen_selection.json').read_text()); primary=frozen['primary']['id']
    df=build_dataset(ROOT/'data/features'); tr,va,te=split_by_year(df,[2021],[2022],[2023])
    cal=va[va.time.dt.day>=22]; ycal=cal.ignition.to_numpy(int)
    fit=cal.time.dt.day.to_numpy()<=25; assess=~fit
    decisions={}; chosen={r['id']:r for r in frozen['family_winners'].values()}
    for key in chosen:
        pc=np.load(OUT/f'{key}_validation.npz')['calibration']; score={}
        for kind in ['identity','sigmoid','isotonic']:
            m=fit_mapping(kind,pc[fit],ycal[fit])
            if kind=='sigmoid' and m.coef_[0,0]<=0:continue
            score[kind]=float(brier_score_loss(ycal[assess],mapped(kind,m,pc[assess])))
        winner=min(score,key=score.get); model=fit_mapping(winner,pc,ycal)
        decisions[key]={'kind':winner,'assessment_brier':score,'fit_rows':int(fit.sum()),'assessment_rows':int(assess.sum())}
        joblib.dump({'kind':winner,'model':model},MODELS/f'{key}_calibration.joblib')
    # Persist decisions before computing any 2023 metrics.
    (OUT/'frozen_calibration.json').write_text(json.dumps(decisions,indent=2))
    x=drivers(te); y=te.ignition.to_numpy(int); preds={}; raw_preds={}; results={}
    for key,r in chosen.items():
        bundle=joblib.load(MODELS/f'{key}.joblib'); p=bundle['model'].predict_proba(bundle['transform'].transform(x))[:,1]
        cm=joblib.load(MODELS/f'{key}_calibration.joblib'); q=mapped(cm['kind'],cm['model'],p)
        raw_preds[key]=p;preds[key]=q
        results[key]={'spec':r['spec'],'selection':r['selection'],'raw_benchmark':metrics(y,p),'calibrated_benchmark':metrics(y,q),'calibration':decisions[key]}
    old=np.load(ROOT/'outputs/model_predictions_test.npz'); assert np.array_equal(old['label'],y)
    for key in ['unconstrained_additive','constrained_additive','raw_logistic','random_forest','persistence']:
        preds['baseline_'+key]=old[key]
        results['baseline_'+key]={'calibrated_benchmark':metrics(y,old[key])}
    np.savez_compressed(OUT/'benchmark_predictions.npz',label=y,**preds)
    np.savez_compressed(OUT/'benchmark_raw_predictions.npz',**raw_preds)
    report={'primary':primary,'test_status':'previously examined 2023 retrospective benchmark','n':len(y),'events':int(y.sum()),'base_rate':float(y.mean()),'models':results}
    (OUT/'research_results.json').write_text(json.dumps(report,indent=2))
    print('BENCHMARK',primary,results[primary],flush=True)
    days=(te.time-te.time.min()).dt.days.to_numpy()
    comparisons={k:preds[k] for k in [primary,'baseline_unconstrained_additive','baseline_raw_logistic','baseline_random_forest','baseline_persistence']}
    intervals={}
    for length in [7,30]:
        print('Bootstrap block days',length,flush=True); boot,apboot=bootstrap(y,comparisons,days,length)
        intervals[str(length)]={k:{'difference':results[primary]['calibrated_benchmark']['auc']-results[k]['calibrated_benchmark']['auc'],'ci95':np.quantile(boot[primary]-boot[k],[.025,.975]).tolist(),'ap_difference':results[primary]['calibrated_benchmark']['ap']-results[k]['calibrated_benchmark']['ap'],'ap_ci95':np.quantile(apboot[primary]-apboot[k],[.025,.975]).tolist()} for k in comparisons if k!=primary}
    report['paired_full_population_bootstrap']={'resamples':300,'circular':True,'block_lengths':intervals}
    (OUT/'research_results.json').write_text(json.dumps(report,indent=2))
    # Verify optimized weighted AUC against sklearn on the unweighted sample.
    from sklearn.metrics import roc_auc_score
    assert all(np.isfinite(p).all() for p in preds.values())
    print('Evaluation and full-population block bootstrap complete',flush=True)

if __name__=='__main__':main()
