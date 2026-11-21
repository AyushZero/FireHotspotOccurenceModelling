"""Verify stored metrics, selection, optimizer/weighted-metric math and selected refit."""
from pathlib import Path
import json,joblib
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score
from sklearn.linear_model import LogisticRegression
from src.dataset import build_dataset,split_by_year
from research_models import drivers,Features,PenalizedLogit
from run_research import metrics
from evaluate_research import mapped

def main():
    root=Path(__file__).resolve().parent; out=root/'outputs/research'
    result=json.loads((out/'research_results.json').read_text()); frozen=json.loads((out/'frozen_selection.json').read_text()); board=json.loads((out/'leaderboard.json').read_text())
    chosen=max(board,key=lambda r:(r['selection']['ap'],r['selection']['auc']))
    assert chosen['id']==result['primary']==frozen['primary']['id']
    pred=np.load(out/'benchmark_predictions.npz');y=pred['label']
    for key,r in result['models'].items():
        actual=metrics(y,pred[key]); assert all(abs(actual[k]-v)<1e-10 for k,v in r['calibrated_benchmark'].items()),key
    # Check tie-aware weighted AUC/AP formulas used inside the block bootstrap.
    rng=np.random.default_rng(42); yy=rng.integers(0,2,500);p=np.round(rng.random(500),1);w=rng.integers(0,4,500)
    order=np.argsort(p,kind='stable'); groups=np.r_[0,np.flatnonzero(np.diff(p[order]))+1]
    pos=np.add.reduceat(w[order]*yy[order],groups);neg=np.add.reduceat(w[order]*(1-yy[order]),groups)
    auc=np.sum(pos*(np.cumsum(neg)-.5*neg))/(pos.sum()*neg.sum())
    pp=pos[::-1];nn=neg[::-1];cp=np.cumsum(pp);ap=np.sum(pp*cp/(cp+np.cumsum(nn)))/pp.sum()
    assert abs(auc-roc_auc_score(yy,p,sample_weight=w))<1e-12
    assert abs(ap-average_precision_score(yy,p,sample_weight=w))<1e-12
    xx=rng.normal(size=(2000,4));yy=(rng.random(2000)<1/(1+np.exp(2-xx[:,0]+xx[:,1]))).astype(int)
    a=PenalizedLogit(C=.1).fit(xx,yy);b=LogisticRegression(C=.1,tol=1e-9,max_iter=3000).fit(xx,yy)
    optimizer_diff=float(np.max(np.abs(a.predict_proba(xx)-b.predict_proba(xx))));assert optimizer_diff<1e-5
    frame=build_dataset(root/'data/features');tr,_,te=split_by_year(frame,[2021],[2022],[2023]);spec=chosen['spec']
    tf=Features(spec['kind'],epsilon=spec.get('epsilon',0)).fit(drivers(tr))
    saved=joblib.load(root/f"models/research/{chosen['id']}.joblib")
    model=saved['model'].__class__(**saved['model'].get_params());model.fit(tf.transform(drivers(tr)),tr.ignition.to_numpy(int))
    raw=model.predict_proba(tf.transform(drivers(te)))[:,1];cm=joblib.load(root/f"models/research/{chosen['id']}_calibration.joblib")
    repro_diff=float(np.max(np.abs(mapped(cm['kind'],cm['model'],raw)-pred[chosen['id']])));assert repro_diff<1e-7
    report={'reported_models_checked':len(result['models']),'candidate_count':len(board),'selected_refit_max_abs_probability_difference':repro_diff,'optimizer_verification_difference':optimizer_diff,'weighted_auc_ap_agree_with_sklearn':True,'selection_matches_validation_maximum':True}
    (out/'verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
