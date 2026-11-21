"""Audit reported numbers, mathematical routines and frozen inference."""
from pathlib import Path
import json,joblib,hashlib
import numpy as np
from scipy.optimize._numdiff import approx_derivative
from sklearn.metrics import roc_auc_score,average_precision_score
from build_standalone_data import load_frames
from evaluation_utils import metrics
from standalone_models import SmoothLogit
from predict_standalone import predict

ROOT=Path(__file__).resolve().parent

def main():
    out=ROOT/'outputs/standalone';result=json.loads((out/'results.json').read_text());frozen=json.loads((out/'frozen_selection.json').read_text());board=json.loads((out/'leaderboard.json').read_text());blend=json.loads((out/'ensemble_search.json').read_text())
    p=np.load(out/'test_predictions.npz');y=p['label'];checks={}
    for key,r in result['models'].items():
        actual=metrics(y,p[key]);assert all(abs(actual[k]-v)<1e-10 for k,v in r['test'].items()),key
    best=max([*board,frozen['family_winners']['ensemble']],key=lambda r:(r['selection']['ap'],r['selection']['auc']))
    assert best['id']==frozen['primary']['id']==result['primary']
    for b in blend:assert min(b['weights'])>=0 and abs(sum(b['weights'])-1)<1e-10
    train,val,test=load_frames();assert train.year.max()==2022 and val.year.unique().tolist()==[2023] and test.year.unique().tolist()==[2024]
    assert np.array_equal(y,test.ignition.to_numpy(int));q=predict(test);diff=float(np.max(np.abs(q-p[result['primary']])));assert diff<1e-10
    rng=np.random.default_rng(42);xx=rng.normal(size=(137,7));yy=rng.integers(0,2,137);R=rng.normal(size=(4,7));pars=rng.normal(size=8);model=SmoothLogit(.3,.07,R)
    _,g=model.value_gradient(pars,xx,yy);numeric=approx_derivative(lambda a:model.value_gradient(a,xx,yy)[0],pars,method='3-point').ravel();error=float(np.max(abs(g-numeric)));assert error<1e-7
    yy=rng.integers(0,2,500);pp=np.round(rng.random(500),1);ww=rng.integers(0,4,500);order=np.argsort(pp,kind='stable');groups=np.r_[0,np.flatnonzero(np.diff(pp[order]))+1]
    pos=np.add.reduceat(ww[order]*yy[order],groups);neg=np.add.reduceat(ww[order]*(1-yy[order]),groups)
    auc=np.sum(pos*(np.cumsum(neg)-.5*neg))/(pos.sum()*neg.sum());cp=np.cumsum(pos[::-1]);ap=np.sum(pos[::-1]*cp/(cp+np.cumsum(neg[::-1])))/pos.sum()
    assert abs(auc-roc_auc_score(yy,pp,sample_weight=ww))<1e-12;assert abs(ap-average_precision_score(yy,pp,sample_weight=ww))<1e-12
    audit={'metric_rows_verified':len(result['models']),'candidate_fits':len(board),'ensemble_candidates':len(blend),'selected_inference_max_abs_difference':diff,'smooth_gradient_max_abs_difference':error,'weighted_auc_ap_match_sklearn':True,'selected_by_validation_only':True,'year_partition_verified':True,'features_sha256':hashlib.sha256((ROOT/'data/standalone/features.nc').read_bytes()).hexdigest()}
    assert audit['features_sha256']==json.loads((out/'protocol.json').read_text())['features_sha256']
    if (out/'smooth_formula.json').exists():
        from scipy.interpolate import BSpline
        from scipy.special import expit
        from research_models import drivers
        formula=json.loads((out/'smooth_formula.json').read_text());a=drivers(test.iloc[:100]);a[:,7:]=np.log1p(1000*a[:,7:]);eta=np.full(len(a),formula['raw_basis_intercept'])
        for j,info in enumerate(formula['features'].values()):
            knots=np.array(info['extended_knot_vector']);coeff=np.r_[info['basis_coefficients'],0]
            eta+=BSpline(knots,coeff,3)(np.clip(a[:,j],knots[3],knots[-4]))
        smooth_key=frozen['family_winners']['smooth_additive']['id'];raw=np.load(out/'test_raw_predictions.npz')[smooth_key][:100]
        formula_diff=float(np.max(abs(expit(eta)-raw)));assert formula_diff<1e-8;audit['exported_smooth_formula_max_abs_difference']=formula_diff
    (out/'verification.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit,indent=2))

if __name__=='__main__':main()
