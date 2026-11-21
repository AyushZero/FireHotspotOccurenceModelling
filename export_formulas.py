"""Export readable fitted mathematical components alongside serialized models."""
from pathlib import Path
import json,joblib,numpy as np

ROOT=Path(__file__).resolve().parent

def main():
    out=ROOT/'outputs/standalone';models=ROOT/'models/standalone';f=json.loads((out/'frozen_selection.json').read_text())
    smooth=f['family_winners']['smooth_additive'];bundle=joblib.load(models/f"{smooth['id']}.joblib");tf=bundle['transform'];m=bundle['model'];theta=m.coef_/tf.scaler.scale_;width=len(theta)//9
    names=['temperature','NDVI','1-NDWI','1-RH','1-SM','wind','slope','FRP_history','count_history']
    formula={'equation':'logit(p_raw) = raw_basis_intercept + sum_j sum_k theta[j,k] B[j,k](transformed_x_j)','input_scaling':'data/standalone/normalization.json; complement NDWI/RH/SM; apply log1p(1000*h) to FRP/count histories','raw_basis_intercept':float(m.intercept_-np.dot(theta,tf.scaler.mean_)),'degree':3,'extrapolation':'constant','omitted_final_basis_coefficient':0,'selection_parameters':smooth['spec'],'features':{name:{'extended_knot_vector':tf.spline.bsplines_[j].t.tolist(),'basis_coefficients':theta[j*width:(j+1)*width].tolist()} for j,name in enumerate(names)}}
    (out/'smooth_formula.json').write_text(json.dumps(formula,indent=2))
    joint=f['family_winners']['joint_index'];b=joblib.load(models/f"{joint['id']}.joblib");t=b['transform'];model=b['model'];coef=float(model.coef_[0,0]/t.scale_[0]);intercept=float(model.intercept_[0]-model.coef_[0,0]*t.mean_[0]/t.scale_[0])
    (out/'multiplicative_formula.json').write_text(json.dumps({'index_parameters':joint['spec']['parameters'],'raw_probability':'sigmoid(intercept + coefficient * joint_index)','intercept':intercept,'coefficient':coef,'note':'The separately selected probability mapping, if any, is applied after this raw probability.'},indent=2))
    selected=f['primary']['id'];cm=joblib.load(models/f'{selected}_calibration.joblib');mapping={'kind':cm['kind']}
    if cm['kind']=='sigmoid':mapping.update(coefficient=float(cm['model'].coef_[0,0]),intercept=float(cm['model'].intercept_[0]),input='clipped raw probability logit')
    if cm['kind']=='isotonic':mapping.update(x=cm['model'].X_thresholds_.tolist(),y=cm['model'].y_thresholds_.tolist())
    (out/'selected_method.json').write_text(json.dumps({'selected':f['primary'],'ensemble':f['family_winners']['ensemble']['spec'],'final_probability_mapping':mapping},indent=2))
    print('Exported fitted formulas and selected-method description')

if __name__=='__main__':main()
