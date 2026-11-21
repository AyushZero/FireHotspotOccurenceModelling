"""Shared scoring, probability mapping and paired moving-block bootstrap."""
import numpy as np
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss

def metrics(y,p):
    k=int(np.ceil(.1*len(y))); ix=np.argsort(-p,kind='stable')[:k]
    return {'auc':float(roc_auc_score(y,p)),'ap':float(average_precision_score(y,p)),'brier':float(brier_score_loss(y,p)),'precision_top10':float(y[ix].mean()),'recall_top10':float(y[ix].sum()/y.sum()),'mean':float(p.mean())}

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
