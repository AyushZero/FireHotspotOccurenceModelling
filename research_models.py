"""Serializable, training-only transformations and matched penalized logits."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.preprocessing import StandardScaler, SplineTransformer

NAMES = ['t_max','ndvi','ndwi','rh_min','sm_top','u10_max','slope','frp_hist','count_hist']

def drivers(frame):
    x=frame[NAMES].to_numpy(dtype=float).copy()
    x[:,2:5]=1-x[:,2:5]
    return x

class Features:
    def __init__(self,kind='standard',knots=4,epsilon=0):
        self.kind=kind; self.knots=knots; self.epsilon=epsilon
    def base(self,x):
        x=x.copy()
        if self.kind in ('loghistory','spline','interactions','tree_enriched'):
            # Inputs are normalized; the scale is fixed, not estimated on held-out rows.
            x[:,7:]=np.log1p(1000*x[:,7:])
        if self.kind=='interactions':
            x=np.column_stack([x,x[:,0]*x[:,3],x[:,1]*x[:,2],x[:,3]*x[:,4],x[:,5]*x[:,6]])
        if self.kind=='multiplicative':
            e=self.epsilon
            dry=.3*x[:,2]+.4*x[:,3]+.3*x[:,4]
            x=((e+x[:,0])*(e+x[:,1])*(e+dry)*(1+.6*x[:,5]+.4*x[:,6])+.6*x[:,7]+.4*x[:,8])[:,None]
        return x
    def fit(self,x):
        a=self.base(x)
        self.spline=None
        if self.kind=='spline':
            self.spline=SplineTransformer(n_knots=self.knots,degree=3,include_bias=False,knots='quantile').fit(a)
            a=self.spline.transform(a)
        self.scaler=StandardScaler().fit(a)
        return self
    def transform(self,x):
        a=self.base(x)
        if self.spline is not None:a=self.spline.transform(a)
        return a if self.kind=='minmax' else self.scaler.transform(a)

class PenalizedLogit:
    def __init__(self,C=1,positive=False):self.C=C; self.positive=positive
    def fit(self,x,y):
        n,d=x.shape; lam=1/(self.C*n)
        def objective(p):
            z=x@p[:-1]+p[-1]; err=expit(z)-y
            value=np.mean(np.logaddexp(0,z)-y*z)+lam*np.dot(p[:-1],p[:-1])/2
            grad=np.r_[x.T@err/n+lam*p[:-1],err.mean()]
            return value,grad
        prior=np.clip(y.mean(),1e-8,1-1e-8)
        fit=minimize(objective,np.r_[np.zeros(d),np.log(prior/(1-prior))],jac=True,method='L-BFGS-B',bounds=([(0,None)]*d+[(None,None)]) if self.positive else None,options={'maxiter':3000,'ftol':1e-11,'gtol':1e-7})
        if not fit.success:raise RuntimeError(str(fit.message))
        self.coef_=fit.x[:-1]; self.intercept_=fit.x[-1]; self.iterations_=fit.nit
        return self
    def predict_proba(self,x):
        p=expit(x@self.coef_+self.intercept_)
        return np.column_stack([1-p,p])
