"""Joint physical-index transform, smooth additive logit and convex ensemble."""
import numpy as np
from scipy.special import expit
from scipy.optimize import minimize
from research_models import Features

class JointIndex:
    def __init__(self,weights,wind,slope,offset,history,mix,power,history_log=False):
        self.weights=np.asarray(weights);self.wind=wind;self.slope=slope;self.offset=offset;self.history=history;self.mix=mix;self.power=power;self.history_log=history_log
    def raw(self,x):
        dry=x[:,2:5]@self.weights;e=self.offset
        environment=((e+x[:,0])*(e+x[:,1])*(e+dry))**self.power
        environment*=1+self.wind*x[:,5]+self.slope*x[:,6]
        h=x[:,7:9]
        if getattr(self,'history_log',False):h=np.log1p(1000*h)/np.log(1001)
        return (environment+self.history*(self.mix*h[:,0]+(1-self.mix)*h[:,1]))[:,None]
    def fit(self,x):
        a=self.raw(x);self.mean_=a.mean(axis=0);self.scale_=np.maximum(a.std(axis=0),1e-8);return self
    def transform(self,x):return (self.raw(x)-self.mean_)/self.scale_

class SmoothLogit:
    def __init__(self,C,alpha,penalty,scales=None):self.C=C;self.alpha=alpha;self.penalty=penalty;self.scales=scales
    def value_gradient(self,p,x,y):
        n=len(y);ridge=1/(self.C*n);R=self.penalty
        w=p[:-1];z=x@w+p[-1];error=expit(z)-y;curvature=R@w
        loss=np.mean(np.logaddexp(0,z)-y*z)+ridge*np.dot(w,w)/2+self.alpha*np.dot(curvature,curvature)/2
        gradient=np.r_[x.T@error/n+ridge*w+self.alpha*(R.T@curvature),error.mean()]
        return loss,gradient
    def fit(self,x,y):
        n,d=x.shape
        # Optimize in original-basis coefficient coordinates. This is the same
        # objective, but avoids a badly conditioned difference operator when
        # rare-event spline columns have tiny training standard deviations.
        saved_scales=getattr(self,'scales',None)
        scales=np.ones(d) if saved_scales is None or self.alpha==0 else np.asarray(saved_scales)
        a=x*scales;R=self.penalty*scales[None,:];ridge=scales**2/(self.C*n)
        def objective(p):
            theta=p[:-1];z=a@theta+p[-1];err=expit(z)-y;curve=R@theta
            value=np.mean(np.logaddexp(0,z)-y*z)+np.dot(ridge*theta,theta)/2+self.alpha*np.dot(curve,curve)/2
            gradient=np.r_[a.T@err/n+ridge*theta+self.alpha*(R.T@curve),err.mean()]
            return value,gradient
        prior=y.mean();fit=minimize(objective,np.r_[np.zeros(d),np.log(prior/(1-prior))],jac=True,method='L-BFGS-B',options={'maxiter':10000,'ftol':1e-10,'gtol':1e-6})
        if not fit.success:raise RuntimeError(str(fit.message))
        self.coef_=fit.x[:-1]*scales;self.intercept_=fit.x[-1];self.iterations_=fit.nit;return self
    def predict_proba(self,x):
        p=expit(x@self.coef_+self.intercept_);return np.column_stack([1-p,p])

def spline_penalty(tf):
    d=len(tf.scaler.scale_);width=d//9;blocks=[]
    # Each feature's omitted final bias basis coefficient is anchored at zero.
    for j in range(9):
        diff=np.diff(np.eye(width+1),n=2,axis=0)[:,:width]
        block=np.zeros((len(diff),d));block[:,j*width:(j+1)*width]=diff/tf.scaler.scale_[None,j*width:(j+1)*width];blocks.append(block)
    return np.vstack(blocks)

class Ensemble:
    def __init__(self,bundles,weights):self.bundles=bundles;self.weights=np.asarray(weights)
    def predict_proba(self,x):
        p=sum(w*b['model'].predict_proba(b['transform'].transform(x))[:,1] for w,b in zip(self.weights,self.bundles) if w>0)
        return np.column_stack([1-p,p])

class IdentityFeatures:
    def transform(self,x):return x
