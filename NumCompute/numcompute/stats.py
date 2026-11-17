
import numpy as np

def describe(X,axis=0):
    
    a=np.asarray(X,dtype=float)
    return {"mean":np.nanmean(a,axis=axis),"median":np.nanmedian(a,axis=axis),
            "std":np.nanstd(a,axis=axis),"min":np.nanmin(a,axis=axis),"max":np.nanmax(a,axis=axis)}

def histogram(X,bins=10,range=None):
    
    a=np.asarray(X,dtype=float);return np.histogram(a[np.isfinite(a)],bins=bins,range=range)

def quantiles(X,q):
    
    return np.nanquantile(X,q)

class RunningStats:
    
    def __init__(self):self.n=0;self.mean=0.;self.m2=0.;self.values=[]
    def update(self,X):
        a=np.asarray(X,dtype=float).ravel();a=a[np.isfinite(a)]
        if not len(a):return self
        n=len(a);avg=a.mean();delta=avg-self.mean;total=self.n+n
        self.m2+=np.sum((a-avg)**2)+delta**2*self.n*n/total
        self.mean+=delta*n/total;self.n=total;self.values.extend(a.tolist())
        return self
    update_stats=update
    def result(self):return {"count":self.n,"mean":self.mean if self.n else np.nan,
                             "variance":self.m2/self.n if self.n else np.nan}
    def quantile(self,q):return np.quantile(self.values,q) if self.values else np.nan
    def histogram(self,bins=10):return np.histogram(self.values,bins=bins)
