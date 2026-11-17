
import numpy as np

def rank(data,method="average"):
    
    a=np.asarray(data,dtype=float)
    if a.ndim!=1:raise ValueError("Expected 1D")
    if method not in ("average","dense","ordinal"):raise ValueError("Unknown method")
    order=np.argsort(a,kind="stable");v=a[order];n=len(v)
    if not n:return np.array([])
    starts=np.r_[True,v[1:]!=v[:-1]]
    group=np.cumsum(starts)-1
    first=np.flatnonzero(starts);last=np.r_[first[1:],n]
    result={"average":(first[group]+last[group]+1)/2,
            "dense":group+1,"ordinal":np.arange(1,n+1)}[method]
    out=np.empty(n,dtype=float);out[order]=result
    out[np.isnan(a)]=np.nan
    return out

def percentile(data,q,interpolation="linear"):
    
    if interpolation not in ("linear","lower","higher","midpoint"):raise ValueError("Unknown interpolation")
    return np.nanpercentile(data,q,method=interpolation)
