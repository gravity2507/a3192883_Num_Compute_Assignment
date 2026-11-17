
import numpy as np

def grad(f,x,h=1e-5,method="central"):
    
    x=np.asarray(x,dtype=float)
    if x.ndim!=1 or (not np.isfinite(h)) or h<=0 or method not in ("central","forward"):raise ValueError("Invalid input")
    eye=np.eye(len(x))*h
    return np.array([((f(x+e)-f(x-e))/(2*h) if method=="central" else
                      (f(x+e)-f(x))/h) for e in eye])

def jacobian(F,x,h=1e-5):
    
    x=np.asarray(x,dtype=float)
    if x.ndim!=1 or (not np.isfinite(h)) or h<=0:raise ValueError("Invalid input")
    return np.column_stack([(np.asarray(F(x+e))-np.asarray(F(x-e)))/(2*h)
                            for e in np.eye(len(x))*h])
