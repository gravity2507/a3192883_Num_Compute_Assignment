
import numpy as np

def distances(X,Y):
    
    X=np.asarray(X,dtype=float);Y=np.asarray(Y,dtype=float)
    return np.sqrt(np.maximum(np.sum(X**2,axis=1)[:,None]+np.sum(Y**2,axis=1)[None,:]-2*X@Y.T,0))

def logsumexp(x,axis=None):
    
    x=np.asarray(x,dtype=float);m=np.max(x,axis=axis,keepdims=True)
    safe=np.where(np.isfinite(m),m,0)
    out=np.log(np.sum(np.exp(x-safe),axis=axis,keepdims=True))+safe
    return np.squeeze(out,axis=axis) if axis is not None else out.item()

def softmax(x,axis=-1):
    
    x=np.asarray(x,dtype=float);z=np.exp(x-np.max(x,axis=axis,keepdims=True))
    return z/np.sum(z,axis=axis,keepdims=True)

def batches(X,size):
    if size<1:raise ValueError("size must be positive")
    for i in range(0,len(X),size):yield X[i:i+size]
