
import numpy as np

def distances(X, Y):
    
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if X.ndim != 2 or Y.ndim != 2 or X.shape[1] != Y.shape[1]:
        raise ValueError("X and Y must be 2D with equal feature counts")
    if not np.isfinite(X).all() or not np.isfinite(Y).all():
        raise ValueError("Distance inputs must contain finite values")
    return np.sqrt(np.sum((X[:, None, :] - Y[None, :, :]) ** 2, axis=2))

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
