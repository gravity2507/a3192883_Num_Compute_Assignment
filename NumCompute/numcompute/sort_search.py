
import numpy as np

def stable_sort(values):
    
    return np.sort(values,kind="stable")

def multi_key_sort(X, columns):
    
    X=np.asarray(X)
    return X[np.lexsort(tuple(X[:,j] for j in reversed(columns)))]

def topk(values,k,largest=True,return_indices=True):
    
    a=np.asarray(values)
    if a.ndim!=1 or not 0<=k<=len(a):raise ValueError("Invalid k or shape")
    if k==0:return np.array([],dtype=int if return_indices else a.dtype)
    key=np.where(np.isnan(a),-np.inf if largest else np.inf,a) if np.issubdtype(a.dtype,np.floating) else a
    idx=np.argpartition(-key if largest else key,k-1)[:k]
    idx=idx[np.argsort(-key[idx] if largest else key[idx],kind="stable")]
    return idx if return_indices else a[idx]

def quickselect(values,k):
    
    a=np.asarray(values)
    if a.ndim!=1 or not 0<=k<len(a):raise ValueError("Invalid k")
    return np.partition(a,k)[k]

def binary_search(sorted_array,x):
    
    a=np.asarray(sorted_array);i=int(np.searchsorted(a,x))
    return i,bool(i<len(a) and a[i]==x)
