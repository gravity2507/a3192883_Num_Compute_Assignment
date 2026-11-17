
import numpy as np

def matrix(X):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2: raise ValueError("Expected a 2D array")
    return X

class StandardScaler:
    
    def __init__(self): self.n = 0
    def fit(self, X):
        self.__init__()
        return self.partial_fit(X)
    def partial_fit(self, X):
        X = matrix(X)
        if not len(X): return self
        n = np.sum(~np.isnan(X), axis=0)
        mean = np.divide(np.nansum(X, axis=0), n, out=np.zeros(X.shape[1]), where=n>0)
        var = np.divide(np.nansum((X-mean)**2, axis=0), n, out=np.zeros(X.shape[1]), where=n>0)
        if self.n == 0:
            self.count, self.mean_, self.m2_ = n, mean, var*n
        else:
            total = self.count+n
            delta = mean-self.mean_
            self.mean_ += np.divide(delta*n, total, out=np.zeros_like(delta), where=total>0)
            self.m2_ += var*n + np.divide(delta**2*self.count*n, total, out=np.zeros_like(delta), where=total>0)
            self.count = total
        self.n += len(X)
        self.var_ = np.divide(self.m2_, self.count, out=np.zeros_like(self.m2_), where=self.count>0)
        return self
    def transform(self, X):
        X=matrix(X)
        return (X-self.mean_)/np.where(self.var_>0, np.sqrt(self.var_), 1)
    def fit_transform(self, X): return self.fit(X).transform(X)

class MinMaxScaler:
    
    def __init__(self): self.seen=False
    def fit(self, X): self.__init__(); return self.partial_fit(X)
    def partial_fit(self, X):
        X=matrix(X)
        if not len(X): return self
        lo=np.min(np.where(np.isnan(X), np.inf, X), axis=0)
        hi=np.max(np.where(np.isnan(X), -np.inf, X), axis=0)
        if self.seen:
            self.min_=np.minimum(self.min_,lo); self.max_=np.maximum(self.max_,hi)
        else: self.min_,self.max_=lo,hi;self.seen=True
        return self
    def transform(self, X):
        d=self.max_-self.min_
        return (matrix(X)-self.min_)/np.where(np.isfinite(d)&(d>0),d,1)
    def fit_transform(self,X): return self.fit(X).transform(X)

class Imputer:
    
    def __init__(self): self.count=None
    def fit(self,X): self.__init__();return self.partial_fit(X)
    def partial_fit(self,X):
        X=matrix(X);n=np.sum(~np.isnan(X),axis=0);s=np.nansum(X,axis=0)
        if self.count is None: self.count=n;self.total=s
        else: self.count+=n;self.total+=s
        self.mean_=np.divide(self.total,self.count,out=np.zeros_like(self.total),where=self.count>0)
        return self
    def transform(self,X):
        X=matrix(X).copy();return np.where(np.isnan(X),self.mean_,X)
    def fit_transform(self,X):return self.fit(X).transform(X)

class OneHotEncoder:
    
    def __init__(self): self.categories_=None
    def fit(self,X): self.__init__();return self.partial_fit(X)
    def partial_fit(self,X):
        X=np.asarray(X,dtype=str)
        if X.ndim!=2:raise ValueError("Expected 2D categories")
        if self.categories_ is None:self.categories_=[np.array([],dtype=str) for _ in range(X.shape[1])]
        if len(self.categories_)!=X.shape[1]:raise ValueError("Column count changed")
        for j in range(X.shape[1]):self.categories_[j]=np.union1d(self.categories_[j],X[:,j])
        return self
    def transform(self,X):
        X=np.asarray(X,dtype=str)
        if X.ndim!=2 or X.shape[1]!=len(self.categories_):raise ValueError("Invalid shape")
        return np.concatenate([(X[:,j,None]==cats).astype(float) for j,cats in enumerate(self.categories_)],axis=1)
    def fit_transform(self,X):return self.fit(X).transform(X)
