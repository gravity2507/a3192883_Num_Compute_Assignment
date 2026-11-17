
import numpy as np

class DecisionTreeClassifier:
    
    def __init__(self,max_depth=5,min_samples_split=2,max_features=None,
                 buffer_size=100,criterion="gini"):
        if max_depth<1 or min_samples_split<2 or buffer_size<1:raise ValueError("Invalid hyperparameter")
        if criterion not in ("gini","entropy"):raise ValueError("criterion must be gini or entropy")
        self.max_depth=max_depth;self.min_samples_split=min_samples_split
        self.max_features=max_features;self.buffer_size=buffer_size;self.criterion=criterion
        self.X_buffer=[];self.y_buffer=[];self.buffer_count=0;self.retrains=0

    def _check(self,X,y=None):
        X=np.asarray(X,dtype=float)
        if X.ndim!=2:raise ValueError("X must be a 2D array")
        if y is not None:
            y=np.asarray(y)
            if y.ndim!=1 or len(y)!=len(X) or not len(y):raise ValueError("y must match nonempty X")
            if not np.all(np.isfinite(y.astype(float))):raise ValueError("Labels must be finite numbers")
            return X,y
        return X

    def _impurity(self,counts):
        p=counts/counts.sum(axis=-1,keepdims=True)
        if self.criterion=="gini":return 1-np.sum(p*p,axis=-1)
        return -np.sum(np.where(p>0,p*np.log2(np.maximum(p,1e-15)),0),axis=-1)

    def _grow(self,X,y,depth):
        counts=np.bincount(y,minlength=len(self.classes_))
        node={"class":int(np.argmax(counts))}
        if depth>=self.max_depth or len(y)<self.min_samples_split or np.count_nonzero(counts)==1:return node
        best_gain=0.;best=None;parent=self._impurity(counts)
        features=range(X.shape[1]) if self.max_features is None else range(min(self.max_features,X.shape[1]))
        for j in features:
            order=np.argsort(X[:,j],kind="stable");values=X[order,j];labels=y[order]
            cuts=np.flatnonzero(values[:-1]<values[1:])
            if not len(cuts):continue
            onehot=np.eye(len(self.classes_),dtype=int)[labels]
            left=np.cumsum(onehot,axis=0)[cuts];right=counts-left
            nleft=cuts+1;nright=len(y)-nleft
            gain=parent-(nleft*self._impurity(left)+nright*self._impurity(right))/len(y)
            k=int(np.argmax(gain))
            if gain[k]>best_gain+1e-12:
                best_gain=float(gain[k]);best=(j,(values[cuts[k]]+values[cuts[k]+1])/2)
        if best is None:return node
        j,t=best;mask=X[:,j]<=t
        node.update(feature=j,threshold=float(t),left=self._grow(X[mask],y[mask],depth+1),
                    right=self._grow(X[~mask],y[~mask],depth+1))
        return node

    def fit(self,X,y):
        
        X,y=self._check(X,y)
        if X.shape[1]==0:raise ValueError("Need at least one feature")
        self.n_features_=X.shape[1]
        self.fill_=np.array([np.median(col[np.isfinite(col)]) if np.any(np.isfinite(col)) else 0.
                             for col in X.T])
        X=np.where(np.isfinite(X),X,self.fill_)
        self.classes_,encoded=np.unique(y,return_inverse=True)
        self.root_=self._grow(X,encoded,0)
        return self

    def partial_fit(self,X,y):
        
        X,y=self._check(X,y)
        if hasattr(self,"n_features_") and X.shape[1]!=self.n_features_:raise ValueError("Feature count changed")
        if self.X_buffer and X.shape[1]!=self.X_buffer[0].shape[1]:raise ValueError("Feature count changed")
        self.X_buffer.append(X.copy());self.y_buffer.append(y.copy());self.buffer_count+=len(y)
        if self.buffer_count>=self.buffer_size:
            self.fit(np.vstack(self.X_buffer),np.concatenate(self.y_buffer))
            self.X_buffer=[];self.y_buffer=[];self.buffer_count=0;self.retrains+=1
        return self

    def predict(self,X):
        
        if not hasattr(self,"root_"):raise ValueError("Tree not trained: add more chunks")
        X=self._check(X)
        if X.shape[1]!=self.n_features_:raise ValueError("Feature count changed")
        X=np.where(np.isfinite(X),X,self.fill_)
        result=np.empty(len(X),dtype=self.classes_.dtype)
        def visit(node,indices):
            if not len(indices):return
            if "feature" not in node:result[indices]=self.classes_[node["class"]];return
            left=X[indices,node["feature"]]<=node["threshold"]
            visit(node["left"],indices[left]);visit(node["right"],indices[~left])
        visit(self.root_,np.arange(len(X)))
        return result
