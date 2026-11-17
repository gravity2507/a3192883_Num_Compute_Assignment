
import numpy as np

def accuracy(y,p):
    
    y=np.asarray(y);p=np.asarray(p)
    if y.shape!=p.shape or not y.size:raise ValueError("Nonempty matching shapes required")
    return float(np.mean(y==p))

def confusion_matrix(y,p,labels=None):
   
    y=np.asarray(y);p=np.asarray(p)
    if y.shape!=p.shape:raise ValueError("Shapes differ")
    labels=np.unique(np.r_[y,p]) if labels is None else np.asarray(labels)
    return np.array([[np.sum((y==a)&(p==b)) for b in labels] for a in labels],dtype=int)

def precision(y,p,positive=1):
    
    y=np.asarray(y);p=np.asarray(p);d=np.sum(p==positive)
    return float(np.sum((y==positive)&(p==positive))/d) if d else 0.

def recall(y,p,positive=1):
    
    y=np.asarray(y);p=np.asarray(p);d=np.sum(y==positive)
    return float(np.sum((y==positive)&(p==positive))/d) if d else 0.

def f1(y,p,positive=1):
    
    a=precision(y,p,positive);b=recall(y,p,positive)
    return 2*a*b/(a+b) if a+b else 0.

def mse(y,p):
    
    y=np.asarray(y,dtype=float);p=np.asarray(p,dtype=float)
    if y.shape!=p.shape or not y.size:raise ValueError("Nonempty matching shapes required")
    return float(np.nanmean((y-p)**2))

def roc_curve(y,s):
    
    y=np.asarray(y);s=np.asarray(s,dtype=float)
    if y.shape!=s.shape or not np.all(np.isin(y,[0,1])):raise ValueError("Binary labels required")
    thresholds=np.r_[np.inf,np.unique(s)[::-1]]
    tpr=np.array([np.mean(s[y==1]>=t) if np.any(y==1) else 0 for t in thresholds])
    fpr=np.array([np.mean(s[y==0]>=t) if np.any(y==0) else 0 for t in thresholds])
    return fpr,tpr,thresholds

def auc(x,y):
    
    return float(np.trapezoid(y,x))

class StreamingMetrics:
    
    def __init__(self,window=None):self.window=window;self.reset()
    def reset(self):self.y=[];self.p=[];return self
    def update(self,y,p):
        y=np.asarray(y).ravel();p=np.asarray(p).ravel()
        if y.shape!=p.shape:raise ValueError("Shapes differ")
        self.y.extend(y.tolist());self.p.extend(p.tolist());return self
    def result(self):
        if not self.y:return {"accuracy":np.nan,"precision":np.nan,"recall":np.nan,"f1":np.nan,"confusion_matrix":np.zeros((0,0),int)}
        y=np.asarray(self.y);p=np.asarray(self.p)
        out={"accuracy":accuracy(y,p),"precision":precision(y,p),"recall":recall(y,p),
             "f1":f1(y,p),"confusion_matrix":confusion_matrix(y,p)}
        if self.window:out["rolling_accuracy"]=accuracy(y[-self.window:],p[-self.window:])
        return out
