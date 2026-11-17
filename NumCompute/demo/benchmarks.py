
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from numcompute.benchmarking import benchmark
from numcompute.tree import DecisionTreeClassifier
from numcompute.metrics import accuracy
from numcompute.visualise import compare_models
rng=np.random.default_rng(7)
a=rng.normal(size=20000)
def loop_sum():
    total=0
    for value in a:total+=value*value
    return total
def vector_sum():return np.sum(a*a)
print("Loop seconds:",benchmark(loop_sum))
print("NumPy seconds:",benchmark(vector_sum))
X=rng.normal(size=(400,3));y=(X[:,0]+X[:,1]>0).astype(int)
results={}
for size in (50,100,200):
    tree=DecisionTreeClassifier(max_depth=4,buffer_size=size)
    scores=[];times=[]
    for i in range(0,len(X),25):
        chunk=X[i:i+25];labels=y[i:i+25]
        if tree.retrains:scores.append(accuracy(labels,tree.predict(chunk)))
        else:scores.append(np.nan)
        times.append(benchmark(lambda:tree.partial_fit(chunk,labels),repeats=1))
    results[size]=scores
    print("buffer",size,"mean accuracy",round(float(np.nanmean(scores)),3),
          "retrain count",tree.retrains,"total update seconds",round(sum(times),4))
ROOT.joinpath("results").mkdir(exist_ok=True)
compare_models(results[50],results[200],labels=("Buffer 50","Buffer 200"),save_path=ROOT/"results"/"buffer_comparison.png")
