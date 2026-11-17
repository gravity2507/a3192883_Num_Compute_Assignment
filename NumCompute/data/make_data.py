
import numpy as np
from pathlib import Path
rng=np.random.default_rng(42)
X=rng.normal(size=(600,3))
y=((X[:,0]+0.7*X[:,1]-0.3*X[:,2])>0).astype(int)
X[rng.random(X.shape)<0.04]=np.nan
path=Path(__file__).resolve().parent/"stream_data.csv"
np.savetxt(path,np.column_stack([X,y]),delimiter=",",header="a,b,c,label",comments="",fmt="%.5f")
print("Created",path)
