
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from numcompute.io import iter_csv_chunks
from numcompute.preprocessing import Imputer
from numcompute.tree import DecisionTreeClassifier
from numcompute.pipeline import Pipeline
from numcompute.stream import StreamTrainer
from numcompute.visualise import plot_metric_over_time,plot_predictions_vs_ground_truth

csv=ROOT/"data"/"stream_data.csv"
if not csv.exists():
    exec(compile((ROOT/"data"/"make_data.py").read_text(),str(ROOT/"data"/"make_data.py"),"exec"))
model=Pipeline([("impute",Imputer()),("tree",DecisionTreeClassifier(max_depth=4,buffer_size=100))])
trainer=StreamTrainer(model)
latest=None
for chunk in iter_csv_chunks(csv,chunk_size=50):
    X,y=chunk[:,:-1],chunk[:,-1].astype(int)
    latest=trainer.fit_chunk(X,y)
    row=trainer.history[-1]
    print("Chunk",row["chunk"],"Accuracy",round(row["accuracy"],3),"Retrains",row["retrains"])
print("Final cumulative accuracy:",trainer.metrics.result()["accuracy"])
ROOT.joinpath("results").mkdir(exist_ok=True)
plot_metric_over_time([r["cumulative_accuracy"] for r in trainer.history],save_path=ROOT/"results"/"accuracy.png")
if latest is not None:
    plot_predictions_vs_ground_truth(y,latest[1],save_path=ROOT/"results"/"predictions.png")
