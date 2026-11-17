
import numpy as np
from .metrics import StreamingMetrics,accuracy

class StreamTrainer:
    
    def __init__(self,model):self.model=model;self.metrics=StreamingMetrics();self.history=[]
    def score_chunk(self,X,y):
        
        tree=self.model.steps[-1][1] if hasattr(self.model,"steps") else self.model
        if not hasattr(tree,"root_"):return None
        try:p=self.model.predict(X)
        except ValueError as exc:
            if "not trained" in str(exc):return None
            raise
        self.metrics.update(y,p)
        return accuracy(y,p),p
    def fit_chunk(self,X,y):
        
        score=self.score_chunk(X,y)
        self.model.partial_fit(X,y)
        tree=self.model.steps[-1][1] if hasattr(self.model,"steps") else self.model
        memory=sum(a.nbytes for a in tree.X_buffer+tree.y_buffer)
        self.history.append({"chunk":len(self.history)+1,
                             "accuracy":score[0] if score else np.nan,
                             "cumulative_accuracy":self.metrics.result()["accuracy"],
                             "buffer_bytes":memory,"retrains":tree.retrains})
        return score
