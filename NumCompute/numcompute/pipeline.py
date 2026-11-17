
class Pipeline:
    
    def __init__(self,steps):
        if not steps:raise ValueError("At least one step required")
        self.steps=steps
    def fit(self,X,y=None):
        for name,step in self.steps[:-1]:X=step.fit_transform(X)
        last=self.steps[-1][1]
        if hasattr(last,"predict"):last.fit(X,y)
        else:last.fit(X)
        return self
    def fit_transform(self,X):
        self.fit(X)
        return self.transform(X)
    def transform(self,X):
        for name,step in self.steps:
            if not hasattr(step,"transform"):break
            X=step.transform(X)
        return X
    def partial_fit(self,X,y=None):
        for name,step in self.steps[:-1]:
            step.partial_fit(X);X=step.transform(X)
        last=self.steps[-1][1]
        if hasattr(last,"predict"):last.partial_fit(X,y)
        else:last.partial_fit(X)
        return self
    def predict(self,X):
        for name,step in self.steps[:-1]:X=step.transform(X)
        return self.steps[-1][1].predict(X)
