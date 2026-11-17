
import matplotlib.pyplot as plt

def _finish(save_path):
    plt.tight_layout()
    if save_path:plt.savefig(save_path,dpi=160)
    plt.show()

def plot_metric_over_time(metric_values,title="Accuracy over time",ylabel="Accuracy",save_path=None):
    
    plt.figure();plt.plot(range(1,len(metric_values)+1),metric_values,marker="o")
    plt.xlabel("Chunk");plt.ylabel(ylabel);plt.title(title);_finish(save_path)

def compare_models(metric1,metric2,labels=("Model A","Model B"),save_path=None):
    
    plt.figure();plt.plot(metric1,label=labels[0]);plt.plot(metric2,label=labels[1])
    plt.xlabel("Chunk");plt.ylabel("Accuracy");plt.title("Streaming model comparison")
    plt.legend();_finish(save_path)

def plot_predictions_vs_ground_truth(y_true,y_pred,save_path=None):
    
    plt.figure();plt.plot(y_true,"o",label="Actual",alpha=.6)
    plt.plot(y_pred,"x",label="Predicted",alpha=.6)
    plt.xlabel("Sample");plt.ylabel("Class");plt.title("Predictions vs ground truth")
    plt.legend();_finish(save_path)
