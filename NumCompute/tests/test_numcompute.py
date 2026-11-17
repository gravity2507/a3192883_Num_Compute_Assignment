import unittest
import numpy as np
from numcompute.io import read_csv,iter_csv_chunks
from numcompute.preprocessing import StandardScaler,MinMaxScaler,Imputer,OneHotEncoder
from numcompute.sort_search import stable_sort,multi_key_sort,topk,quickselect,binary_search
from numcompute.rank import rank,percentile
from numcompute.stats import RunningStats,describe,histogram,quantiles
from numcompute.metrics import accuracy,precision,recall,f1,mse,confusion_matrix,StreamingMetrics,roc_curve,auc
from numcompute.optim import grad,jacobian
from numcompute.utils import softmax,logsumexp,distances,batches
from numcompute.tree import DecisionTreeClassifier
from numcompute.pipeline import Pipeline
from numcompute.stream import StreamTrainer

class TestNumCompute(unittest.TestCase):
    def setUp(self):
        self.X=np.array([[0.,0.],[1.,1.],[2.,2.],[3.,3.]])
        self.y=np.array([0,0,1,1])
    def test_01_sort(self):self.assertEqual(stable_sort([3,1,2]).tolist(),[1,2,3])
    def test_02_multikey(self):self.assertEqual(multi_key_sort([[2,1],[1,2]],[0])[:,0].tolist(),[1,2])
    def test_03_topk(self):self.assertEqual(topk([1,5,3],2,return_indices=False).tolist(),[5,3])
    def test_04_empty_topk(self):self.assertEqual(len(topk([],0)),0)
    def test_05_bad_topk(self):
        with self.assertRaises(ValueError):topk([1],2)
    def test_06_quickselect(self):self.assertEqual(quickselect([4,2,3],1),3)
    def test_07_search_found(self):self.assertEqual(binary_search([1,3,5],3),(1,True))
    def test_08_search_missing(self):self.assertEqual(binary_search([1,3],2),(1,False))
    def test_09_rank_ties(self):np.testing.assert_allclose(rank([2,2,3]),[1.5,1.5,3])
    def test_10_dense_rank(self):np.testing.assert_allclose(rank([2,2,3],"dense"),[1,1,2])
    def test_11_percentile(self):self.assertEqual(percentile([1,3],50),2)
    def test_12_stats(self):self.assertEqual(describe([1,2,3])["mean"],2)
    def test_13_hist(self):self.assertEqual(histogram([1,2,np.nan],bins=2)[0].sum(),2)
    def test_14_quantiles(self):self.assertEqual(quantiles([1,3],.5),2)
    def test_15_running_stats(self):
        s=RunningStats().update([1,2]).update([3]);self.assertAlmostEqual(s.result()["variance"],2/3)
    def test_16_scaler(self):np.testing.assert_allclose(StandardScaler().fit_transform(self.X).mean(axis=0),0,atol=1e-12)
    def test_17_scaler_chunks(self):
        s=StandardScaler().partial_fit(self.X[:2]).partial_fit(self.X[2:]);self.assertAlmostEqual(s.mean_[0],1.5)
    def test_18_zero_variance(self):self.assertTrue(np.isfinite(StandardScaler().fit_transform(np.ones((3,2)))).all())
    def test_19_minmax(self):self.assertAlmostEqual(MinMaxScaler().fit_transform(self.X)[-1,0],1)
    def test_20_imputer(self):self.assertEqual(Imputer().fit_transform([[1],[np.nan]])[1,0],1)
    def test_21_onehot(self):self.assertEqual(OneHotEncoder().fit_transform([['a'],['b']]).shape,(2,2))
    def test_22_onehot_stream(self):
        e=OneHotEncoder().partial_fit([['a']]).partial_fit([['b']]);self.assertEqual(e.transform([['a']]).shape,(1,2))
    def test_23_accuracy(self):self.assertEqual(accuracy([0,1],[0,1]),1)
    def test_24_precision(self):self.assertEqual(precision([1,0],[1,1]),.5)
    def test_25_recall(self):self.assertEqual(recall([1,1],[1,0]),.5)
    def test_26_f1(self):self.assertEqual(f1([1,1],[1,0]),2/3)
    def test_27_mse(self):self.assertEqual(mse([1,2],[2,3]),1)
    def test_28_confusion(self):self.assertEqual(confusion_matrix([0,1],[0,1]).trace(),2)
    def test_29_metrics_chunks(self):self.assertEqual(StreamingMetrics().update([0],[0]).update([1],[0]).result()["accuracy"],.5)
    def test_30_rolling(self):self.assertEqual(StreamingMetrics(window=1).update([0,1],[0,0]).result()["rolling_accuracy"],0)
    def test_31_gradient(self):np.testing.assert_allclose(grad(lambda x:sum(x*x),[2.,3.]),[4,6],atol=1e-4)
    def test_32_jacobian(self):np.testing.assert_allclose(jacobian(lambda x:2*x,[1.,2.]),2*np.eye(2),atol=1e-4)
    def test_33_softmax(self):self.assertAlmostEqual(sum(softmax([1000,1001])),1)
    def test_34_logsumexp(self):self.assertTrue(np.isfinite(logsumexp([1000,1001])))
    def test_35_distances(self):self.assertAlmostEqual(distances([[0,0]],[[3,4]])[0,0],5)
    def test_36_batches(self):self.assertEqual(len(list(batches(range(5),2))),3)
    def test_37_tree_fit(self):self.assertEqual(accuracy(self.y,DecisionTreeClassifier().fit(self.X,self.y).predict(self.X)),1)
    def test_38_tree_not_trained(self):
        with self.assertRaises(ValueError):DecisionTreeClassifier(buffer_size=4).predict(self.X)
    def test_39_buffer(self):
        t=DecisionTreeClassifier(buffer_size=3);t.partial_fit(self.X[:2],self.y[:2]);self.assertEqual(t.retrains,0)
        t.partial_fit(self.X[2:],self.y[2:]);self.assertEqual(t.retrains,1);self.assertEqual(t.buffer_count,0)
    def test_40_old_tree(self):
        t=DecisionTreeClassifier(buffer_size=4).partial_fit(self.X,self.y)
        old=t.predict(self.X);t.partial_fit([[100,100]],[0]);np.testing.assert_array_equal(old,t.predict(self.X))
    def test_41_tree_nan(self):
        X=self.X.copy();X[0,0]=np.nan
        self.assertEqual(len(DecisionTreeClassifier().fit(X,self.y).predict(X)),4)
    def test_42_tree_entropy(self):self.assertEqual(len(DecisionTreeClassifier(criterion="entropy").fit(self.X,self.y).predict(self.X)),4)
    def test_43_tree_shape(self):
        with self.assertRaises(ValueError):DecisionTreeClassifier().fit(self.X,[1])
    def test_44_pipeline(self):
        m=Pipeline([('imp',Imputer()),('tree',DecisionTreeClassifier(buffer_size=4))])
        m.partial_fit(self.X,self.y);self.assertEqual(len(m.predict(self.X)),4)
    def test_45_stream(self):
        s=StreamTrainer(DecisionTreeClassifier(buffer_size=2));s.fit_chunk(self.X[:2],self.y[:2]);s.fit_chunk(self.X[2:],self.y[2:]);self.assertEqual(len(s.history),2)
    def test_46_noncontiguous(self):
        X=np.arange(40,dtype=float).reshape(10,4)[:,::2]
        self.assertEqual(StandardScaler().fit_transform(X).shape,(10,2))
    def test_47_roc(self):self.assertAlmostEqual(auc(*roc_curve([0,1],[.1,.9])[:2]),1)
    def test_48_tree_empty(self):
        with self.assertRaises(ValueError):DecisionTreeClassifier().fit(np.empty((0,2)),np.array([]))
    def test_49_duplicate_features(self):
        X=np.ones((4,2));self.assertEqual(len(DecisionTreeClassifier().fit(X,self.y).predict(X)),4)
    def test_50_nan_rank(self):self.assertTrue(np.isnan(rank([1,np.nan])[1]))

if __name__=="__main__":unittest.main()
