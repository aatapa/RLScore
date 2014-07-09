import unittest

import numpy as np
from scipy import sparse

from rlscore.learner.rls import RLS
from rlscore.reader import read_sparse
from rlscore import model as mod
from rlscore.utilities import array_tools


class Test(unittest.TestCase):
    
    
    def setUp(self):
        np.random.seed(100)
    
    
    def testModel(self):
        
        train_labels = np.random.random((10))
        test_labels = np.random.random((10))
        train_features = np.random.random((10,100))
        test_features = np.random.random((10,100))
        kwargs = {}
        kwargs["train_labels"] = train_labels
        kwargs["train_features"] = train_features
        kwargs["regparam"] = 1
        learner = RLS.createLearner(**kwargs)
        learner.train()
        model = learner.getModel()
        print
        #print 'Ten data points, single label '
        model = mod.LinearModel(np.random.random((100)))
        self.all_pred_cases(model)
        
        model = mod.LinearModel(np.random.random((100, 2)))
        self.all_pred_cases(model)
        
        #model = mod.LinearModel(np.random.random((1, 2)))
        #self.all_pred_cases(model)
        
        kwargs["kernel"] = "GaussianKernel"
        train_labels = np.random.random((10))
        kwargs["train_labels"] = train_labels
        learner = RLS.createLearner(**kwargs)
        learner.train()
        model = learner.getModel()
        self.all_pred_cases(model)
        
        kwargs["kernel"] = "GaussianKernel"
        train_labels = np.random.random((10,2))
        kwargs["train_labels"] = train_labels
        learner = RLS.createLearner(**kwargs)
        learner.train()
        model = learner.getModel()
        self.all_pred_cases(model)
        #test_perf = cindex(test_labels, P)
        #print "test set performance: %f" %test_perf
    
    
    def all_pred_cases(self, model, fcount = 100):
        print
        test_features = np.random.random((10,100))
        P = model.predict(test_features)
        print type(test_features), type(P), test_features.shape, P.shape
        test_features = np.random.random((1,100))
        P = model.predict(test_features)
        print type(test_features), type(P), test_features.shape, P.shape
        test_features = np.random.random((100))
        P = model.predict(test_features)
        print type(test_features), type(P), test_features.shape, P.shape
        test_features = np.mat(np.random.random((10,100)))
        P = model.predict(test_features)
        print type(test_features), type(P), test_features.shape, P.shape
        test_features = np.mat(np.random.random((1,100)))
        P = model.predict(test_features)
        print type(test_features), type(P), test_features.shape, P.shape
        test_features = sparse.coo_matrix(([1,2,3,1,1,3], ([0, 1, 2, 3, 4, 6], [0,1,2,3,5,6])), (10, 100), dtype = np.float64)
        P = model.predict(test_features)
        print type(test_features), type(P), test_features.shape, P.shape
        test_features = sparse.coo_matrix(([1,2,3,1,1,3], ([0, 0, 0, 0, 0, 0], [0, 1, 2, 3, 5, 6])), (1, 100), dtype = np.float64)
        P = model.predict(test_features)
        print type(test_features), type(P), test_features.shape, P.shape


if __name__=="__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(Test)
    unittest.TextTestRunner(verbosity=2).run(suite)
