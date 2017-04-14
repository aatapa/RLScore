import unittest

import numpy as np
from scipy import sparse

from rlscore.learner.rls import RLS
from rlscore import predictor as mod


class Test(unittest.TestCase):
    
    
    def setUp(self):
        np.random.seed(100)
    
    
    def testModel(self):
        
        Y = np.random.random((10))
        X = np.random.random((10,100))
        kwargs = {}
        kwargs["Y"] = Y
        kwargs["X"] = X
        kwargs["regparam"] = 1
        learner = RLS(**kwargs)
        model = learner.predictor
        print
        #print 'Ten data points, single label '
        model = mod.LinearPredictor(np.random.random((100)))
        self.all_pred_cases(model)
        
        model = mod.LinearPredictor(np.random.random((100, 2)))
        self.all_pred_cases(model)
        
        #model = mod.LinearPredictor(np.random.random((1, 2)))
        #self.all_pred_cases(model)
        
        kwargs["kernel"] = "GaussianKernel"
        Y = np.random.random((10))
        kwargs["Y"] = Y
        learner = RLS(**kwargs)
        model = learner.predictor
        self.all_pred_cases(model)
        
        kwargs["kernel"] = "GaussianKernel"
        Y = np.random.random((10,2))
        kwargs["Y"] = Y
        learner = RLS(**kwargs)
        model = learner.predictor
        self.all_pred_cases(model)
        #test_perf = cindex(test_labels, P)
        #print "test set performance: %f" %test_perf
    
    
    def all_pred_cases(self, model, fcount = 100):
        print
        test_features = np.random.random((10,100))
        P = model.predict(test_features)
        print(type(test_features), type(P), test_features.shape, P.shape)
        test_features = np.random.random((1,100))
        P = model.predict(test_features)
        print(type(test_features), type(P), test_features.shape, P.shape)
        test_features = np.random.random((100))
        P = model.predict(test_features)
        print(type(test_features), type(P), test_features.shape, P.shape)
        test_features = np.mat(np.random.random((10,100)))
        P = model.predict(test_features)
        print(type(test_features), type(P), test_features.shape, P.shape)
        test_features = np.mat(np.random.random((1,100)))
        P = model.predict(test_features)
        print(type(test_features), type(P), test_features.shape, P.shape)
        test_features = sparse.coo_matrix(([1,2,3,1,1,3], ([0, 1, 2, 3, 4, 6], [0,1,2,3,5,6])), (10, 100), dtype = np.float64)
        P = model.predict(test_features)
        print(type(test_features), type(P), test_features.shape, P.shape)
        test_features = sparse.coo_matrix(([1,2,3,1,1,3], ([0, 0, 0, 0, 0, 0], [0, 1, 2, 3, 5, 6])), (1, 100), dtype = np.float64)
        P = model.predict(test_features)
        print(type(test_features), type(P), test_features.shape, P.shape)


if __name__=="__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(Test)
    unittest.TextTestRunner(verbosity=2).run(suite)
