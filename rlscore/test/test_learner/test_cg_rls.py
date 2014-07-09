import unittest

import numpy as np

from rlscore.learner import CGRLS
from rlscore.learner import RLS
from rlscore.kernel import LinearKernel



class Test(unittest.TestCase):
    
    def setUp(self):
        np.random.seed(100)

    
    def testCGRLS(self):
        m, n = 100, 300
        for regparam in [0.00000001, 1, 100000000]:
            Xtrain = np.mat(np.random.rand(m, n))
            Y = np.mat(np.random.rand(m, 1))
            rpool = {}
            rpool['train_features'] = Xtrain
            rpool['train_labels'] = Y
            rpool['regparam'] = regparam
            rpool["bias"] = 1.0
            rls = RLS.createLearner(**rpool)
            rls.solve(regparam)
            model = rls.getModel()
            W = model.W
            b = model.b
            rls = CGRLS.createLearner(**rpool)
            rls.train()
            model = rls.getModel()
            W2 = model.W
            b2 = model.b
            for i in range(W.shape[0]):
                #for j in range(W.shape[1]):
                #    self.assertAlmostEqual(W[i,j],W2[i,j],places=5)
                    self.assertAlmostEqual(W[i], W2[i], places=5)
            self.assertAlmostEqual(b, b2, places=5)