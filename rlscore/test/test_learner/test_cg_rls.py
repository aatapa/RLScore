import unittest

import numpy as np

from rlscore.learner import CGRLS
from rlscore.learner import RLS



class Test(unittest.TestCase):
    
    def setUp(self):
        np.random.seed(100)

    
    def testCGRLS(self):
        m, n = 100, 300
        for regparam in [0.00000001, 1, 100000000]:
            Xtrain = np.mat(np.random.rand(m, n))
            Y = np.mat(np.random.rand(m, 1))
            rpool = {}
            rpool['X'] = Xtrain
            rpool['Y'] = Y
            rpool['regparam'] = regparam
            rpool["bias"] = 2.0
            rls = RLS(**rpool)
            rls.solve(regparam)
            model = rls.predictor
            W = model.W
            b = model.b
            rls = CGRLS(**rpool)
            model = rls.predictor
            W2 = model.W
            b2 = model.b
            for i in range(W.shape[0]):
                    self.assertAlmostEqual(W[i], W2[i], places=5)
            self.assertAlmostEqual(b, b2, places=5)