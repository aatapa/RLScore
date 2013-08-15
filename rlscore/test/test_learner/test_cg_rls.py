import unittest

import numpy as np

from rlscore.learner import CGRLS
from rlscore.learner import RLS
from rlscore import data_sources
from rlscore.kernel import LinearKernel



class Test(unittest.TestCase):
    
    def setUp(self):
        np.random.seed(100)

    
    def testCGRLS(self):
        m, n = 100, 300
        for regparam in [0.00000001, 1, 100000000]:
            Xtrain = np.mat(np.random.rand(n, m))
            Y = np.mat(np.random.rand(m, 1))
            rpool = {}
            rpool[data_sources.TRAIN_FEATURES] = Xtrain.T
            rpool[data_sources.TRAIN_LABELS] = Y
            rpool[data_sources.TIKHONOV_REGULARIZATION_PARAMETER] = regparam
            rpool["bias"] = 1.0
            k = LinearKernel.createKernel(**rpool)
            rpool[data_sources.KERNEL_OBJ] = k
            rls = RLS.createLearner(**rpool)
            rls.solve(regparam)
            model = rls.getModel()
            W = model.W
            b = model.b
            rls = CGRLS.createLearner(**rpool)
            rls.solve(regparam)
            model = rls.getModel()
            W2 = model.W
            b2 = model.b
            for i in range(W.shape[0]):
                for j in range(W.shape[1]):
                    self.assertAlmostEqual(W[i,j],W2[i,j],places=5)
            for i in range(b.shape[0]):
                for j in range(b.shape[1]):
                    self.assertAlmostEqual(b[i,j],b2[i,j],places=5)