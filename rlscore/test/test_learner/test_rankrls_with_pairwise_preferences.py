import random
import unittest

import numpy as np
from scipy.sparse import coo_matrix

from rlscore.learner.rankrls_with_pairwise_preferences import PPRankRLS
from rlscore.kernel import GaussianKernel



class Test(unittest.TestCase):
    
    def setUp(self):
        np.random.seed(100)
        random.seed(100)
    
    
    def testPairwisePreferences(self):
        m, n = 100, 300
        Xtrain = np.mat(np.random.rand(m, n))
        Xtest = np.mat(np.random.rand(5, n))
        for regparam in [0.00000001, 1, 100000000]:
            Y = np.mat(np.random.rand(m, 1))
            pairs = []
            for i in range(1000):
                a = random.randint(0, m - 1)
                b = random.randint(0, m - 1)
                if Y[a] > Y[b]:
                    pairs.append((a, b))
                else:
                    pairs.append((b, a))
            pairs = np.array(pairs)
            rpool = {}
            rpool['X'] = Xtrain
            rpool["pairs_start_inds"] = pairs[:,0]
            rpool["pairs_end_inds"] = pairs[:,1]
            rpool['regparam'] = regparam
            rpool["bias"] = 1.0
            rpool["kernel"] = "GaussianKernel"
            ker = GaussianKernel(Xtrain, 1.0)
            trainkm = ker.getKM(Xtrain)
            rls = PPRankRLS(**rpool)
            model = rls.predictor   
            P1 = model.predict(Xtest)
            Im = np.mat(np.identity(m))
            vals = np.concatenate([np.ones((pairs.shape[0]), dtype = np.float64), -np.ones((pairs.shape[0]), dtype = np.float64)])
            row = np.concatenate([np.arange(pairs.shape[0]), np.arange(pairs.shape[0])])
            col = np.concatenate([pairs[:, 0], pairs[:, 1]])
            coo = coo_matrix((vals, (row, col)), shape = (pairs.shape[0], Xtrain.shape[0]))
            L = (coo.T * coo).todense()
            P2 = np.dot(ker.getKM(Xtest), np.mat((L * trainkm + regparam * Im).I * coo.T * np.mat(np.ones((pairs.shape[0], 1)))))
            for i in range(P1.shape[0]):
                    self.assertAlmostEqual(P1[i], P2[i,0], places = 3)