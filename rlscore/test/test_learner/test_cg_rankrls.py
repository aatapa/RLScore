import random
import unittest

import numpy as np
from scipy.sparse import coo_matrix

from rlscore.learner import CGRankRLS
from rlscore.learner.cg_rankrls import PCGRankRLS
from rlscore.learner import QueryRankRLS


class Test(unittest.TestCase):
    
    def setUp(self):
        np.random.seed(100)
        random.seed(100)
    
    
    def testOrdinalRegression(self):
        m, n = 100, 300
        for regparam in [0.00000001, 1, 100000000]:
        #for regparam in [1000]:
            Xtrain = np.mat(np.random.rand(n, m))
            Y = np.mat(np.random.rand(m, 1))
            rpool = {}
            rpool['X'] = Xtrain.T
            rpool['Y'] = Y
            rpool['regparam'] = regparam
            rpool["bias"] = 1.0
            rls = CGRankRLS(**rpool)
            model = rls.predictor   
            W = model.W
            In = np.mat(np.identity(n))
            Im = np.mat(np.identity(m))
            L = np.mat(Im-(1./m)*np.ones((m,m), dtype=np.float64))
            G = Xtrain*L*Xtrain.T+regparam*In
            W2 = np.squeeze(np.array(G.I*Xtrain*L*Y))
            for i in range(W.shape[0]):
                #for j in range(W.shape[1]):
                #    self.assertAlmostEqual(W[i,j],W2[i,j], places=5)
                    self.assertAlmostEqual(W[i], W2[i], places = 5)
    
    
    def testPairwisePreferences(self):
        m, n = 100, 300
        for regparam in [0.00000001, 1, 100000000]:
            Xtrain = np.mat(np.random.rand(n, m))
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
            rpool['X'] = Xtrain.T
            #rpool['Y'] = Y
            rpool['train_preferences'] = pairs
            rpool['regparam'] = regparam
            rpool["bias"] = 1.0
            rls = PCGRankRLS(**rpool)
            model = rls.predictor   
            W = model.W
            In = np.mat(np.identity(n))
            Im = np.mat(np.identity(m))
            vals = np.concatenate([np.ones((pairs.shape[0]), dtype=np.float64), -np.ones((pairs.shape[0]), dtype=np.float64)])
            row = np.concatenate([np.arange(pairs.shape[0]),np.arange(pairs.shape[0])])
            col = np.concatenate([pairs[:,0], pairs[:,1]])
            coo = coo_matrix((vals, (row, col)), shape=(pairs.shape[0], Xtrain.T.shape[0]))
            L = (coo.T*coo).todense()
            G = Xtrain*L*Xtrain.T+regparam*In
            W2 = np.squeeze(np.array(G.I*Xtrain*coo.T*np.mat(np.ones((pairs.shape[0],1)))))
            for i in range(W.shape[0]):
                #for j in range(W.shape[1]):
                #    self.assertAlmostEqual(W[i,j],W2[i,j], places=4)
                    self.assertAlmostEqual(W[i], W2[i], places=4)
                    
    def testQueryData(self):
        np.random.seed(100)
        floattype = np.float64
        m, n = 100, 400 #data, features
        Xtrain = np.mat(np.random.rand(m, n))
        Y = np.mat(np.zeros((m, 1), dtype=floattype))
        Y[:, 0] = np.sum(Xtrain, 1)
        qidlist = [0 for i in range(100)]
        for h in range(5, 12):
            qidlist[h] = 1
        for h in range(12, 32):
            qidlist[h] = 2
        for h in range(32, 34):
            qidlist[h] = 3
        for h in range(34, 85):
            qidlist[h] = 4
        for h in range(85, 100):
            qidlist[h] = 5
        kwargs = {}
        kwargs['X'] = Xtrain
        kwargs['Y'] = Y
        kwargs['qids'] = qidlist
        kwargs['regparam'] = 1.0
        learner1 = QueryRankRLS(**kwargs)
        learner2 = CGRankRLS(**kwargs)
        mdiff = np.max(1. - learner1.predictor.W / learner2.predictor.W)
        if mdiff > 0.01:
            assert False