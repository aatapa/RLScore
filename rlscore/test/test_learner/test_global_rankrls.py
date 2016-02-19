import unittest
from numpy.testing import assert_allclose
import numpy as np
import numpy.linalg as la

from rlscore.learner import GlobalRankRLS

class Test(unittest.TestCase):
    
    def setUp(self):
        np.random.seed(100)
        m= 30
        self.Xtrain1 = np.random.rand(m, 20)
        self.Xtrain2 = np.random.rand(m, 40)
        self.Ytrain1 = np.random.randn(m)
        self.Ytrain2 = np.random.randn(m, 5)
        self.bvectors = [0,3,5,22]
        
    def test_linear(self):
        #Test that learning with linear kernel works correctly both
        #with low and high-dimensional data
        for X in [self.Xtrain1, self.Xtrain2]:
            for Y in [self.Ytrain1, self.Ytrain2]:
                #Basic case
                m = X.shape[0]
                L = m * np.eye(m) -  np.ones((m,m))
                primal_rls = GlobalRankRLS(X, Y, regparam=1.0, bias=0.)
                W = primal_rls.predictor.W
                d = X.shape[1]
                W2 = np.linalg.solve(np.dot(X.T, np.dot(L, X)) + np.eye(d), np.dot(X.T, np.dot(L, Y)))
                assert_allclose(W, W2)
                #For RankRLS, bias should have no effect
                primal_rls = GlobalRankRLS(X, Y, regparam=1.0, bias=5.)
                W2 = primal_rls.predictor.W
                assert_allclose(W, W2)
                #Fast regularization
                primal_rls.solve(10)
                W = primal_rls.predictor.W
                W2 = np.linalg.solve(np.dot(X.T, np.dot(L, X)) + 10 * np.eye(d), np.dot(X.T, np.dot(L, Y)))
                assert_allclose(W, W2)
                #reduced set approximation
                primal_rls = GlobalRankRLS(X, Y, basis_vectors = X[self.bvectors], regparam=5.0)
                W = primal_rls.predictor.W
                K = np.dot(X, X.T)
                Kr = K[:, self.bvectors]
                Krr = K[np.ix_(self.bvectors, self.bvectors)]
                A = np.linalg.solve(np.dot(Kr.T, np.dot(L, Kr))+ 5.0 * Krr, np.dot(Kr.T, np.dot(L, Y)))
                W2 = np.dot(X[self.bvectors].T, A)
                assert_allclose(W, W2)
                #Precomputed kernel matrix
                dual_rls = GlobalRankRLS(K, Y, kernel = "PrecomputedKernel", regparam=0.01)
                W = np.dot(X.T, dual_rls.predictor.W)
                W2 = np.linalg.solve(np.dot(X.T, np.dot(L, X)) + 0.01 * np.eye(d), np.dot(X.T, np.dot(L, Y)))
                assert_allclose(W, W2)
                #Pre-computed linear kernel, reduced set approximation
                dual_rls = GlobalRankRLS(Kr, Y, kernel="PrecomputedKernel", basis_vectors = Krr, regparam=5.0)
                W = np.dot(X[self.bvectors].T, dual_rls.predictor.W)
                A = np.linalg.solve(np.dot(Kr.T, np.dot(L, Kr))+ 5.0 * Krr, np.dot(Kr.T, np.dot(L, Y)))
                W2 = np.dot(X[self.bvectors].T, A)
                assert_allclose(W, W2)
    
    def testAllPairsRankRLS(self):
        
        print("Testing the cross-validation routines of the GlobalRankRLS module.\n\n")
        
        np.random.seed(100)
        floattype = np.float64
        
        m, n, h = 30, 200, 10
        Xtrain = np.random.rand(m, n)
        trainlabels = np.random.rand(m, h)
        trainkm = np.dot(Xtrain, Xtrain.T)
        ylen = 1
        
        L = np.mat(m * np.eye(m) - np.ones((m, m), dtype=floattype))
        
        hoindices = [5, 7]
        hoindices3 = [5, 7, 9]
        hocompl = list(set(range(m)) - set(hoindices))
        hocompl3 = list(set(range(m)) - set(hoindices3))
        
        loglambdas = range(-5, 5)
        for j in range(0, len(loglambdas)):
            regparam = 2. ** loglambdas[j]
            print("\nRegparam 2^%1d" % loglambdas[j])
            
            Kcv = trainkm[np.ix_(hocompl, hocompl)]
            Ycv = trainlabels[hocompl]
            Ktest = trainkm[np.ix_(hocompl, hoindices)]
            
            Xcv = Xtrain[hocompl]
            Xtest = Xtrain[hoindices]
            
            Lcv = np.mat((m - 2) * np.eye(m - 2) - np.ones((m - 2, m - 2), dtype=floattype))
            
            oind = 1
            rpool = {}
            rpool['Y'] = Ycv
            rpool['X'] = Xcv
            rpool['regparam'] = regparam
            naivedualrls = GlobalRankRLS(**rpool)
            naivedualrls.solve(regparam)
            hopreds = []
            
            hopred = naivedualrls.predictor.predict(Xtest)
            print(str(hopred[0, oind]) + ' ' + str(hopred[1, oind]) + ' Naive')
            hopreds.append((hopred[0, oind], hopred[1, oind]))
            
            rpool = {}
            rpool['Y'] = trainlabels
            rpool['X'] = Xtrain
            rpool['regparam'] = regparam
            hodualrls = GlobalRankRLS(**rpool)
            hodualrls.solve(regparam)
            hopred_start, hopred_end = hodualrls.leave_pair_out(np.array([hoindices[0], 1, 1, 2]), np.array([hoindices[1], 2, 3, 3]))
            print(str(hopred_start[0][1]) + ' ' + str(hopred_end[0][1]) + ' Fast')
            hopreds.append((hopred_start[0][1], hopred_end[0][1]))
            self.assertAlmostEqual(hopreds[0][0], hopreds[1][0])
            self.assertAlmostEqual(hopreds[0][1], hopreds[1][1])
            
            #Test with single output just in case.
            rpool = {}
            rpool['Y'] = trainlabels[:, 1]
            rpool['X'] = Xtrain
            rpool['regparam'] = regparam
            hodualrls = GlobalRankRLS(**rpool)
            hodualrls.solve(regparam)
            hopred_start, hopred_end = hodualrls.leave_pair_out(np.array([hoindices[0], 1, 1, 2]), np.array([hoindices[1], 2, 3, 3]))
            #print(str(hopred_start[0]) + ' ' + str(hopred_end[0]) + ' Fast')
            
            #Test with single output just in case.
            rpool = {}
            rpool['Y'] = trainlabels[:, 1]
            rpool['X'] = Xtrain
            rpool['regparam'] = regparam
            hodualrls = GlobalRankRLS(**rpool)
            hodualrls.solve(regparam)
            hopred_start, hopred_end = hodualrls.leave_pair_out(np.array([hoindices[0]]), np.array([hoindices[1]]))
            #print(str(hopred_start) + ' ' + str(hopred_end) + ' Fast')
            
            hopreds = []
            
            rpool = {}
            rpool['Y'] = trainlabels
            rpool['X'] = Xtrain
            rpool['regparam'] = regparam
            hoprimalrls = GlobalRankRLS(**rpool)
            hoprimalrls.solve(regparam)
            hopred = hoprimalrls.holdout(hoindices)
            print(str(hopred[0, oind]) + ' ' + str(hopred[1, oind]) + ' HO')
            hopreds.append((hopred[0, oind], hopred[1, oind]))
            
            hopred = Xtest * la.inv(Xcv.T * Lcv * Xcv + regparam * np.mat(np.eye(n))) * Xcv.T * Lcv * Ycv
            print(str(hopred[0, oind]) + ' ' + str(hopred[1, oind]) + ' Dumb (primal)')
            hopreds.append((hopred[0, oind], hopred[1, oind]))
            
            hopred0 = hopreds.pop(0)
            for hopred in hopreds:
                self.assertAlmostEqual(hopred0[0],hopred[0])
                self.assertAlmostEqual(hopred0[1],hopred[1])
