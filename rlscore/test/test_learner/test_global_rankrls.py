import unittest
from numpy.testing import assert_allclose
import numpy as np
import numpy.linalg as la

from rlscore.learner import GlobalRankRLS
from rlscore.kernel import GaussianKernel, PolynomialKernel

class Test(unittest.TestCase):
    
    def setUp(self):
        np.random.seed(100)
        m= 30
        self.Xtrain1 = np.random.rand(m, 20)
        self.Xtrain2 = np.random.rand(m, 40)
        self.Ytrain1 = np.random.randn(m)
        self.Ytrain2 = np.random.randn(m, 5)
        self.bvectors = [0,3,5,22]
        
    #@unittest.skip("does not work")          
    def test_linear_subset(self):
        X = self.Xtrain1
        Y = self.Ytrain1
        m = X.shape[0]
        L = m * np.eye(m) -  np.ones((m,m))
        #reduced set approximation
        primal_rls = GlobalRankRLS(X, Y, basis_vectors = X[self.bvectors], regparam=5.0)
        W = primal_rls.predictor.W
        K = np.dot(X, X.T)
        Kr = K[:, self.bvectors]
        Krr = K[np.ix_(self.bvectors, self.bvectors)]
        A = np.linalg.solve(np.dot(Kr.T, np.dot(L, Kr))+ 5.0 * Krr, np.dot(Kr.T, np.dot(L, Y)))
        W_reduced = np.dot(X[self.bvectors].T, A)
        assert_allclose(W, W_reduced)
        
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
                #primal_rls = GlobalRankRLS(X, Y, basis_vectors = X[self.bvectors], regparam=5.0)
                #W = primal_rls.predictor.W
                K = np.dot(X, X.T)
                Kr = K[:, self.bvectors]
                Krr = K[np.ix_(self.bvectors, self.bvectors)]
                A = np.linalg.solve(np.dot(Kr.T, np.dot(L, Kr))+ 5.0 * Krr, np.dot(Kr.T, np.dot(L, Y)))
                W_reduced = np.dot(X[self.bvectors].T, A)
                #assert_allclose(W, W_reduced)
                #Pre-computed linear kernel, reduced set approximation
                dual_rls = GlobalRankRLS(Kr, Y, kernel="PrecomputedKernel", basis_vectors = Krr, regparam=5.0)
                W = np.dot(X[self.bvectors].T, dual_rls.predictor.W)
                assert_allclose(W, W_reduced)
                #Precomputed kernel matrix
                dual_rls = GlobalRankRLS(K, Y, kernel = "PrecomputedKernel", regparam=0.01)
                W = np.dot(X.T, dual_rls.predictor.W)
                W2 = np.linalg.solve(np.dot(X.T, np.dot(L, X)) + 0.01 * np.eye(d), np.dot(X.T, np.dot(L, Y)))
                assert_allclose(W, W2)
                
    def test_kernel(self):
        #tests that learning with kernels works
        for X in [self.Xtrain1, self.Xtrain2]:
            for Y in [self.Ytrain1, self.Ytrain2]:
                m = X.shape[0]
                L = m * np.eye(m) -  np.ones((m,m))
                #Basic case
                dual_rls = GlobalRankRLS(X, Y, kernel= "GaussianKernel", regparam=5.0, gamma=0.01)
                kernel = GaussianKernel(X, gamma = 0.01)
                K = kernel.getKM(X)
                m = K.shape[0]
                A = dual_rls.predictor.A
                A2 = np.linalg.solve(np.dot(L, K) +5.0*np.eye(m), np.dot(L, Y) )
                assert_allclose(A, A2)
                #Fast regularization
                dual_rls.solve(1000)
                A = dual_rls.predictor.A
                A2 = np.linalg.solve(np.dot(L, K) + 1000 * np.eye(m), np.dot(L, Y))
                assert_allclose(A, A2)
                #Precomputed kernel
                dual_rls = GlobalRankRLS(K, Y, kernel="PrecomputedKernel", regparam = 1000)
                assert_allclose(dual_rls.predictor.W, A2)
                #Reduced set approximation
                kernel = PolynomialKernel(X[self.bvectors], gamma=0.5, coef0 = 1.2, degree = 2)              
                Kr = kernel.getKM(X)
                Krr = kernel.getKM(X[self.bvectors])
                dual_rls = GlobalRankRLS(X, Y, kernel="PolynomialKernel", basis_vectors = X[self.bvectors], regparam = 200, gamma=0.5, coef0=1.2, degree = 2)
                A = dual_rls.predictor.A
                A2 = np.linalg.solve(np.dot(Kr.T, np.dot(L, Kr))+ 200 * Krr, np.dot(Kr.T, np.dot(L, Y)))
                assert_allclose(A, A2)
                dual_rls = GlobalRankRLS(Kr, Y, kernel="PrecomputedKernel", basis_vectors = Krr, regparam=200)
                A = dual_rls.predictor.W
                assert_allclose(A, A2)
                
    def test_holdout(self):
        for X in [self.Xtrain1, self.Xtrain2]:
            for Y in [self.Ytrain1, self.Ytrain2]:
                m = X.shape[0]
                hoindices = [3, 5, 8, 10, 17, 21]
                hocompl = list(set(range(m)) - set(hoindices))
                #Holdout with linear kernel
                rls1 = GlobalRankRLS(X, Y)
                rls2 = GlobalRankRLS(X[hocompl], Y[hocompl])
                P1 = rls1.holdout(hoindices)
                P2 = rls2.predict(X[hoindices])
                assert_allclose(P1, P2)
                #Holdout with bias
                rls1 = GlobalRankRLS(X, Y, bias = 3.0)
                rls2 = GlobalRankRLS(X[hocompl], Y[hocompl], bias = 3.0)
                P1 = rls1.holdout(hoindices)
                P2 = rls2.predict(X[hoindices])
                assert_allclose(P1, P2)
                #Fast regularization
                for i in range(-5, 5):
                    rls1.solve(2**i)
                    rls2.solve(2**i)
                    P1 = rls1.holdout(hoindices)
                    P2 = rls2.predict(X[hoindices])
                    assert_allclose(P1, P2)
                #Kernel holdout
                rls1 = GlobalRankRLS(X, Y, kernel = "GaussianKernel", gamma = 0.01)
                rls2 = GlobalRankRLS(X[hocompl], Y[hocompl], kernel = "GaussianKernel", gamma = 0.01)
                P1 = rls1.holdout(hoindices)
                P2 = rls2.predict(X[hoindices])
                assert_allclose(P1, P2)
                for i in range(-15, 15):
                    rls1.solve(2**i)
                    rls2.solve(2**i)
                    P1 = rls1.holdout(hoindices)
                    P2 = rls2.predict(X[hoindices])
                    assert_allclose(P1, P2, rtol=1e-06)
                #Incorrect indices
                I = [0, 3, 100]
                self.assertRaises(IndexError, rls1.holdout, I)
                I = [-1, 0, 2]
                self.assertRaises(IndexError, rls1.holdout, I)
                I = [1,1,2]
                self.assertRaises(IndexError, rls1.holdout, I)
                
    def test_leave_pair_out(self):
        #compares holdout and leave-pair-out
        start = [0, 2, 3, 5]
        end = [1, 3, 6, 8]
        for X in [self.Xtrain1, self.Xtrain2]:
            for Y in [self.Ytrain1, self.Ytrain2]:
                #LPO with linear kernel
                rls1 = GlobalRankRLS(X, Y, regparam = 7.0, bias=3.0)
                lpo_start, lpo_end = rls1.leave_pair_out(start, end)
                ho_start, ho_end = [], []
                for i in range(len(start)):
                    P = rls1.holdout([start[i], end[i]])
                    ho_start.append(P[0])
                    ho_end.append(P[1])
                ho_start = np.array(ho_start)
                ho_end = np.array(ho_end)
                assert_allclose(ho_start, lpo_start)
                assert_allclose(ho_end, lpo_end)
                #LPO Gaussian kernel
                rls1 = GlobalRankRLS(X, Y, regparam = 11.0, kenerl="PolynomialKernel", coef0=1, degree=3)
                lpo_start, lpo_end = rls1.leave_pair_out(start, end)
                ho_start, ho_end = [], []
                for i in range(len(start)):
                    P = rls1.holdout([start[i], end[i]])
                    ho_start.append(P[0])
                    ho_end.append(P[1])
                ho_start = np.array(ho_start)
                ho_end = np.array(ho_end)
                assert_allclose(ho_start, lpo_start)
                assert_allclose(ho_end, lpo_end)

    
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
