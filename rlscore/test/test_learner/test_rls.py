import unittest

import numpy as np
from numpy.testing import assert_allclose
import numpy.linalg as la

from rlscore.learner import RLS
from rlscore.kernel import LinearKernel
from rlscore.kernel import GaussianKernel

class Test(unittest.TestCase):
    
    def setUp(self):
        np.random.seed(100)
        m, n = 100, 50
        self.Xtrain1 = np.random.rand(m, n)
        self.Xtrain2 = np.random.rand(m, 120)
        self.Ytrain1 = np.random.randn(m)
        self.Ytrain2 = np.random.randn(m, 5)
        self.bvectors = [0,3,5,22,44]
        
    def test_linear(self):
        #Test that learning with linear kernel works correctly both
        #with low and high-dimensional data
        for X in [self.Xtrain1, self.Xtrain2]:
            for Y in [self.Ytrain1, self.Ytrain2]:
                #Basic case
                primal_rls = RLS(X, Y, regparam=1.0, bias=0.)
                W = primal_rls.predictor.W
                d = X.shape[1]
                W2 = np.linalg.solve(np.dot(X.T, X) + np.eye(d), np.dot(X.T, Y))
                assert_allclose(W, W2)
                #Fast regularization algorithm
                primal_rls.solve(10.)
                W = primal_rls.predictor.W
                W2 = np.linalg.solve(np.dot(X.T, X) + 10.*np.eye(d), np.dot(X.T, Y))
                assert_allclose(W, W2)
                #Bias term included
                primal_rls = RLS(X, Y, regparam=1.0, bias=2.)
                O = np.sqrt(2.) * np.ones((X.shape[0],1))
                X_new = np.hstack((X, O))
                W = primal_rls.predictor.W
                W2 = np.linalg.solve(np.dot(X_new.T, X_new) + np.eye(d+1), np.dot(X_new.T, Y))
                b = primal_rls.predictor.b
                b2 = W2[-1]
                W2 = W2[:-1]
                assert_allclose(W, W2)
                assert_allclose(b, np.sqrt(2) * b2)
                #reduced set approximation
                primal_rls = RLS(X, Y, basis_vectors = X[self.bvectors], regparam=5.0, bias=2.)
                W = primal_rls.predictor.W
                b = primal_rls.predictor.b
                K = np.dot(X_new, X_new.T)
                Kr = K[:, self.bvectors]
                Krr = K[np.ix_(self.bvectors, self.bvectors)]
                A = np.linalg.solve(np.dot(Kr.T, Kr)+ 5.0 * Krr, np.dot(Kr.T, Y))
                W2 = np.dot(X_new[self.bvectors].T, A)
                b2 = W2[-1]
                W2 = W2[:-1]
                assert_allclose(W, W2)
                assert_allclose(b, np.sqrt(2) * b2)
                #Using pre-computed linear kernel matrix
                kernel = LinearKernel(X, bias = 2.)
                K = kernel.getKM(X)
                dual_rls = RLS(K, Y, kernel = "PrecomputedKernel", regparam=0.01)
                W = np.dot(X_new.T, dual_rls.predictor.W)
                b = W[-1]
                W = W[:-1]
                W2 = np.linalg.solve(np.dot(X_new.T, X_new) + 0.01 * np.eye(d+1), np.dot(X_new.T, Y))
                b2 = W2[-1]
                W2 = W2[:-1]
                assert_allclose(W, W2)
                assert_allclose(b, b2)
                #Pre-computed linear kernel, reduced set approximation
                kernel = LinearKernel(X[self.bvectors], bias = 2.)
                dual_rls = RLS(kernel.getKM(X), Y, kernel="PrecomputedKernel", basis_vectors = kernel.getKM(X[self.bvectors]), regparam=5.0)
                W = np.dot(X_new[self.bvectors].T, dual_rls.predictor.W)
                b = W[-1]
                W = W[:-1]
                K = np.dot(X_new, X_new.T)
                Kr = K[:, self.bvectors]
                Krr = K[np.ix_(self.bvectors, self.bvectors)]
                A = np.linalg.solve(np.dot(Kr.T, Kr)+ 5.0 * Krr, np.dot(Kr.T, Y))
                W2 = np.dot(X_new[self.bvectors].T, A)
                b2 = W2[-1]
                W2 = W2[:-1]
                assert_allclose(W, W2)
                assert_allclose(b, b2)
                
                
                
                
                
    def test_kernel(self):
        for X in [self.Xtrain1, self.Xtrain2]:
            for Y in [self.Ytrain1, self.Ytrain2]:
                pass
        
        
    def testRLS(self):
        
        print
        print
        print
        print
        print("Testing the cross-validation routines of the RLS module.")
        print
        print
        floattype = np.float64
        
        m, n = 400, 100
        Xtrain = np.random.rand(m, n)
        K = np.dot(Xtrain, Xtrain.T)
        ylen = 2
        Y = np.zeros((m, ylen), dtype=floattype)
        Y = np.random.rand(m, ylen)
        
        hoindices = [45]
        hoindices2 = [45, 50]
        hoindices3 = [45, 50, 55]
        hocompl = list(set(range(m)) - set(hoindices))
        
        Kho = K[np.ix_(hocompl, hocompl)]
        Yho = Y[hocompl]
        
        kwargs = {}
        kwargs['Y'] = Y
        kwargs['X'] = K
        kwargs['kernel'] = 'PrecomputedKernel'
        dualrls = RLS(**kwargs)
        
        kwargs = {}
        kwargs["X"] = Xtrain
        kwargs["Y"] = Y
        kwargs["bias"] = 0.
        primalrls = RLS(**kwargs)
        
        kwargs = {}
        kwargs['Y'] = Yho
        kwargs['X'] = Kho
        kwargs['kernel'] = 'PrecomputedKernel'
        dualrls_naive = RLS(**kwargs)
        
        testkm = K[np.ix_(hocompl, hoindices)]
        trainX = Xtrain[hocompl]
        testX = Xtrain[hoindices]
        kwargs = {}
        kwargs['Y'] = Yho
        kwargs['X'] = trainX
        kwargs["bias"] = 0.
        primalrls_naive = RLS(**kwargs)
        
        loglambdas = range(-5, 5)
        for j in range(0, len(loglambdas)):
            regparam = 2. ** loglambdas[j]
            print
            print("Regparam 2^%1d" % loglambdas[j])
            
            dumbho = np.dot(testkm.T, np.dot(la.inv(Kho + regparam * np.eye(Kho.shape[0])), Yho))
            print(str(dumbho) + ' Dumb HO (dual)')
            
            dualrls_naive.solve(regparam)
            predho1 = dualrls_naive.predictor.predict(testkm.T)
            print(str(predho1) + ' Naive HO (dual)')
            
            dualrls.solve(regparam)
            predho2 = dualrls.holdout(hoindices)
            print(str(predho2) + ' Fast HO (dual)')
            
            dualrls.solve(regparam)
            predho = dualrls.leave_one_out()[hoindices[0]]
            print(str(predho) + ' Fast LOO (dual)')
            
            primalrls_naive.solve(regparam)
            predho3 = primalrls_naive.predictor.predict(testX)
            print(str(predho3) + ' Naive HO (primal)')
            
            primalrls.solve(regparam)
            predho4 = primalrls.holdout(hoindices)
            print(str(predho4) + ' Fast HO (primal)')
            for predho in [predho1, predho2, predho3, predho4]:
                self.assertEqual(dumbho.shape, predho.shape)
                for row in range(predho.shape[0]):
                    for col in range(predho.shape[1]):
                        self.assertAlmostEqual(dumbho[row,col],predho[row,col])
            primalrls.solve(regparam)
            predho = primalrls.leave_one_out()[hoindices[0]]
            print(str(predho) + ' Fast LOO (primal)')
        print()
        hoindices = range(100, 300)
        hocompl = list(set(range(m)) - set(hoindices))
        
        Kho = K[np.ix_(hocompl, hocompl)]
        Yho = Y[hocompl]
        testkm = K[np.ix_(hocompl, hoindices)]
        
        dumbho = np.dot(testkm.T, np.dot(la.inv(Kho + regparam * np.eye(Kho.shape[0])), Yho))
        
        kwargs = {}
        kwargs['Y'] = Y
        kwargs['X'] = Xtrain
        dualrls.solve(regparam)
        predho2 = dualrls.holdout(hoindices2)
        print(str(predho2) + ' Fast HO')
        hopred = dualrls.leave_pair_out(np.array([hoindices2[0], 4, 6]), np.array([hoindices2[1], 5, 7]))
        print(str(hopred[0][0]) + '\n' + str(hopred[1][0]) + ' Fast LPO')
        
