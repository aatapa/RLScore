import unittest

import numpy as np
from numpy.testing import assert_allclose
import numpy.linalg as la
from scipy.sparse import csc_matrix

from rlscore.learner import RLS
from rlscore.kernel import LinearKernel
from rlscore.kernel import GaussianKernel
from rlscore.kernel import PolynomialKernel

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
        #tests that learning with kernels works
        for X in [self.Xtrain1, self.Xtrain2]:
            for Y in [self.Ytrain1, self.Ytrain2]:
                #Basic case
                dual_rls = RLS(X, Y, kernel= "GaussianKernel", regparam=5.0, gamma=0.01)
                kernel = GaussianKernel(X, gamma = 0.01)
                K = kernel.getKM(X)
                m = K.shape[0]
                A = dual_rls.predictor.A
                A2 = np.linalg.solve(K+5.0*np.eye(m), Y)
                assert_allclose(A, A2)
                #Fast regularization
                dual_rls.solve(1000)
                A = dual_rls.predictor.A
                A2 = np.linalg.solve(K+ 1000 * np.eye(m), Y)
                assert_allclose(A, A2)
                #Precomputed kernel
                dual_rls = RLS(K, Y, kernel="PrecomputedKernel", regparam = 1000)
                assert_allclose(dual_rls.predictor.W, A2)
                #Reduced set approximation
                kernel = PolynomialKernel(X[self.bvectors], gamma=0.5, coef0 = 1.2, degree = 2)              
                Kr = kernel.getKM(X)
                Krr = kernel.getKM(X[self.bvectors])
                dual_rls = RLS(X, Y, kernel="PolynomialKernel", basis_vectors = X[self.bvectors], regparam = 200, gamma=0.5, coef0=1.2, degree = 2)
                A = dual_rls.predictor.A
                A2 = np.linalg.solve(np.dot(Kr.T, Kr)+ 200 * Krr, np.dot(Kr.T, Y))
                assert_allclose(A, A2)
                dual_rls = RLS(Kr, Y, kernel="PrecomputedKernel", basis_vectors = Krr, regparam=200)
                A = dual_rls.predictor.W
                assert_allclose(A, A2)
                
    def test_sparse(self):
        #mix of linear and kernel learning testing that learning works also with
        #sparse data matrices
        for X in [self.Xtrain1, self.Xtrain2]:
            for Y in [self.Ytrain1, self.Ytrain2]:
                Xsp = csc_matrix(X)
                #linear kernel without bias
                primal_rls = RLS(Xsp, Y, regparam=2.0, bias=0.)
                W = primal_rls.predictor.W
                d = X.shape[1]
                W2 = np.linalg.solve(np.dot(X.T, X) + 2.0 * np.eye(d), np.dot(X.T, Y))
                assert_allclose(W, W2)
                #linear kernel with bias
                primal_rls = RLS(Xsp, Y, regparam=1.0, bias=2.)
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
                primal_rls = RLS(Xsp, Y, basis_vectors = Xsp[self.bvectors], regparam=5.0, bias=2.)
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
                #Kernels 
                dual_rls = RLS(Xsp, Y, kernel= "GaussianKernel", regparam=5.0, gamma=0.01)
                kernel = GaussianKernel(X, gamma = 0.01)
                K = kernel.getKM(X)
                m = K.shape[0]
                A = dual_rls.predictor.A
                A2 = np.linalg.solve(K+5.0*np.eye(m), Y)
                assert_allclose(A, A2)
                
    def test_holdout(self):
        for X in [self.Xtrain1, self.Xtrain2]:
            for Y in [self.Ytrain1, self.Ytrain2]:
                m = X.shape[0]
                hoindices = [3, 5, 8, 10, 17, 21]
                hocompl = list(set(range(m)) - set(hoindices))
                #Holdout with linear kernel
                rls1 = RLS(X, Y)
                rls2 = RLS(X[hocompl], Y[hocompl])
                P1 = rls1.holdout(hoindices)
                P2 = rls2.predict(X[hoindices])
                assert_allclose(P1, P2)
                #Holdout with bias
                rls1 = RLS(X, Y, bias = 3.0)
                rls2 = RLS(X[hocompl], Y[hocompl], bias = 3.0)
                P1 = rls1.holdout(hoindices)
                P2 = rls2.predict(X[hoindices])
                assert_allclose(P1, P2)
                #Fast regularization
                for i in range(-15, 15):
                    rls1.solve(2**i)
                    rls2.solve(2**i)
                    P1 = rls1.holdout(hoindices)
                    P2 = rls2.predict(X[hoindices])
                    assert_allclose(P1, P2)
                #Kernel holdout
                rls1 = RLS(X, Y, kernel = "GaussianKernel", gamma = 0.01)
                rls2 = RLS(X[hocompl], Y[hocompl], kernel = "GaussianKernel", gamma = 0.01)
                P1 = rls1.holdout(hoindices)
                P2 = rls2.predict(X[hoindices])
                assert_allclose(P1, P2)
                for i in range(-15, 15):
                    rls1.solve(2**i)
                    rls2.solve(2**i)
                    P1 = rls1.holdout(hoindices)
                    P2 = rls2.predict(X[hoindices])
                    assert_allclose(P1, P2)
                #Incorrect indices
                I = [0, 3, 100]
                self.assertRaises(IndexError, rls1.holdout, I)
                I = [-1, 0, 2]
                self.assertRaises(IndexError, rls1.holdout, I)
                I = [1,1,2]
                self.assertRaises(IndexError, rls1.holdout, I)
                
    def test_loo(self):
        for X in [self.Xtrain1, self.Xtrain2]:
            for Y in [self.Ytrain1, self.Ytrain2]:
                m = X.shape[0]
                #LOO with linear kernel
                rls1 = RLS(X, Y, regparam = 7.0, bias=3.0)
                P1 = rls1.leave_one_out()
                P2 = []
                for i in range(X.shape[0]): 
                    X_train = np.delete(X, i, axis=0)
                    Y_train = np.delete(Y, i, axis=0)
                    X_test = X[i]
                    rls2 = RLS(X_train, Y_train, regparam = 7.0, bias = 3.0)
                    P2.append(rls2.predict(X_test))
                P2 = np.array(P2)
                assert_allclose(P1, P2)
                #Fast regularization
                rls1.solve(1024)
                P1 = rls1.leave_one_out()
                P2 = []
                for i in range(X.shape[0]): 
                    X_train = np.delete(X, i, axis=0)
                    Y_train = np.delete(Y, i, axis=0)
                    X_test = X[i]
                    rls2 = RLS(X_train, Y_train, regparam = 1024, bias = 3.0)
                    P2.append(rls2.predict(X_test))
                P2 = np.array(P2)
                assert_allclose(P1, P2)
                #kernels
                rls1 = RLS(X, Y, kernel = "GaussianKernel", gamma = 0.01)
                P1 = rls1.leave_one_out()
                P2 = []
                for i in range(X.shape[0]): 
                    X_train = np.delete(X, i, axis=0)
                    Y_train = np.delete(Y, i, axis=0)
                    X_test = X[i]
                    rls2 = RLS(X_train, Y_train, kernel = "GaussianKernel", gamma = 0.01)
                    P2.append(rls2.predict(X_test))
                P2 = np.array(P2)
                assert_allclose(P1, P2)
                
    def test_leave_pair_out(self):
        #compares holdout and leave-pair-out
        start = [0, 2, 3, 5]
        end = [1, 3, 6, 8]
        for X in [self.Xtrain1, self.Xtrain2]:
            for Y in [self.Ytrain1, self.Ytrain2]:
                #LPO with linear kernel
                rls1 = RLS(X, Y, regparam = 7.0, bias=3.0)
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
                rls1 = RLS(X, Y, regparam = 11.0, kenerl="PolynomialKernel", coef0=1, degree=3)
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
            dumbho = np.squeeze(dumbho)
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
                assert_allclose(dumbho, predho)
                #for row in range(predho.shape[0]):
                #    for col in range(predho.shape[1]):
                #        self.assertAlmostEqual(dumbho[row,col],predho[row,col])
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
        
