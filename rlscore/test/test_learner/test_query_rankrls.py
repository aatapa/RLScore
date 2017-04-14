import numpy as np
import numpy.linalg as la
from numpy.testing import assert_allclose
import unittest

from rlscore.learner import QueryRankRLS
from rlscore.kernel import GaussianKernel, PolynomialKernel


def mapQids(qids):
    """Maps qids to running numbering starting from zero, and partitions
    the training data indices so that each partition corresponds to one
    query"""
    qid_dict = {}
    folds = {}
    counter = 0
    for index, qid in enumerate(qids):
        if not qid in qid_dict:
            qid_dict[qid] = counter
            folds[qid] = []
            counter += 1
        folds[qid].append(index)
    indslist = []
    for f in folds.values():
        indslist.append(f)
    return indslist

def generate_qids(m):
    qids = []
    qsize = int(m / 10)
    for i in range(int(m / qsize)):
        qids = qids + [i] * qsize
    qids = qids + [i + 1] * (m % qsize)
    objcount = np.max(qids)+1
    P = np.zeros((m, objcount))
    for i in range(m):
        qid = qids[i]
        P[i, qid] = 1.
    labelcounts = np.sum(P, axis=0)
    P = np.divide(P, np.sqrt(labelcounts))
    D = np.ones((1, m))
    L = np.multiply(np.eye(m), D) - np.dot(P, P.T)
    return qids, L

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
        qids, L = generate_qids(m)
        #reduced set approximation
        primal_rls = QueryRankRLS(X, Y, qids, basis_vectors = X[self.bvectors], regparam=0.001)
        W = primal_rls.predictor.W
        K = np.dot(X, X.T)
        Kr = K[:, self.bvectors]
        Krr = K[np.ix_(self.bvectors, self.bvectors)]
        A = np.linalg.solve(np.dot(Kr.T, np.dot(L, Kr))+ 0.001 * Krr, np.dot(Kr.T, np.dot(L, Y)))
        #W_reduced = np.dot(X[self.bvectors].T, A)
        W_reduced = np.dot(X[self.bvectors].T, A)
        assert_allclose(W, W_reduced)
        
    def test_linear(self):
        #Test that learning with linear kernel works correctly both
        #with low and high-dimensional data
        for X in [self.Xtrain1, self.Xtrain2]:
            for Y in [self.Ytrain1, self.Ytrain2]:
                #Basic case
                m = X.shape[0]
                qids, L = generate_qids(m)
                primal_rls = QueryRankRLS(X, Y, qids, regparam=1.0, bias=0.)
                W = primal_rls.predictor.W
                d = X.shape[1]
                W2 = np.linalg.solve(np.dot(X.T, np.dot(L, X)) + np.eye(d), np.dot(X.T, np.dot(L, Y)))
                assert_allclose(W, W2)
                #For RankRLS, bias should have no effect
                primal_rls = QueryRankRLS(X, Y, qids, regparam=1.0, bias=5.)
                W2 = primal_rls.predictor.W
                assert_allclose(W, W2)
                #Fast regularization
                primal_rls.solve(10)
                W = primal_rls.predictor.W
                W2 = np.linalg.solve(np.dot(X.T, np.dot(L, X)) + 10 * np.eye(d), np.dot(X.T, np.dot(L, Y)))
                assert_allclose(W, W2)
                #reduced set approximation
                primal_rls = QueryRankRLS(X, Y, qids, basis_vectors = X[self.bvectors], regparam=5.0)
                W = primal_rls.predictor.W
                K = np.dot(X, X.T)
                Kr = K[:, self.bvectors]
                Krr = K[np.ix_(self.bvectors, self.bvectors)]
                A = np.linalg.solve(np.dot(Kr.T, np.dot(L, Kr))+ 5.0 * Krr, np.dot(Kr.T, np.dot(L, Y)))
                W_reduced = np.dot(X[self.bvectors].T, A)
                #assert_allclose(W, W_reduced)
                #Pre-computed linear kernel, reduced set approximation
                dual_rls = QueryRankRLS(Kr, Y, qids, kernel="PrecomputedKernel", basis_vectors = Krr, regparam=5.0)
                W = np.dot(X[self.bvectors].T, dual_rls.predictor.W)
                assert_allclose(W, W_reduced)
#                 #Precomputed kernel matrix
#                 dual_rls = GlobalRankRLS(K, Y, kernel = "PrecomputedKernel", regparam=0.01)
#                 W = np.dot(X.T, dual_rls.predictor.W)
#                 W2 = np.linalg.solve(np.dot(X.T, np.dot(L, X)) + 0.01 * np.eye(d), np.dot(X.T, np.dot(L, Y)))
#                 assert_allclose(W, W2)

    def test_kernel(self):
        #tests that learning with kernels works
        for X in [self.Xtrain1, self.Xtrain2]:
            for Y in [self.Ytrain1, self.Ytrain2]:
                m = X.shape[0]
                qids, L = generate_qids(m)
                #Basic case
                dual_rls = QueryRankRLS(X, Y, qids, kernel= "GaussianKernel", regparam=5.0, gamma=0.01)
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
                dual_rls = QueryRankRLS(K, Y, qids, kernel="PrecomputedKernel", regparam = 1000)
                assert_allclose(dual_rls.predictor.W, A2)
                #Reduced set approximation
                kernel = PolynomialKernel(X[self.bvectors], gamma=0.5, coef0 = 1.2, degree = 2)              
                Kr = kernel.getKM(X)
                Krr = kernel.getKM(X[self.bvectors])
                dual_rls = QueryRankRLS(X, Y, qids, kernel="PolynomialKernel", basis_vectors = X[self.bvectors], regparam = 200, gamma=0.5, coef0=1.2, degree = 2)
                A = dual_rls.predictor.A
                A2 = np.linalg.solve(np.dot(Kr.T, np.dot(L, Kr))+ 200 * Krr, np.dot(Kr.T, np.dot(L, Y)))
                assert_allclose(A, A2)
                dual_rls = QueryRankRLS(Kr, Y, qids, kernel="PrecomputedKernel", basis_vectors = Krr, regparam=200)
                A = dual_rls.predictor.W
                assert_allclose(A, A2)
                
    def test_holdout(self):
        for X in [self.Xtrain1, self.Xtrain2]:
            for Y in [self.Ytrain1, self.Ytrain2]:
                m = X.shape[0]
                qids, L = generate_qids(m)
                qids = np.array(qids)
                hoindices = np.where(qids == 1)[0]
                hocompl = list(set(range(m)) - set(hoindices))
                #Holdout with linear kernel
                rls1 = QueryRankRLS(X, Y, qids)
                rls2 = QueryRankRLS(X[hocompl], Y[hocompl], qids[hocompl])
                P1 = rls1.holdout(hoindices)
                P2 = rls2.predict(X[hoindices])
                assert_allclose(P1, P2)
                #Holdout with bias
                rls1 = QueryRankRLS(X, Y, qids, bias = 3.0)
                rls2 = QueryRankRLS(X[hocompl], Y[hocompl], qids[hocompl], bias = 3.0)
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
                rls1 = QueryRankRLS(X, Y, qids, kernel = "GaussianKernel", gamma = 0.01)
                rls2 = QueryRankRLS(X[hocompl], Y[hocompl], qids[hocompl], kernel = "GaussianKernel", gamma = 0.01)
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
                I = [0,4,8]
                self.assertRaises(IndexError, rls1.holdout, I)
    
    def testLabelRankRLS(self):
        
        print("Testing the cross-validation routines of the QueryRankRLS module.\n")
        
        np.random.seed(100)
        floattype = np.float64
        
        m, n = 100, 400 #data, features
        Xtrain = np.mat(np.random.rand(m, n))
        K = Xtrain * Xtrain.T
        ylen = 1
        Y = np.mat(np.zeros((m, ylen), dtype=floattype))
        Y[:, 0] = np.sum(Xtrain, 1)
        
        
        labelcount = 5
        
        hoindices = range(labelcount)
        hocompl = list(set(range(m)) - set(hoindices))
        
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
        qidlist_cv = qidlist[5: len(qidlist)]
        
        objcount = max(qidlist) + 1
        P = np.mat(np.zeros((m, objcount), dtype=np.float64))
        for i in range(m):
            qid = qidlist[i]
            P[i, qid] = 1.
        labelcounts = np.sum(P, axis=0)
        P = np.divide(P, np.sqrt(labelcounts))
        D = np.mat(np.ones((1, m), dtype=np.float64))
        L = np.multiply(np.eye(m), D) - P * P.T
        
        Kcv = K[np.ix_(hocompl, hocompl)]
        Lcv = L[np.ix_(hocompl, hocompl)]
        
        Xcv = Xtrain[hocompl]
        Xtest = Xtrain[hoindices]
        Yho = Y[hocompl]
        
        rpool = {}
        rpool["X"] = Xtrain
        rpool["Y"] = Y
        rpool["qids"] = qidlist
        primalrls = QueryRankRLS(**rpool)        
        
        rpool = {}
        rpool["X"] = K
        rpool['kernel'] = 'PrecomputedKernel'
        rpool["Y"] = Y
        rpool["qids"] = qidlist        
        dualrls = QueryRankRLS(**rpool)
        
        rpool = {}
        rpool['X'] = Xcv
        rpool['Y'] = Yho
        rpool['qids'] = qidlist_cv
        primalrls_naive = QueryRankRLS(**rpool)

        rpool = {}
        rpool['X'] = Kcv
        rpool['kernel'] = 'PrecomputedKernel'        
        rpool['Y'] = Yho
        #rpool['X'] = Xcv
        rpool['qids'] = qidlist_cv
        dualrls_naive = QueryRankRLS(**rpool)
        
        testkm = K[np.ix_(hocompl, hoindices)]
        
        loglambdas = range(-5, 5)
        for j in range(0, len(loglambdas)):
            regparam = 2. ** loglambdas[j]
            print
            print("Regparam 2^%1d" % loglambdas[j])
            
            
            print(str(np.squeeze(np.array((testkm.T * la.inv(Lcv * Kcv + regparam * np.eye(Lcv.shape[0])) * Lcv * Yho).T))) + ' Dumb HO')
            
            predhos = []
            primalrls_naive.solve(regparam)
            predho = primalrls_naive.predictor.predict(Xtest)
            print(str(predho.T) + ' Naive HO (primal)')
            predhos.append(predho)
            
            dualrls_naive.solve(regparam)
            predho = dualrls_naive.predictor.predict(testkm.T)
            print(str(predho.T) + ' Naive HO (dual)')
            predhos.append(predho)
            
            primalrls.solve(regparam)
            predho = np.squeeze(primalrls.holdout(hoindices))
            print(str(predho.T) + ' Fast HO (primal)')
            predhos.append(predho)
            
            dualrls.solve(regparam)
            predho = np.squeeze(dualrls.holdout(hoindices))
            print(str(predho.T) + ' Fast HO (dual)')
            predhos.append(predho)
            
            predho0 = predhos.pop(0)
            for predho in predhos:
                self.assertEqual(predho0.shape, predho.shape)
                for row in range(predho.shape[0]):
                    #for col in range(predho.shape[1]):
                    #    self.assertAlmostEqual(predho0[row,col],predho[row,col], places=5)
                        self.assertAlmostEqual(predho0[row],predho[row], places=5)
