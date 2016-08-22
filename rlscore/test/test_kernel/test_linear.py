import unittest

import numpy as np
from scipy import sparse as sp

from rlscore.kernel import LinearKernel


class Test(unittest.TestCase):
    
    def setUp(self):
        #randomly generate data matrix
        self.X = np.random.random((10,50))
        #some basis vectors
        self.trainsets = [self.X, self.X.T]
        #self.basis_vectors = [0,3,7,8]
        self.bvecinds = [0,3,7,8]
        self.setParams()
        
    def setParams(self):
        self.kernel = LinearKernel
        self.paramsets = [{"bias":0.}, {"bias":3.}]
        
    def k_func(self, x1, x2, params):
        #linear kernel is simply the dot product, optionally
        #with a bias parameter
        bias = params["bias"]
        return np.dot(x1, x2)+bias
    
    def testGetTrainingKernelMatrix(self):
        #Tests that the computational shortcuts used in constructing
        #the training kernel matrices work correctly
        #First data matrix has more features than examples,
        for X in self.trainsets:
            #Reduced set approximation is also tested
            for X_tr in [X, X[self.bvecinds]]:
                rpool = {'X' : X}
                for paramset in self.paramsets:
                    p = {}
                    p.update(rpool)
                    p.update(paramset)
                    k = self.kernel(**p)
                    K = k.getKM(X).T
                    x_indices = range(X.shape[0])
                    for i, x_ind in enumerate(x_indices):
                        for j in range(X.shape[0]):
                            correct = self.k_func(X[x_ind], X[j], paramset)
                            k_output = K[i,j]
                            self.assertAlmostEqual(correct,k_output)
                            
    
    def testGetTestKernelMatrix(self):
        #Tests that the computational shortcuts used in constructing
        #kernel matrix between training and test data work correctly
        #First data matrix has more features than examples, second the other
        #way around, third one is just full of zeros
        for X in self.trainsets:
            X_test = np.random.random((22,X.shape[1]))
            #Reduced set approximation is also tested
            #for bvecinds, bvecs in zip([None, self.bvecinds], [None, X[self.bvecinds]]):
            for X_tr in [X, X[self.bvecinds]]:
                rpool = {'X' : X}
                for paramset in self.paramsets:
                    p = {}
                    p.update(rpool)
                    p.update(paramset)
                    k = self.kernel(**p)
                    K = k.getKM(X_test).T
                    x_indices = range(X.shape[0])
                    for i, x_ind in enumerate(x_indices):
                        for j in range(X_test.shape[0]):
                            correct = self.k_func(X[x_ind], X_test[j], paramset)
                            k_output = K[i,j]
                            self.assertAlmostEqual(correct,k_output)


    def testDataTypes(self):
        X = self.trainsets[0]
        Xt = np.random.random((22,X.shape[1]))
        X_mat = np.mat(X)
        Xt_mat = np.mat(Xt)
        X_sp = sp.csr_matrix(X)
        Xt_sp = sp.csr_matrix(Xt)
        trainsets = [X, X_mat, X_sp]
        testsets = [Xt, Xt_mat, Xt_sp]
        params = self.paramsets[0]
        for X_tr in [X, X[self.bvecinds]]:
            K_c = None
            for X in trainsets:
                p = {'X' : X_tr}
                p.update(params)
                k = self.kernel(**p)
                for Xt in testsets:
                    K = k.getKM(Xt).T
                    self.assertTrue(type(K)==np.ndarray)
                    if K_c is None:
                        K_c = K
                    else:
                        self.assertTrue(np.allclose(K_c, K))
        