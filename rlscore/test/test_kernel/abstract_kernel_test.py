"""This module contains the test functions for the kernels"""
import unittest

from numpy import random
from numpy import zeros
from numpy import dot
from scipy import sparse as sp
import numpy as np

from rlscore import data_sources


class AbstractKernelTest(unittest.TestCase):
    
    def setUp(self):
        #randomly generate data matrix
        self.X = random.random((10,50))
        #some basis vectors
        self.trainsets = [self.X, self.X.T]
        self.bvectors = [0,3,7,8]
        self.setParams()
        #self.paramsets = [{"bias":0.}, {"bias":3.}]
        #set kernel
        
    def setParams(self):
        #self.kernel = XXX
        #self.paramsets = YYYY
        raise NotImplementedError()    
    
    def k_func(self, x1, x2, bias=0.):
        #linear kernel is simply the dot product, optionally
        #with a bias parameter
        #return (x1.T*x2)[0,0]+bias
        raise NotImplementedError()
                
    def testGetTrainingKernelMatrix(self):
        #Tests that the computational shortcuts used in constructing
        #the training kernel matrices work correctly
        #First data matrix has more features than examples,
        for X in self.trainsets:
            #Reduced set approximation is also tested
            for bvectors in [None, self.bvectors]:
                rpool = {data_sources.TRAIN_FEATURES : X, data_sources.BASIS_VECTORS:bvectors}
                for paramset in self.paramsets:
                    p = {}
                    p.update(rpool)
                    p.update(paramset)
                    k = self.kernel.createKernel(**p)
                    K = k.getKM(X).T
                    if bvectors != None:
                        x_indices = bvectors
                    else:
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
            X_test = random.random((22,X.shape[1]))
            #Reduced set approximation is also tested
            for bvectors in [None, self.bvectors]:
                rpool = {data_sources.TRAIN_FEATURES : X, data_sources.BASIS_VECTORS:bvectors}
                for paramset in self.paramsets:
                    p = {}
                    p.update(rpool)
                    p.update(paramset)
                    k = self.kernel.createKernel(**p)
                    K = k.getKM(X_test).T
                    if bvectors != None:
                        x_indices = bvectors
                    else:
                        x_indices = range(X.shape[0])
                    for i, x_ind in enumerate(x_indices):
                        for j in range(X_test.shape[0]):
                            correct = self.k_func(X[x_ind], X_test[j], paramset)
                            k_output = K[i,j]
                            self.assertAlmostEqual(correct,k_output)


    def testDataTypes(self):
        X = self.trainsets[0]
        Xt = random.random((22,X.shape[1]))
        X_mat = np.mat(X)
        Xt_mat = np.mat(Xt)
        X_sp = sp.csr_matrix(X)
        Xt_sp = sp.csr_matrix(Xt)
        trainsets = [X, X_mat, X_sp]
        testsets = [Xt, Xt_mat, Xt_sp]
        params = self.paramsets[0]
        for bvectors in [None, self.bvectors]:
            K_c = None
            for X in trainsets:
                p = {data_sources.TRAIN_FEATURES : X,
                         data_sources.BASIS_VECTORS:bvectors}
                p.update(params)
                k = self.kernel.createKernel(**p)
                for Xt in testsets:
                    K = k.getKM(Xt).T
                    self.assertTrue(type(K)==np.ndarray)
                    if K_c == None:
                        K_c = K
                    else:
                        self.assertTrue(np.allclose(K_c, K))
                    
