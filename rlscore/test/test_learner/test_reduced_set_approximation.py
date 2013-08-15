import sys
import unittest

from numpy import *
import numpy.linalg as la
import numpy as np

from rlscore import data_sources
from rlscore.learner import RLS
from rlscore.kernel import GaussianKernel
from rlscore.kernel import RsetKernel


class Test(unittest.TestCase):
    
    def setUp(self):
        random.seed(100)
        self.X = random.random((10,100))
        #data matrix full of zeros
        self.X_zeros = zeros((10,100))
        self.testm = [self.X, self.X.T, self.X_zeros]
        #some basis vectors
        self.bvectors = [0,3,7,8]
        
    def testRLS(self):
        
        print
        print
        print
        print
        print "Testing the cross-validation routines of the RLS module."
        print
        print
        floattype = float64
        
        m, n = 100, 300
        Xtrain = mat(random.rand(n, m))
        #K = Xtrain.T * Xtrain
        ylen = 1
        Y = mat(zeros((m, ylen), dtype=floattype))
        Y = mat(random.rand(m, 1))
        
        
        def complement(indices, m):
            compl = range(m)
            for ind in indices:
                compl.remove(ind)
            return compl
        
        #hoindices = [45, 50, 55]
        hoindices = [0, 1, 2]
        hocompl = complement(hoindices, m)
        
        #bk = LinearKernel.Kernel()
        #bk = GaussianKernel.Kernel()
        bk = GaussianKernel.createKernel(**{data_sources.TRAIN_FEATURES:Xtrain[:,self.bvectors].T, 'gamma':'0.001'})
        rk = RsetKernel.createKernel(**{'base_kernel':bk, 'basis_features':Xtrain[:,self.bvectors].T, data_sources.TRAIN_FEATURES:Xtrain.T})
        
        rpool = {}
        rpool[data_sources.TRAIN_FEATURES] = Xtrain.T
        bk2 = GaussianKernel.createKernel(**{data_sources.TRAIN_FEATURES:Xtrain.T, 'gamma':'0.001'})
        K = np.mat(bk2.getKM(Xtrain.T))
        
        #svals, evecs = Decompositions.decomposeKernelMatrix(K)
        #dualrls = RLS()
        #dualrls.setDecomposition(svals, evecs)
        #dualrls.setLabels(Y)
        Kho = K[ix_(hocompl, hocompl)]
        Yho = Y[hocompl]
        
        rpool = {}
        rpool['train_labels'] = Y
        rpool[data_sources.KMATRIX] = K[self.bvectors]
        rpool[data_sources.BASIS_VECTORS] = self.bvectors
        dualrls = RLS.createLearner(**rpool)
        
        rpool = {}
        rpool['train_labels'] = Y
        rpool[data_sources.TRAIN_FEATURES] = Xtrain.T
        rpool[data_sources.BASIS_VECTORS] = self.bvectors
        #svals, evecs, U = Decompositions.decomposeDataMatrix(Xtrain)
        #primalrls = RLS()
        #primalrls.setDecomposition(svals, evecs, U)
        #primalrls.setLabels(Y)
        primalrls = RLS.createLearner(**rpool)
        #svals, evecs = Decompositions.decomposeKernelMatrix(Kho)
        
        testkm = K[ix_(hocompl, hoindices)]
        Xhocompl = Xtrain[:, hocompl]
        testX = Xtrain[:, hoindices]
        
        rpool = {}
        rpool[data_sources.TRAIN_LABELS] = Yho
        rpool[data_sources.TRAIN_FEATURES] = Xhocompl.T
        rk = RsetKernel.createKernel(**{'base_kernel':bk, 'basis_features':Xtrain[:,self.bvectors].T, data_sources.TRAIN_FEATURES:Xhocompl.T})
        rpool[data_sources.KERNEL_OBJ] = rk
        #rpool[data_sources.KMATRIX] = Kho
        #dualrls_naive = RLS.createLearner(train_labels = Yho, kmatrix = Kho)
        #print rpool
        dualrls_naive = RLS.createLearner(**rpool)
        #dualrls_naive = RLS()
        #dualrls_naive.setDecomposition(svals, evecs)
        #dualrls_naive.setLabels(Yho)
        
        rpool = {}
        rpool['train_labels'] = Yho
        rpool[data_sources.TRAIN_FEATURES] = Xhocompl.T
        #dualrls_naive = RLS.createLearner(train_labels = Yho, kmatrix = Kho)
        primalrls_naive = RLS.createLearner(**rpool)
        #svals, evecs, U = Decompositions.decomposeDataMatrix(Xhocompl)
        #primalrls_naive = RLS()
        #primalrls_naive.setDecomposition(svals,evecs,U)
        #primalrls_naive.setLabels(Yho)
        
        rsaK = K[:, self.bvectors] * la.inv(K[ix_(self.bvectors, self.bvectors)]) * K[self.bvectors]
        rsaKho = rsaK[ix_(hocompl, hocompl)]
        rsa_testkm = rsaK[ix_(hocompl, hoindices)]
        loglambdas = range(-5, 5)
        for j in range(0, len(loglambdas)):
            regparam = 2. ** loglambdas[j]
            print
            print "Regparam 2^%1d" % loglambdas[j]
            
            print (rsa_testkm.T * la.inv(rsaKho + regparam * eye(rsaKho.shape[0])) * Yho).T, 'Dumb HO (dual)'
            dumbho = rsa_testkm.T * la.inv(rsaKho + regparam * eye(rsaKho.shape[0])) * Yho
            
            dualrls_naive.solve(regparam)
            predho1 = dualrls_naive.getModel().predict(testX.T)
            print predho1.T, 'Naive HO (dual)'
            
            dualrls.solve(regparam)
            predho2 = dualrls.computeHO(hoindices)
            print predho2.T, 'Fast HO (dual)'
            
            #primalrls.solve(regparam)
            #predho4 = primalrls.computeHO(hoindices)
            #print predho4.T, 'Fast HO (primal)'
            for predho in [dumbho, predho1, predho2]:
                self.assertEqual(dumbho.shape, predho.shape)
                for row in range(predho.shape[0]):
                    for col in range(predho.shape[1]):
                        self.assertAlmostEqual(dumbho[row,col],predho[row,col])
            #primalrls.solve(regparam)
            #predho = primalrls.computeLOO()[hoindices[0]]
            #print predho.T, 'Fast LOO (primal)'
