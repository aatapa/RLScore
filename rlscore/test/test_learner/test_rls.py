import unittest

from numpy import *
import numpy.linalg as la

from rlscore import data_sources
from rlscore.utilities import decomposition
from rlscore.utilities import adapter
from rlscore.learner import RLS

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
        
        m, n = 400, 100
        Xtrain = mat(random.rand(n, m))
        K = Xtrain.T * Xtrain
        ylen = 2
        Y = mat(zeros((m, ylen), dtype=floattype))
        Y = mat(random.rand(m, ylen))
        
        
        def complement(indices, m):
            compl = range(m)
            for ind in indices:
                compl.remove(ind)
            return compl
        
        #hoindices = [45, 50, 55]
        hoindices = [45]
        hocompl = complement(hoindices, m)
        
        Kho = K[ix_(hocompl, hocompl)]
        Yho = Y[hocompl]
        
        rpool = {}
        rpool['train_labels'] = Y
        rpool[data_sources.KMATRIX] = K
        dualrls = RLS.createLearner(**rpool)
        
        rpool = {}
        rpool["train_features"] = Xtrain.T
        rpool["train_labels"] = Y
        primalrls = RLS.createLearner(**rpool)
        #primalrls.setDecomposition(svals, evecs, U)
        #primalrls.svecs = evecs
        #primalrls.svals = svals
        #primalrls.setLabels(Y)
        
        rpool = {}
        rpool['train_labels'] = Yho
        rpool[data_sources.KMATRIX] = Kho
        #rpool['parameters'] = params
        dualrls_naive = RLS.createLearner(**rpool)
        
        testkm = K[ix_(hocompl, hoindices)]
        trainX = Xtrain[:, hocompl]
        testX = Xtrain[:, hoindices]
        rpool = {}
        rpool['train_labels'] = Yho
        rpool[data_sources.TRAIN_FEATURES] = trainX.T
        #rpool['parameters'] = params
        #dualrls_naive = RLS.createLearner(train_labels = Yho, kmatrix = Kho, parameters = params)
        primalrls_naive = RLS.createLearner(**rpool)
        #svals, evecs, U = Decompositions.decomposeDataMatrix(trainX)
        #primalrls_naive = RLS()
        #primalrls_naive.setDecomposition(svals,evecs,U)
        #primalrls_naive.setLabels(Yho)
        
        loglambdas = range(-5, 5)
        for j in range(0, len(loglambdas)):
            regparam = 2. ** loglambdas[j]
            print
            print "Regparam 2^%1d" % loglambdas[j]
            
            #print (testkm.T * la.inv(Kho + regparam * eye(Kho.shape[0])) * Yho), 'Dumb HO (dual)'
            dumbho = testkm.T * la.inv(Kho + regparam * eye(Kho.shape[0])) * Yho
            print dumbho, 'Dumb HO (dual)'
            
            dualrls_naive.solve(regparam)
            rpool = {}
            rpool[data_sources.PREDICTION_FEATURES] = testkm.T
            predho1 = dualrls_naive.getModel().predictFromPool(rpool)
            print predho1, 'Naive HO (dual)'
            
            dualrls.solve(regparam)
            predho2 = dualrls.computeHO(hoindices)
            print predho2, 'Fast HO (dual)'
            
            dualrls.solve(regparam)
            predho = dualrls.computeLOO()[hoindices[0]]
            print predho, 'Fast LOO (dual)'
            
            primalrls_naive.solve(regparam)
            predho3 = primalrls_naive.getModel().predict(testX.T)
            print predho3, 'Naive HO (primal)'
            
            primalrls.solve(regparam)
            predho4 = primalrls.computeHO(hoindices)
            print predho4, 'Fast HO (primal)'
            for predho in [predho1, predho2, predho3, predho4]:
                self.assertEqual(dumbho.shape, predho.shape)
                for row in range(predho.shape[0]):
                    for col in range(predho.shape[1]):
                        self.assertAlmostEqual(dumbho[row,col],predho[row,col])
            primalrls.solve(regparam)
            predho = primalrls.computeLOO()[hoindices[0]]
            print predho, 'Fast LOO (primal)'
        print
        hoindices = range(100, 300)
        hocompl = complement(hoindices, m)
        
        Kho = K[ix_(hocompl, hocompl)]
        Yho = Y[hocompl]
        testkm = K[ix_(hocompl, hoindices)]
        
        #print (testkm.T * la.inv(Kho + regparam * eye(Kho.shape[0])) * Yho), 'Dumb HO (dual)'
        dumbho = testkm.T * la.inv(Kho + regparam * eye(Kho.shape[0])) * Yho
        #print dumbho, 'Dumb HO (dual)'
        
        rpool = {}
        rpool['train_labels'] = Yho
        rpool[data_sources.KMATRIX] = Kho
        #rpool['parameters'] = params
        #dualrls_naive = RLS.createLearner(train_labels = Yho, kmatrix = Kho, parameters = params)
        dualrls_naive = RLS.createLearner(**rpool)
        dualrls_naive.solve(regparam)
        rpool = {}
        rpool[data_sources.PREDICTION_FEATURES] = testkm.T
        predho1 = dualrls_naive.getModel().predictFromPool(rpool)
        print sum(abs(predho1-dumbho)), 'Naive HO (dual)'
        
        dualrls.solve(regparam)
        predho2 = dualrls.computeHO(hoindices)
        print sum(abs(predho2-dumbho)), 'Fast HO (dual)'
        
