import unittest

from numpy import *
import numpy.linalg as la

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
        self.basis_vectors = [0,3,7,8]
        
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
        Xtrain = mat(random.rand(m, n))
        K = Xtrain * Xtrain.T
        ylen = 2
        Y = mat(zeros((m, ylen), dtype=floattype))
        Y = mat(random.rand(m, ylen))
        
        #hoindices = [45, 50, 55]
        hoindices = [45]
        hocompl = list(set(range(m)) - set(hoindices))
        
        Kho = K[ix_(hocompl, hocompl)]
        Yho = Y[hocompl]
        
        kwargs = {}
        kwargs['train_labels'] = Y
        kwargs['kernel_matrix'] = K
        dualrls = RLS.createLearner(**kwargs)
        
        kwargs = {}
        kwargs["train_features"] = Xtrain
        kwargs["train_labels"] = Y
        primalrls = RLS.createLearner(**kwargs)
        
        kwargs = {}
        kwargs['train_labels'] = Yho
        kwargs['kernel_matrix'] = Kho
        dualrls_naive = RLS.createLearner(**kwargs)
        
        testkm = K[ix_(hocompl, hoindices)]
        trainX = Xtrain[hocompl]
        testX = Xtrain[hoindices]
        kwargs = {}
        kwargs['train_labels'] = Yho
        kwargs['train_features'] = trainX
        primalrls_naive = RLS.createLearner(**kwargs)
        
        loglambdas = range(-5, 5)
        for j in range(0, len(loglambdas)):
            regparam = 2. ** loglambdas[j]
            print
            print "Regparam 2^%1d" % loglambdas[j]
            
            dumbho = testkm.T * la.inv(Kho + regparam * eye(Kho.shape[0])) * Yho
            print dumbho, 'Dumb HO (dual)'
            
            dualrls_naive.solve(regparam)
            predho1 = dualrls_naive.getModel().predict(testkm.T)
            print predho1, 'Naive HO (dual)'
            
            dualrls.solve(regparam)
            predho2 = dualrls.computeHO(hoindices)
            print predho2, 'Fast HO (dual)'
            
            dualrls.solve(regparam)
            predho = dualrls.computeLOO()[hoindices[0]]
            print predho, 'Fast LOO (dual)'
            
            primalrls_naive.solve(regparam)
            predho3 = primalrls_naive.getModel().predict(testX)
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
        hocompl = list(set(range(m)) - set(hoindices))
        
        Kho = K[ix_(hocompl, hocompl)]
        Yho = Y[hocompl]
        testkm = K[ix_(hocompl, hoindices)]
        
        dumbho = testkm.T * la.inv(Kho + regparam * eye(Kho.shape[0])) * Yho
        
        kwargs = {}
        kwargs['train_labels'] = Yho
        kwargs['kernel_matrix'] = Kho
        dualrls_naive = RLS.createLearner(**kwargs)
        dualrls_naive.solve(regparam)
        predho1 = dualrls_naive.getModel().predict(testkm.T)
        print sum(abs(predho1-dumbho)), 'Naive HO (dual)'
        
        dualrls.solve(regparam)
        predho2 = dualrls.computeHO(hoindices)
        print sum(abs(predho2-dumbho)), 'Fast HO (dual)'
        
