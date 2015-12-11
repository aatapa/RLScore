import unittest

import numpy as np
import numpy.linalg as la

from rlscore.learner import RLS

class Test(unittest.TestCase):
    
    def setUp(self):
        np.random.seed(100)
        self.X = np.random.random((10,100))
        #data matrix full of zeros
        self.X_zeros = np.zeros((10,100))
        self.testm = [self.X, self.X.T, self.X_zeros]
        #some basis vectors
        self.basis_vectors = [0,3,7,8]
        
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
        hopred = dualrls.leave_pairs_out(np.array([hoindices2[0], 4, 6]), np.array([hoindices2[1], 5, 7]))
        print(str(hopred[0][0]) + '\n' + str(hopred[1][0]) + ' Fast LPO')
        
