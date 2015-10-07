import sys
import unittest

from numpy import *
import numpy.linalg as la
import numpy as np

from rlscore.learner import RLS
from rlscore.kernel import GaussianKernel
from rlscore.kernel import RsetKernel


class Test(unittest.TestCase):
    
    
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
        Xtrain = random.rand(m, n)
        ylen = 1
        Y = mat(zeros((m, ylen), dtype=floattype))
        Y = mat(random.rand(m, 1))
        basis_vectors = [0,3,7,8]
        
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
        bk = GaussianKernel(**{'X':Xtrain[basis_vectors], 'gamma':0.001})
        rk = RsetKernel(**{'base_kernel':bk, 'basis_features':Xtrain[basis_vectors], 'X':Xtrain})
        
        rpool = {}
        rpool['X'] = Xtrain
        bk2 = GaussianKernel(**{'X':Xtrain, 'gamma':0.001})
        K = np.mat(bk2.getKM(Xtrain))
        
        Kho = K[ix_(hocompl, hocompl)]
        Yho = Y[hocompl]
        
        #rpool = {}
        #rpool['Y'] = Y
        #rpool['kernel_matrix'] = K[basis_vectors]
        #rpool['basis_vectors'] = basis_vectors
        #dualrls = RLS.createLearner(**rpool)
        
        rpool = {}
        rpool['Y'] = Y
        rpool['X'] = Xtrain
        rpool['basis_vectors'] = Xtrain[basis_vectors]
        primalrls = RLS.createLearner(**rpool)
        
        testkm = K[ix_(hocompl, hoindices)]
        Xhocompl = Xtrain[hocompl]
        testX = Xtrain[hoindices]
        
        rpool = {}
        rpool['Y'] = Yho
        rpool['X'] = Xhocompl
        rk = RsetKernel(**{'base_kernel':bk, 'basis_features':Xtrain[basis_vectors], 'X':Xhocompl})
        rpool['kernel_obj'] = rk
        dualrls_naive = RLS.createLearner(**rpool)
        
        rpool = {}
        rpool['Y'] = Yho
        rpool['X'] = Xhocompl
        primalrls_naive = RLS.createLearner(**rpool)
        
        rsaK = K[:, basis_vectors] * la.inv(K[ix_(basis_vectors, basis_vectors)]) * K[basis_vectors]
        rsaKho = rsaK[ix_(hocompl, hocompl)]
        rsa_testkm = rsaK[ix_(hocompl, hoindices)]
        loglambdas = range(-5, 5)
        for j in range(0, len(loglambdas)):
            regparam = 2. ** loglambdas[j]
            print
            print "Regparam 2^%1d" % loglambdas[j]
            
            print (rsa_testkm.T * la.inv(rsaKho + regparam * eye(rsaKho.shape[0])) * Yho).T, 'Dumb HO (dual)'
            dumbho = np.squeeze(np.array(rsa_testkm.T * la.inv(rsaKho + regparam * eye(rsaKho.shape[0])) * Yho))
            
            dualrls_naive.solve(regparam)
            predho1 = np.squeeze(dualrls_naive.getModel().predict(testX))
            print predho1.T, 'Naive HO (dual)'
            
            #dualrls.solve(regparam)
            #predho2 = np.squeeze(dualrls.computeHO(hoindices))
            #print predho2.T, 'Fast HO (dual)'
            
            for predho in [dumbho, predho1]:#, predho2]:
                self.assertEqual(dumbho.shape, predho.shape)
                for row in range(predho.shape[0]):
                    #for col in range(predho.shape[1]):
                    #    self.assertAlmostEqual(dumbho[row,col],predho[row,col])
                        self.assertAlmostEqual(dumbho[row],predho[row])
