import unittest

from numpy import *
import numpy.linalg as la
import numpy as np

from rlscore.learner import RLS
from rlscore.kernel import GaussianKernel


class Test(unittest.TestCase):
    
    
    def testRLS(self):
        
        print("\n\n\n\nTesting the cross-validation routines of the RLS module.\n\n")
        
        m, n = 100, 300
        Xtrain = random.rand(m, n)
        Y = mat(random.rand(m, 1))
        basis_vectors = [0,3,7,8]
        
        #hoindices = [45, 50, 55]
        hoindices = [0, 1, 2]
        hocompl = list(set(range(m)) - set(hoindices))
        
        bk = GaussianKernel(**{'X':Xtrain[basis_vectors], 'gamma':0.001})
        
        rpool = {}
        rpool['X'] = Xtrain
        bk2 = GaussianKernel(**{'X':Xtrain, 'gamma':0.001})
        K = np.mat(bk2.getKM(Xtrain))
        
        Yho = Y[hocompl]
        
        
        rpool = {}
        rpool['Y'] = Y
        rpool['X'] = Xtrain
        rpool['basis_vectors'] = Xtrain[basis_vectors]
        
        Xhocompl = Xtrain[hocompl]
        testX = Xtrain[hoindices]
        
        rpool = {}
        rpool['Y'] = Yho
        rpool['X'] = Xhocompl
        rpool["kernel"] = "RsetKernel"
        rpool["base_kernel"] = bk
        rpool["basis_features"] = Xtrain[basis_vectors]
        #rk = RsetKernel(**{'base_kernel':bk, 'basis_features':Xtrain[basis_vectors], 'X':Xhocompl})
        dualrls_naive = RLS(**rpool)
        
        rpool = {}
        rpool['Y'] = Yho
        rpool['X'] = Xhocompl
        
        rsaK = K[:, basis_vectors] * la.inv(K[ix_(basis_vectors, basis_vectors)]) * K[basis_vectors]
        rsaKho = rsaK[ix_(hocompl, hocompl)]
        rsa_testkm = rsaK[ix_(hocompl, hoindices)]
        loglambdas = range(-5, 5)
        for j in range(0, len(loglambdas)):
            regparam = 2. ** loglambdas[j]
            print("\nRegparam 2^%1d" % loglambdas[j])
            
            print((rsa_testkm.T * la.inv(rsaKho + regparam * eye(rsaKho.shape[0])) * Yho).T, 'Dumb HO (dual)')
            dumbho = np.squeeze(np.array(rsa_testkm.T * la.inv(rsaKho + regparam * eye(rsaKho.shape[0])) * Yho))
            
            dualrls_naive.solve(regparam)
            predho1 = np.squeeze(dualrls_naive.predictor.predict(testX))
            print(predho1.T, 'Naive HO (dual)')
            
            #dualrls.solve(regparam)
            #predho2 = np.squeeze(dualrls.computeHO(hoindices))
            #print predho2.T, 'Fast HO (dual)'
            
            for predho in [dumbho, predho1]:#, predho2]:
                self.assertEqual(dumbho.shape, predho.shape)
                for row in range(predho.shape[0]):
                    #for col in range(predho.shape[1]):
                    #    self.assertAlmostEqual(dumbho[row,col],predho[row,col])
                        self.assertAlmostEqual(dumbho[row],predho[row])
