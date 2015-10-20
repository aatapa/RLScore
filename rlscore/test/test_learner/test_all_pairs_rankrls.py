import sys
import unittest

import numpy as np
import numpy.linalg as la

from rlscore.learner import GlobalRankRLS

class Test(unittest.TestCase):
    
    def testAllPairsRankRLS(self):
        
        print("Testing the cross-validation routines of the GlobalRankRLS module.\n\n")
        
        np.random.seed(100)
        floattype = np.float64
        
        m, n, h = 30, 200, 10
        Xtrain = np.random.rand(m, n)
        trainlabels = np.random.rand(m, h)
        trainkm = np.dot(Xtrain, Xtrain.T)
        ylen = 1
        
        L = np.mat(m * np.eye(m) - np.ones((m, m), dtype=floattype))
        
        hoindices = [5, 7]
        hoindices3 = [5, 7, 9]
        hocompl = list(set(range(m)) - set(hoindices))
        hocompl3 = list(set(range(m)) - set(hoindices3))
        
        loglambdas = range(-5, 5)
        for j in range(0, len(loglambdas)):
            regparam = 2. ** loglambdas[j]
            print("\nRegparam 2^%1d" % loglambdas[j])
            
            Kcv = trainkm[np.ix_(hocompl, hocompl)]
            Ycv = trainlabels[hocompl]
            Ktest = trainkm[np.ix_(hocompl, hoindices)]
            
            Xcv = Xtrain[hocompl]
            Xtest = Xtrain[hoindices]
            
            Lcv = np.mat((m - 2) * np.eye(m - 2) - np.ones((m - 2, m - 2), dtype=floattype))
            
            oind = 1
            rpool = {}
            rpool['Y'] = Ycv
            rpool['X'] = Xcv
            rpool['regparam'] = regparam
            naivedualrls = GlobalRankRLS(**rpool)
            naivedualrls.solve(regparam)
            hopreds = []
            
            hopred = naivedualrls.predictor.predict(Xtest)
            print(str(hopred[0, oind]) + ' ' + str(hopred[1, oind]) + ' Naive')
            hopreds.append((hopred[0, oind], hopred[1, oind]))
            
            rpool = {}
            rpool['Y'] = trainlabels
            rpool['X'] = Xtrain
            rpool['regparam'] = regparam
            hodualrls = GlobalRankRLS(**rpool)
            hodualrls.solve(regparam)
            hopred = hodualrls.computePairwiseCV([hoindices[0]], [hoindices[1]], oind=oind)
            print(str(hopred[0][0]) + ' ' + str(hopred[1][0]) + ' Fast')
            hopreds.append((hopred[0][0], hopred[1][0]))
            self.assertAlmostEqual(hopreds[0][0], hopreds[1][0])
            self.assertAlmostEqual(hopreds[0][1], hopreds[1][1])
            hopreds = []
            
            rpool = {}
            rpool['Y'] = trainlabels
            rpool['X'] = Xtrain
            rpool['regparam'] = regparam
            hoprimalrls = GlobalRankRLS(**rpool)
            hoprimalrls.solve(regparam)
            hopred = hoprimalrls.computeHO(hoindices)
            print(str(hopred[0, oind]) + ' ' + str(hopred[1, oind]) + ' HO')
            hopreds.append((hopred[0, oind], hopred[1, oind]))
            
            hopred = Xtest * la.inv(Xcv.T * Lcv * Xcv + regparam * np.mat(np.eye(n))) * Xcv.T * Lcv * Ycv
            print(str(hopred[0, oind]) + ' ' + str(hopred[1, oind]) + ' Dumb (primal)')
            hopreds.append((hopred[0, oind], hopred[1, oind]))
            
            hopred0 = hopreds.pop(0)
            for hopred in hopreds:
                self.assertAlmostEqual(hopred0[0],hopred[0])
                self.assertAlmostEqual(hopred0[1],hopred[1])
