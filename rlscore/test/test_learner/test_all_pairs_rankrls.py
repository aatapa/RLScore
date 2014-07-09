import sys
import unittest

import numpy as np
import numpy.linalg as la

from rlscore.learner import AllPairsRankRLS
from rlscore.kernel import LinearKernel

class Test(unittest.TestCase):
    
    def testAllPairsRankRLS(self):
        
        print
        print
        print
        print
        print "Testing the cross-validation routines of the AllPairsRankRLS module."
        print
        print
        
        np.random.seed(100)
        floattype = np.float64
        
        m, n, h = 30, 200, 10
        Xtrain = np.mat(np.random.rand(m, n))
        #trainlabels = np.sum(Xtrain, 0)
        trainlabels = np.mat(np.random.rand(m, h))
        trainkm = Xtrain * Xtrain.T
        ylen = 1
        
        def complement(indices, m):
            compl = range(m)
            for ind in indices:
                compl.remove(ind)
            return compl
        
        L = np.mat(m * np.eye(m) - np.ones((m, m), dtype=floattype))
        
        hoindices = [5, 7]
        hoindices3 = [5, 7, 9]
        hocompl = complement(hoindices, m)
        hocompl3 = complement(hoindices3, m)
        
        loglambdas = range(-5, 5)
        for j in range(0, len(loglambdas)):
            regparam = 2. ** loglambdas[j]
            print
            print "Regparam 2^%1d" % loglambdas[j]
            
            Kcv = trainkm[np.ix_(hocompl, hocompl)]
            Ycv = trainlabels[hocompl]
            Ktest = trainkm[np.ix_(hocompl, hoindices)]
            
            Xcv = Xtrain[hocompl]
            Xtest = Xtrain[hoindices]
            
            Lcv = np.mat((m - 2) * np.eye(m - 2) - np.ones((m - 2, m - 2), dtype=floattype))
            
            oind = 1
            rpool = {}
            rpool['train_labels'] = Ycv
            rpool['train_features'] = Xcv
            rpool['regparam'] = regparam
            naivedualrls = AllPairsRankRLS.createLearner(**rpool)
            naivedualrls.solve(regparam)
            hopreds = []
            
            hopred = naivedualrls.getModel().predict(Xtest)
            print hopred[0, oind], hopred[1, oind], 'Naive'
            hopreds.append((hopred[0, oind], hopred[1, oind]))
            
            rpool = {}
            rpool['train_labels'] = trainlabels
            rpool['train_features'] = Xtrain
            rpool['regparam'] = regparam
            hodualrls = AllPairsRankRLS.createLearner(**rpool)
            hodualrls.solve(regparam)
            hopred = hodualrls.computePairwiseCV([hoindices], oind=oind)
            print hopred[0][0], hopred[0][1], 'Fast'
            hopreds.append((hopred[0][0], hopred[0][1]))
            self.assertAlmostEqual(hopreds[0][0], hopreds[1][0])
            self.assertAlmostEqual(hopreds[0][1], hopreds[1][1])
            hopreds = []
            
            rpool = {}
            rpool['train_labels'] = trainlabels
            rpool['train_features'] = Xtrain
            rpool['regparam'] = regparam
            hoprimalrls = AllPairsRankRLS.createLearner(**rpool)
            hoprimalrls.solve(regparam)
            hopred = hoprimalrls.computeHO(hoindices)
            print hopred[0, oind], hopred[1, oind], 'HO'
            hopreds.append((hopred[0, oind], hopred[1, oind]))
            
            hopred = Xtest * la.inv(Xcv.T * Lcv * Xcv + regparam * np.mat(np.eye(n))) * Xcv.T * Lcv * Ycv
            print hopred[0, oind], hopred[1, oind], 'Dumb (primal)'
            hopreds.append((hopred[0, oind], hopred[1, oind]))
            
            hopred0 = hopreds.pop(0)
            for hopred in hopreds:
                self.assertAlmostEqual(hopred0[0],hopred[0])
                self.assertAlmostEqual(hopred0[1],hopred[1])
