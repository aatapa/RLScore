import sys
import unittest

from numpy import *
import numpy.linalg as la

from rlscore import data_sources
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
        
        random.seed(100)
        floattype = float64
        
        m, n, h = 30, 200, 10
        Xtrain = mat(random.rand(n, m))
        #trainlabels = sum(Xtrain, 0).T
        trainlabels = mat(random.rand(m, h))
        trainkm = Xtrain.T * Xtrain
        ylen = 1
        
        def complement(indices, m):
            compl = range(m)
            for ind in indices:
                compl.remove(ind)
            return compl
        
        L = mat(m * eye(m) - ones((m, m), dtype=floattype))
        
        hoindices = [5, 7]
        hoindices3 = [5, 7, 9]
        hocompl = complement(hoindices, m)
        hocompl3 = complement(hoindices3, m)
        
        loglambdas = range(-5, 5)
        for j in range(0, len(loglambdas)):
            regparam = 2. ** loglambdas[j]
            print
            print "Regparam 2^%1d" % loglambdas[j]
            
            Kcv = trainkm[ix_(hocompl, hocompl)]
            Ycv = trainlabels[hocompl]
            Ktest = trainkm[ix_(hocompl, hoindices)]
            
            Xcv = Xtrain[:, hocompl]
            Xtest = Xtrain[:, hoindices]
            
            Lcv = mat((m - 2) * eye(m - 2) - ones((m - 2, m - 2), dtype=floattype))
            
            oind = 1
            rpool = {}
            rpool['train_labels'] = Ycv
            rpool['train_features'] = Xcv.T
            rpool['regparam'] = regparam
            naivedualrls = AllPairsRankRLS.createLearner(**rpool)
            naivedualrls.solve(regparam)
            hopreds = []
            rpool = {}
            rpool[data_sources.PREDICTION_FEATURES] = Xtest.T
            hopred = naivedualrls.getModel().predictFromPool(rpool)
            print hopred[0, oind], hopred[1, oind], 'Naive'
            hopreds.append((hopred[0, oind], hopred[1, oind]))
            
            rpool = {}
            rpool['train_labels'] = trainlabels
            rpool['train_features'] = Xtrain.T
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
            rpool['train_features'] = Xtrain.T
            rpool['regparam'] = regparam
            hoprimalrls = AllPairsRankRLS.createLearner(**rpool)
            hoprimalrls.solve(regparam)
            hopred = hoprimalrls.computeHO(hoindices)
            print hopred[0, oind], hopred[1, oind], 'HO'
            hopreds.append((hopred[0, oind], hopred[1, oind]))
            
            hopred = Xtest.T * la.inv(Xcv * Lcv * Xcv.T + regparam * mat(eye(n))) * Xcv * Lcv * Ycv
            print hopred[0, oind], hopred[1, oind], 'Dumb (primal)'
            hopreds.append((hopred[0, oind], hopred[1, oind]))
            
            hopred0 = hopreds.pop(0)
            for hopred in hopreds:
                self.assertAlmostEqual(hopred0[0],hopred[0])
                self.assertAlmostEqual(hopred0[1],hopred[1])