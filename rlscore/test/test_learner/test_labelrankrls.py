import sys

from numpy import *
import numpy.linalg as la

from rlscore import data_sources
from rlscore.utilities import decomposition
import unittest
from rlscore.learner import LabelRankRLS
from rlscore.kernel import LinearKernel

def mapQids(qids):
    """Maps qids to running numbering starting from zero, and partitions
    the training data indices so that each partition corresponds to one
    query"""
    #Used in FileReader, rls_predict
    qid_dict = {}
    folds = {}
    qid_list = []
    counter = 0
    for index, qid in enumerate(qids):
        if not qid in qid_dict:
            qid_dict[qid] = counter
            folds[qid] = []
            counter += 1
        folds[qid].append(index)
    indslist = []
    for f in folds.values():
        indslist.append(f)
    return indslist

class Test(unittest.TestCase):
    
    def testLabelRankRLS(self):
        
        print
        print
        print
        print
        print "Testing the cross-validation routines of the LabelRankRLS module."
        print
        print
        
        random.seed(100)
        floattype = float64
        
        m, n = 100, 400
        Xtrain = mat(random.rand(n, m))
        K = Xtrain.T * Xtrain
        ylen = 1
        Y = mat(zeros((m, ylen), dtype=floattype))
        Y[:, 0] = sum(Xtrain, 0).T
        
        
        
        def complement(indices, m):
            compl = range(m)
            for ind in indices:
                compl.remove(ind)
            return compl
        
        objcount = 20
        labelcount = 5
        
        hoindices = range(labelcount)
        hocompl = complement(hoindices, m)
        
        size = m
        
        P = mat(zeros((m, objcount), dtype=float64))
        Q = mat(zeros((objcount, m), dtype=float64))
        qidlist = [0 for i in range(100)]
        for h in range(5, 12):
            qidlist[h] = 1
        for h in range(12, 32):
            qidlist[h] = 2
        for h in range(32, 34):
            qidlist[h] = 3
        for h in range(34, 85):
            qidlist[h] = 4
        for h in range(85, 100):
            qidlist[h] = 5
        qidlist_cv = qidlist[5: len(qidlist)]
        
        objcount = max(qidlist) + 1
        P = mat(zeros((m, objcount), dtype=float64))
        for i in range(m):
            qid = qidlist[i]
            P[i, qid] = 1.
        labelcounts = sum(P, axis=0)
        D = mat(zeros((1, m), dtype=float64))
        for i in range(m):
            qid = qidlist[i]
            D[0, i] = labelcounts[0, qid]
        L = multiply(eye(m), D) - P * P.T
        
        Kcv = K[ix_(hocompl, hocompl)]
        Ycv = Y[hocompl]
        Ktest = K[ix_(hoindices, hocompl)]
        Lcv = L[ix_(hocompl, hocompl)]
        
        Xcv = Xtrain[:, hocompl]
        #Pcv = P[hocompl]#KLUDGE!!!!!
        Pcv = P[ix_(hocompl, range(1, P.shape[1]))]#KLUDGE!!!!!
        Xtest = Xtrain[:, hoindices]
        Yho = Y[hocompl]
        
        
        #svals, evecs, U = decomposition.decomposeDataMatrix(Xtrain)
        
        rpool = {}
        rpool["train_features"] = Xtrain.T
        rpool["train_labels"] = Y
        rpool["train_qids"] = mapQids(qidlist)
        primalrls = LabelRankRLS.createLearner(**rpool)        
        
        #primalrls = LabelRankRLS()
        #primalrls.setDecomposition(svals, evecs, U)
        #primalrls.svecs = evecs
        #primalrls.svals = svals
        #primalrls.setLabels(Y)
        #primalrls.setQids(mapQids(qidlist))
        #primalrls = LabelRankRLS(Y, qidlist, svals, evecs, U=U)
        
        #svals, evecs = decomposition.decomposeKernelMatrix(K)
        rpool = {}
        rpool["kmatrix"] = K
        rpool["train_labels"] = Y
        rpool["train_qids"] = mapQids(qidlist)        
        dualrls = LabelRankRLS.createLearner(**rpool)
        #dualrls.setDecomposition(svals,evecs)
        #dualrls.svecs = evecs
        #dualrls.svals = svals        
        #dualrls.setLabels(Y)
        #dualrls.setQids(mapQids(qidlist))
        #dualrls = LabelRankRLS(Y, qidlist, svals, evecs)
        
        
        rpool = {}
        rpool[data_sources.TRAIN_FEATURES] = Xcv.T
        rpool[data_sources.TRAIN_LABELS] = Yho
        rpool[data_sources.KERNEL_OBJ] = LinearKernel.createKernel(**rpool)
        rpool[data_sources.TRAIN_QIDS] = mapQids(qidlist_cv)
        params = {}
        rpool[data_sources.PARAMETERS] = params
        primalrls_naive = LabelRankRLS.createLearner(**rpool)

        
        rpool = {}
        rpool[data_sources.KMATRIX] = Kcv
        rpool[data_sources.TRAIN_LABELS] = Yho
        rpool[data_sources.TRAIN_FEATURES] = Xcv.T
        rpool[data_sources.KERNEL_OBJ] = LinearKernel.createKernel(**rpool)
        rpool[data_sources.TRAIN_QIDS] = mapQids(qidlist_cv)
        params = {}
        rpool[data_sources.PARAMETERS] = params
        dualrls_naive = LabelRankRLS.createLearner(**rpool)
        
        
        
        testkm = K[ix_(hocompl, hoindices)]
        
        loglambdas = range(-5, 5)
        for j in range(0, len(loglambdas)):
            regparam = 2. ** loglambdas[j]
            print
            print "Regparam 2^%1d" % loglambdas[j]
            
            
            print (testkm.T * la.inv(Lcv * Kcv + regparam * eye(Lcv.shape[0])) * Lcv * Yho).T, 'Dumb HO'
            
            predhos = []
            primalrls_naive.solve(regparam)
            predpool = {}
            predpool[data_sources.PREDICTION_FEATURES]=Xtest.T
            predho = primalrls_naive.getModel().predictFromPool(predpool)
            print predho.T, 'Naive HO (primal)'
            predhos.append(predho)
            
            dualrls_naive.solve(regparam)
            predpool = {}
            predpool[data_sources.PREDICTION_FEATURES]=testkm.T
            predho = dualrls_naive.getModel().predictFromPool(predpool)
            print predho.T, 'Naive HO (dual)'
            predhos.append(predho)
            
            primalrls.solve(regparam)
            predho = primalrls.computeHO(hoindices)
            print predho.T, 'Fast HO (primal)'
            predhos.append(predho)
            
            dualrls.solve(regparam)
            predho = dualrls.computeHO(hoindices)
            print predho.T, 'Fast HO (dual)'
            predhos.append(predho)
            
            predho0 = predhos.pop(0)
            for predho in predhos:
                self.assertEqual(predho0.shape, predho.shape)
                for row in range(predho.shape[0]):
                    for col in range(predho.shape[1]):
                        self.assertAlmostEqual(predho0[row,col],predho[row,col], places=5)