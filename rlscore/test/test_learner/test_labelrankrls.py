import numpy as np
import numpy.linalg as la

import unittest
from rlscore.learner import QueryRankRLS

def mapQids(qids):
    """Maps qids to running numbering starting from zero, and partitions
    the training data indices so that each partition corresponds to one
    query"""
    #Used in FileReader, rls_predict
    qid_dict = {}
    folds = {}
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
        
        print("Testing the cross-validation routines of the QueryRankRLS module.\n")
        
        np.random.seed(100)
        floattype = np.float64
        
        m, n = 100, 400 #data, features
        Xtrain = np.mat(np.random.rand(m, n))
        K = Xtrain * Xtrain.T
        ylen = 1
        Y = np.mat(np.zeros((m, ylen), dtype=floattype))
        Y[:, 0] = np.sum(Xtrain, 1)
        
        
        labelcount = 5
        
        hoindices = range(labelcount)
        hocompl = list(set(range(m)) - set(hoindices))
        
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
        P = np.mat(np.zeros((m, objcount), dtype=np.float64))
        for i in range(m):
            qid = qidlist[i]
            P[i, qid] = 1.
        labelcounts = np.sum(P, axis=0)
        P = np.divide(P, np.sqrt(labelcounts))
        D = np.mat(np.ones((1, m), dtype=np.float64))
        L = np.multiply(np.eye(m), D) - P * P.T
        
        Kcv = K[np.ix_(hocompl, hocompl)]
        Lcv = L[np.ix_(hocompl, hocompl)]
        
        Xcv = Xtrain[hocompl]
        Xtest = Xtrain[hoindices]
        Yho = Y[hocompl]
        
        rpool = {}
        rpool["X"] = Xtrain
        rpool["Y"] = Y
        rpool["qids"] = mapQids(qidlist)
        primalrls = QueryRankRLS(**rpool)        
        
        rpool = {}
        rpool["X"] = K
        rpool['kernel'] = 'precomputed'
        rpool["Y"] = Y
        rpool["qids"] = mapQids(qidlist)        
        dualrls = QueryRankRLS(**rpool)
        
        rpool = {}
        rpool['X'] = Xcv
        rpool['Y'] = Yho
        rpool['qids'] = mapQids(qidlist_cv)
        primalrls_naive = QueryRankRLS(**rpool)

        rpool = {}
        rpool['X'] = Kcv
        rpool['kernel'] = 'precomputed'        
        rpool['Y'] = Yho
        #rpool['X'] = Xcv
        rpool['qids'] = mapQids(qidlist_cv)
        dualrls_naive = QueryRankRLS(**rpool)
        
        testkm = K[np.ix_(hocompl, hoindices)]
        
        loglambdas = range(-5, 5)
        for j in range(0, len(loglambdas)):
            regparam = 2. ** loglambdas[j]
            print
            print("Regparam 2^%1d" % loglambdas[j])
            
            
            print(str(np.squeeze(np.array((testkm.T * la.inv(Lcv * Kcv + regparam * np.eye(Lcv.shape[0])) * Lcv * Yho).T))) + ' Dumb HO')
            
            predhos = []
            primalrls_naive.solve(regparam)
            predho = primalrls_naive.predictor.predict(Xtest)
            print(str(predho.T) + ' Naive HO (primal)')
            predhos.append(predho)
            
            dualrls_naive.solve(regparam)
            predho = dualrls_naive.predictor.predict(testkm.T)
            print(str(predho.T) + ' Naive HO (dual)')
            predhos.append(predho)
            
            primalrls.solve(regparam)
            predho = np.squeeze(primalrls.computeHO(hoindices))
            print(str(predho.T) + ' Fast HO (primal)')
            predhos.append(predho)
            
            dualrls.solve(regparam)
            predho = np.squeeze(dualrls.computeHO(hoindices))
            print(str(predho.T) + ' Fast HO (dual)')
            predhos.append(predho)
            
            predho0 = predhos.pop(0)
            for predho in predhos:
                self.assertEqual(predho0.shape, predho.shape)
                for row in range(predho.shape[0]):
                    #for col in range(predho.shape[1]):
                    #    self.assertAlmostEqual(predho0[row,col],predho[row,col], places=5)
                        self.assertAlmostEqual(predho0[row],predho[row], places=5)
