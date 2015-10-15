from numpy import *
import numpy.linalg as la
import unittest

from rlscore.learner.greedy_nfold_rls import GreedyNFoldRLS

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
        print "Testing the correctness of the GreedyRLS module."
        print
        print
        floattype = float64
        
        #m, n = 10, 30
        tsize, fsize = 100, 7
        desiredfcount = 5
        Xtrain = mat(random.rand(tsize, fsize))
        
        bias = 2.
        #bias = 0.
        
        bias_slice = sqrt(bias)*mat(ones((Xtrain.shape[0],1),dtype=float64))
        Xtrain_biased = hstack([Xtrain,bias_slice])
        ylen = 1
        Y = mat(zeros((tsize, ylen), dtype=floattype))
        Y = mat(random.rand(tsize, 1))
        
        
        #P = mat(zeros((m, objcount), dtype=float64))
        #Q = mat(zeros((objcount, m), dtype=float64))
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
        
        qidmap = {}
        for i in range(len(qidlist)):
            qid = qidlist[i]
            if qidmap.has_key(qid):
                sameqids = qidmap[qid]
                sameqids.append(i)
            else:
                qidmap[qid] = [i]
        indslist = []
        for qid in qidmap.keys():
            indslist.append(qidmap[qid])
        
        def complement(indices, m):
            compl = range(m)
            for ind in indices:
                compl.remove(ind)
            return compl
        
        selected = []
        
        rp = 1.
        currentfcount=0
        while currentfcount < desiredfcount:
            
            selected_plus_bias = selected + [fsize]
            bestlooperf = 9999999999.
            K = Xtrain_biased[:, selected_plus_bias]*Xtrain_biased[:,selected_plus_bias].T
            
            for ci in range(fsize):
                if ci in selected_plus_bias: continue
                cv = Xtrain_biased[:, ci]
                updK = Xtrain_biased[:, selected_plus_bias+[ci]]*Xtrain_biased[:, selected_plus_bias+[ci]].T #+ mat(ones((tsize,tsize)))
                updG = la.inv(updK + rp * mat(eye(tsize)))
                #print updG * C * Y
                updGG = la.inv(Xtrain_biased[:, selected_plus_bias+[ci]] * Xtrain_biased[:, selected_plus_bias+[ci]].T + rp * mat(eye(tsize)))
                #print updG * C * Y - updGG * C * Y
                                
                looperf = 0.
                
                for qi in range(len(indslist)):
                #for qi in range(1):
                    hoinds = indslist[qi]
                    hocompl = complement(hoinds, tsize)
                    updcutK = updK[ix_(hocompl, hocompl)]
                    updcrossK = updK[ix_(hoinds, hocompl)]
                    
                    loopred = updcrossK * la.inv(updcutK + rp * mat(eye(tsize-len(hoinds)))) * Y[hocompl]
                    #print loopred, qi, hoinds
                    #print 'lqodiff', C[ix_(hoinds, hoinds)]*Y[hoinds, 0] - loopred
                    lqodiff = Y[hoinds, 0] - loopred
                    #looperf += ((loopred - Y[hoinds, 0]).T * (loopred - Y[hoinds, 0]))[0, 0]
                    looperf += (lqodiff.T * lqodiff)[0, 0]
                                        
                    #invupdGGqid = la.inv(updGG[ix_(hoinds, hoinds)])
                    #lqopred = C[ix_(hoinds, hoinds)] * Y[hoinds, 0] - invupdGGqid * (updGG * C * Y)[hoinds, 0]
                    #print lqopred, qi, hoinds, sum(abs(loopred-lqopred))
                    
                    #foo = C[ix_(hoinds, hoinds)]*updcrossK*cutC * la.inv(cutC * updcutK * cutC+ rp * mat(eye(tsize-len(hoinds)))) * cutC * Y[hocompl]
                    #updGG = la.inv(C * Xtrain_biased[selected_plus_bias+[ci]].T * Xtrain_biased[selected_plus_bias+[ci]]* C+ rp * mat(eye(tsize)))
                    #invupdGGqid = la.inv(updGG[ix_(hoinds, hoinds)])
                    #bar = C[ix_(hoinds, hoinds)] * Y[hoinds, 0] - invupdGGqid * (updGG * C * Y)[hoinds, 0]
                    #print foo
                    #print bar
                
                print looperf,'bar'
                if looperf < bestlooperf:
                    bestcind = ci
                    bestlooperf = looperf
            
            selected.append(bestcind)
            print selected
            currentfcount += 1
        
        selected_plus_bias = selected + [fsize]
        K = Xtrain_biased[:, selected_plus_bias]*Xtrain_biased[:,selected_plus_bias].T
        G = la.inv(K+rp * mat(eye(tsize)))
        A = Xtrain_biased[:, selected_plus_bias].T*G*Y
        print A
        #A = mat(eye(fsize+1))[:,selected_plus_bias]*(Xtrain_biased[selected_plus_bias]*A)
        
        print
        print 'foo'        
    
        rpool = {}
        rpool['X'] = Xtrain
        rpool['Y'] = Y
        rpool['qids'] = indslist
        rpool['subsetsize'] = desiredfcount
        rpool['regparam'] = rp
        rpool['bias'] = 2.
        grls = GreedyNFoldRLS(**rpool)
        print grls.selected
        print grls.A[selected_plus_bias]
        


