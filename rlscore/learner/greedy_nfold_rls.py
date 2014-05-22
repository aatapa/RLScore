
from numpy import *
import scipy
import scipy.sparse as sp

from abstract_learner import AbstractSupervisedLearner
from abstract_learner import AbstractIterativeLearner
from rlscore import data_sources
from rlscore.measure.sq_mprank_measure import sqmprank

class GreedyNFoldRLS(AbstractSupervisedLearner, AbstractIterativeLearner):
    
    def loadResources(self):
        """
        Loads the resources from the previously set resource pool.
        
        @raise Exception: when some of the resources required by the learner is not available in the ResourcePool object.
        """
        AbstractIterativeLearner.loadResources(self)
        
        Y = self.resource_pool[data_sources.TRAIN_LABELS]
        self.Y = Y
        #Number of training examples
        self.size = Y.shape[0]
        if not Y.shape[1] == 1:
            raise Exception('GreedyRLS currently supports only one output at a time. The output matrix is now of shape ' + str(Y.shape) + '.')
        
        X = self.resource_pool[data_sources.TRAIN_FEATURES]
        self.setDataMatrix(X.T)
        if self.resource_pool.has_key('bias'):
            self.bias = float(self.resource_pool['bias'])
        else:
            self.bias = 0.
        if self.resource_pool.has_key(data_sources.PERFORMANCE_MEASURE):
            self.measure = self.resource_pool[data_sources.PERFORMANCE_MEASURE]
        else:
            self.measure = None
        qids = self.resource_pool[data_sources.TRAIN_QIDS]
        if not self.resource_pool.has_key(data_sources.CVFOLDS):
            self.resource_pool[data_sources.CVFOLDS] = qids
        self.setQids(qids)
        self.results = {}
    
    
    def setQids(self, qids):
        """Sets the qid parameters of the training examples. The list must have as many qids as there are training examples.
        
        @param qids: A list of qid parameters.
        @type qids: List of integers."""
        
        self.qidlist = [-1 for i in range(self.size)]
        for i in range(len(qids)):
            for j in qids[i]:
                if j >= self.size:
                    raise Exception("Index %d in query out of training set index bounds" %j)
                elif j < 0:
                    raise Exception("Negative index %d in query, query indices must be non-negative" %j)
                else:
                    self.qidlist[j] = i
        if -1 in self.qidlist:
            raise Exception("Not all training examples were assigned a query")
        
        
        self.qidmap = {}
        for i in range(len(self.qidlist)):
            qid = self.qidlist[i]
            if self.qidmap.has_key(qid):
                sameqids = self.qidmap[qid]
                sameqids.append(i)
            else:
                self.qidmap[qid] = [i]
        self.indslist = []
        for qid in self.qidmap.keys():
            self.indslist.append(self.qidmap[qid])
    
    
    def setDataMatrix(self, X):
        """
        Sets the label data for RLS.
        
        @param X: Features of the training examples.
        @type X: scipy sparse matrix
        """
        if isinstance(X, scipy.sparse.base.spmatrix):
            self.X = X.todense()
        else:
            self.X = X
    
    
    def train(self):
        regparam = float(self.resource_pool[data_sources.TIKHONOV_REGULARIZATION_PARAMETER])
        self.regparam = regparam
        self.solve_bu(regparam)
    
    
    def solve_bu(self, regparam):
        """Trains RLS with the given value of the regularization parameter
        
        @param regparam: value of the regularization parameter
        @type regparam: float
        """
        
        self.regparam = regparam
        
        
        X = self.X
        Y = self.Y
        
        if not hasattr(self, "bias"):
            self.bias = 0.
        
        tsize = self.size
        fsize = X.shape[0]
        assert X.shape[1] == tsize
        
        rp = regparam
        rpinv = 1. / rp
        
        
        if not self.resource_pool.has_key('subsetsize'):
            raise Exception("Parameter 'subsetsize' must be given.")
        desiredfcount = int(self.resource_pool['subsetsize'])
        if not fsize >= desiredfcount:
            raise Exception('The overall number of features ' + str(fsize) + ' is smaller than the desired number ' + str(desiredfcount) + ' of features to be selected.')
        
        
        
        if self.resource_pool.has_key('calculate_test_error'):
            calculate_test_error = self.resource_pool['calculate_test_error']
            if calculate_test_error == 'True':
                calculate_test_error = True
                self.testY = self.resource_pool[data_sources.TEST_LABELS]
                self.testX = self.resource_pool[data_sources.PREDICTION_FEATURES].todense()
                self.testQids = self.resource_pool[data_sources.PREDICTION_QIDS].readQids()
                
                self.testperformances = []
                
                self.testqidmap = {}
                for i in range(len(self.testQids)):
                    qid = self.testQids[i]
                    if self.testqidmap.has_key(qid):
                        sameqids = self.testqidmap[qid]
                        sameqids.append(i)
                    else:
                        self.testqidmap[qid] = [i]
                self.testindslist = []
                for qid in self.testqidmap.keys():
                    self.testindslist.append(self.testqidmap[qid])
            else:
                calculate_test_error = False
        else:
            calculate_test_error = False
        
        
        
        
        #Biaz
        cv = sqrt(self.bias)*mat(ones((1, tsize)))
        ca = rpinv * (1. / (1. + cv * rpinv * cv.T)) * (cv * rpinv)
        
        
        self.A = rpinv * Y - cv.T * rpinv * (1. / (1. + cv * rpinv * cv.T)) * (cv * rpinv * Y)
        
        XT = X.T
        GXT = rpinv * XT - cv.T * rpinv * (1. / (1. + cv * rpinv * cv.T)) * ((cv * rpinv) * XT)
        #diagG = []
        #invdiagG = []
        yac = []
        yyac = []
        
        #for qid in self.qidmap.keys():
        for inds in self.indslist:
            #inds = self.qidmap[key]
            #diagGqid = rpinv * mat(eye(len(inds))) - cv.T[inds, 0] * ca[0, inds]
            u = cv.T[inds, 0]
            v = ca[0, inds]
            #M = rp * mat(eye(len(inds)))
            #invdiagGqid = M - rp * rp * u * (1. / (-1. + rp * v * u)) * v
            #invdiagGqid = la.inv(diagGqid)
            temp = rp * GXT[inds] - rp * rp * u * (1. / (-1. + rp * v * u)) * (v * GXT[inds])
            #yac.append(invdiagGqid * GXT[inds])
            yac.append(temp)
            temp = rp * self.A[inds] - rp * rp * u * (1. / (-1. + rp * v * u)) * (v * self.A[inds])
            yyac.append(temp)
        
        listX = []
        for ci in range(fsize):
            listX.append(X[ci])
        
        self.selected = []
        
        currentfcount = 0
        self.performances = []
        while currentfcount < desiredfcount:
            
            #if not self.measure == None:
            #    bestlqocvperf = None
            #else:
            bestlqocvperf = float('inf')
            
            for ci in range(fsize):
                if ci in self.selected: continue
                cv = listX[ci]
                GXT_ci = GXT[:, ci]
                const = 1. / (1. + cv * GXT_ci)[0, 0]
                #ca = GXT_ci * (1. / (1. + cv * GXT_ci))
                #ca = GXT_ci * const
                cvA = (const * (cv * self.A))[0, 0]
                updA = self.A - GXT_ci * cvA
                #invupddiagG = 1. / (diagG - multiply(ca, GXT_ci))
                lqocvperf = 0.
                for qi in range(len(self.indslist)):
                    inds = self.indslist[qi]
                    V = GXT_ci[inds].T
                    MVT = yac[qi][:, ci]
                    gamma = (1. / (-const ** -1. + V * MVT))[0, 0]
                    lqodiff = yyac[qi] - cvA * MVT - gamma * MVT * (MVT.T * updA[inds])
                    lqocvperf += (lqodiff.T * lqodiff)[0, 0]
                
                #print lqocvperf,'barb'
                if lqocvperf < bestlqocvperf:
                    bestcind = ci
                    bestlqocvperf = lqocvperf
                
                '''
                if not self.measure == None:
                    loopred = Y - multiply(invupddiagG, updA)
                    looperf = self.measure.multiOutputPerformance(Y, loopred)
                    if bestlooperf == None:
                        bestlooperf = looperf
                        bestcind = ci
                    if self.measure.comparePerformances(looperf, bestlooperf) > 0:
                        bestcind = ci
                        bestlooperf = looperf
                else:
                    #This default squared performance is a bit faster to compute than the one loaded separately.
                    loodiff = multiply(invupddiagG, updA)
                    looperf = (loodiff.T * loodiff)[0, 0]
                    if looperf < bestlooperf:
                        bestcind = ci
                        bestlooperf = looperf
                '''
                
            self.bestlqocvperf = bestlqocvperf
            self.performances.append(bestlqocvperf)
            cv = listX[bestcind]
            GXT_ci = GXT[:, bestcind]
            const = (1. / (1. + cv * GXT_ci))[0, 0]
            #ca = GXT_ci
            cvA = const * cv * self.A
            self.A = self.A - GXT_ci * cvA
            cvGXT = const * cv * GXT
            GXT = GXT - GXT_ci * cvGXT
            for qi in range(len(self.indslist)):
                inds = self.indslist[qi]
                #diagGqid = diagG[qi]
                #upddiagGqid = diagGqid - const * GXT_ci[inds] * GXT_ci[inds].T
                #diagG[qi] = upddiagGqid
                #invdiagGqid = invdiagG[qi]
                #M = invdiagG[qi]
                V = GXT_ci[inds].T
                MVT = yac[qi][:, bestcind]
                gammaMVT = MVT * (1. / (-const ** -1. + V * MVT))
                #invupddiagGqid = M - const * MVT * (1. / (-1. + const * V * MVT)) * MVT.T
                #invupddiagGqid = invdiagGqid - invdiagGqid * const * GXT_ci[inds] * (1. / (-1. + GXT_ci[inds].T * invdiagGqid * const * GXT_ci[inds])) * GXT_ci[inds].T * invdiagGqid
                #invdiagG[qi] = invupddiagGqid
                #yac[qi] = invupddiagGqid * GXT[inds]
                #yac[qi] = yac[qi] - const * MVT * cvGXT - const * MVT * (1. / (-1. + const * V * MVT)) * (MVT.T * GXT[inds])
                #yyac[qi] = invupddiagGqid * A[inds]
                yyac[qi] = yyac[qi] - MVT * cvA - gammaMVT * (MVT.T * self.A[inds])
                yac[qi] = yac[qi] - MVT * cvGXT - gammaMVT * (MVT.T * GXT[inds])
            #diagG = diagG - multiply(ca, GXT_ci)
            self.selected.append(bestcind)
            currentfcount += 1
            
            if calculate_test_error:
                if self.measure == None:
                    pm = sqmprank
                else:
                    pm = self.measure
                bias_slice = sqrt(self.bias) * mat(ones((1,X.shape[1]),dtype=float64))
                X_biased = vstack([X,bias_slice])
                selected_plus_bias = self.selected+[fsize]
                cutdiag = sp.lil_matrix((fsize+1, currentfcount + 1))
                for ci, col in zip(selected_plus_bias, range(currentfcount + 1)):
                    cutdiag[ci, col] = 1.
                W = cutdiag * (X_biased[selected_plus_bias] * self.A)
                bias_slice = sqrt(self.bias) * mat(ones((1,self.testX.shape[1]),dtype=float64))
                testX_biased = vstack([self.testX,bias_slice])
                #print testX_biased.T.shape, W.shape
                self.Y_predicted = testX_biased.T * W
            self.callback()
        self.finished()
        
        bias_slice = sqrt(self.bias) * mat(ones((1,X.shape[1]),dtype=float64))
        X = vstack([X,bias_slice])
        selected_plus_bias = self.selected+[fsize]
        cutdiag = sp.lil_matrix((fsize+1, currentfcount + 1))
        for ci, col in zip(selected_plus_bias, range(currentfcount + 1)):
            cutdiag[ci, col] = 1.
        self.A = cutdiag * (X[selected_plus_bias] * self.A)
        self.results[data_sources.SELECTED_FEATURES] = self.selected
        self.results[data_sources.GREEDYRLS_LOO_PERFORMANCES] = self.performances
        if calculate_test_error:
            self.results[data_sources.GREEDYRLS_TEST_PERFORMANCES] = self.testperformances
