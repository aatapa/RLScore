#
# The MIT License (MIT)
#
# This file is part of RLScore 
#
# Copyright (c) 2014 - 2016 Tapio Pahikkala, Antti Airola
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import scipy
import scipy.sparse as sp
import numpy as np

class GreedyNFoldRLS(object):
    
    def loadResources(self):
        """
        Loads the resources from the previously set resource pool.
        
        @raise Exception: when some of the resources required by the learner is not available in the ResourcePool object.
        """
        
        Y = self.resource_pool['Y']
        self.Y = Y
        #Number of training examples
        self.size = Y.shape[0]
        if not Y.shape[1] == 1:
            raise Exception('GreedyRLS currently supports only one output at a time. The output matrix is now of shape ' + str(Y.shape) + '.')
        
        X = self.resource_pool['X']
        self.setDataMatrix(X.T)
        if self.resource_pool.has_key('bias'):
            self.bias = float(self.resource_pool['bias'])
        else:
            self.bias = 0.
        if self.resource_pool.has_key('measure'):
            self.measure = self.resource_pool['measure']
        else:
            self.measure = None
        qids = self.resource_pool['qids']
        if not self.resource_pool.has_key('cross-validation_folds'):
            self.resource_pool['cross-validation_folds'] = qids
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
        regparam = float(self.resource_pool['regparam'])
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
                self.testY = self.resource_pool['test_labels']
                self.testX = self.resource_pool['prediction_features'].todense()
                self.testQids = self.resource_pool['test_qids'].readQids()
                
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
        cv = np.sqrt(self.bias)*np.mat(np.ones((1, tsize)))
        ca = rpinv * (1. / (1. + cv * rpinv * cv.T)) * (cv * rpinv)
        
        
        self.A = rpinv * Y - cv.T * rpinv * (1. / (1. + cv * rpinv * cv.T)) * (cv * rpinv * Y)
        
        XT = X.T
        GXT = rpinv * XT - cv.T * rpinv * (1. / (1. + cv * rpinv * cv.T)) * ((cv * rpinv) * XT)
        yac = []
        yyac = []
        
        for inds in self.indslist:
            u = cv.T[inds, 0]
            v = ca[0, inds]
            temp = rp * GXT[inds] - rp * rp * u * (1. / (-1. + rp * v * u)) * (v * GXT[inds])
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
            
            bestlqocvperf = float('inf')
            
            for ci in range(fsize):
                if ci in self.selected: continue
                cv = listX[ci]
                GXT_ci = GXT[:, ci]
                const = 1. / (1. + cv * GXT_ci)[0, 0]
                cvA = (const * (cv * self.A))[0, 0]
                updA = self.A - GXT_ci * cvA
                lqocvperf = 0.
                for qi in range(len(self.indslist)):
                    inds = self.indslist[qi]
                    V = GXT_ci[inds].T
                    MVT = yac[qi][:, ci]
                    gamma = (1. / (-const ** -1. + V * MVT))[0, 0]
                    lqodiff = yyac[qi] - cvA * MVT - gamma * MVT * (MVT.T * updA[inds])
                    lqocvperf += (lqodiff.T * lqodiff)[0, 0]
                
                if lqocvperf < bestlqocvperf:
                    bestcind = ci
                    bestlqocvperf = lqocvperf
                
                '''
                if not self.measure is None:
                    loopred = Y - multiply(invupddiagG, updA)
                    looperf = self.measure.multiOutputPerformance(Y, loopred)
                    if bestlooperf is None:
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
            cvA = const * cv * self.A
            self.A = self.A - GXT_ci * cvA
            cvGXT = const * cv * GXT
            GXT = GXT - GXT_ci * cvGXT
            for qi in range(len(self.indslist)):
                inds = self.indslist[qi]
                V = GXT_ci[inds].T
                MVT = yac[qi][:, bestcind]
                gammaMVT = MVT * (1. / (-const ** -1. + V * MVT))
                yyac[qi] = yyac[qi] - MVT * cvA - gammaMVT * (MVT.T * self.A[inds])
                yac[qi] = yac[qi] - MVT * cvGXT - gammaMVT * (MVT.T * GXT[inds])
            self.selected.append(bestcind)
            currentfcount += 1
            
            if calculate_test_error:
                bias_slice = np.sqrt(self.bias) * np.mat(np.ones((1,X.shape[1]),dtype=np.float64))
                X_biased = np.vstack([X,bias_slice])
                selected_plus_bias = self.selected+[fsize]
                cutdiag = sp.lil_matrix((fsize+1, currentfcount + 1))
                for ci, col in zip(selected_plus_bias, range(currentfcount + 1)):
                    cutdiag[ci, col] = 1.
                W = cutdiag * (X_biased[selected_plus_bias] * self.A)
                bias_slice = np.sqrt(self.bias) * np.mat(np.ones((1,self.testX.shape[1]),dtype=np.float64))
                testX_biased = np.vstack([self.testX,bias_slice])
                self.Y_predicted = testX_biased.T * W
            if not self.callbackfun is None:
                self.callbackfun.callback(self)
        if not self.callbackfun is None:
            self.callbackfun.finished(self)
        
        bias_slice = np.sqrt(self.bias) * np.mat(np.ones((1,X.shape[1]),dtype=np.float64))
        X = np.vstack([X,bias_slice])
        selected_plus_bias = self.selected+[fsize]
        cutdiag = sp.lil_matrix((fsize+1, currentfcount + 1))
        for ci, col in zip(selected_plus_bias, range(currentfcount + 1)):
            cutdiag[ci, col] = 1.
        self.A = cutdiag * (X[selected_plus_bias] * self.A)
        self.results['selected_features'] = self.selected
        self.results['GreedyRLS_LOO_performances'] = self.performances
        if calculate_test_error:
            self.results['GreedyRLS_test_performances'] = self.testperformances
