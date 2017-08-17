#
# The MIT License (MIT)
#
# This file is part of RLScore 
#
# Copyright (c) 2010 - 2016 Tapio Pahikkala, Antti Airola
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

from scipy import sparse as sp
import numpy as np
from rlscore import predictor
from rlscore.utilities import array_tools
from rlscore.learner import _greedy_rls
from rlscore.predictor import PredictorInterface

SELECTED_FEATURES = 'selected_features'
GREEDYRLS_LOO_PERFORMANCES = 'GreedyRLS_LOO_performances'
GREEDYRLS_TEST_PERFORMANCES = 'GreedyRLS_test_performances'

class GreedyRLS(PredictorInterface):
    """Linear time greedy forward selection for RLS.
    
    Performs greedy forward selection, where at each step the feature selected
    is the one whose addition leads to lowest leave-one-out mean squared error.

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape = [n_samples, n_features]
        Data matrix
    Y : {array-like}, shape = [n_samples] or [n_samples, n_labels] (if n_labels >1)
        Training set labels
    regparam : float (regparam > 0)
        regularization parameter
    subsetsize : int (0 < subsetsize <= n_labels)
        number of features to be selected
    bias : float, optional
        value of constant feature added to each data point (default 1)
        
    Attributes
    ----------
    selected : list, shape = [subsetsize]
        indices of the selected features, in the same order as they were selected
    performances : list, shape = [subsetsize]
        leave-one-out (mean squared) error after adding each feature
 
    Notes
    -----
    
    Computational complexity of training:
    m = n_samples, d = n_features, k = subsetsize, l = n_labels
    
    O(mdkl)
    
    Greedy RLS is described in  [1,2]. The extension of the method to multi-target learning
    was considered in [3].
    
    References
    ----------
           
    [1] Tapio Pahikkala, Antti Airola, and Tapio Salakoski.
    Speeding up Greedy Forward Selection for Regularized Least-Squares.
    Proceedings of The Ninth International Conference on Machine Learning and Applications,
    325-330, IEEE Computer Society, 2010.
    
    [2] Tapio Pahikkala, Sebastian Okser, Antti Airola, Tapio Salakoski, and Tero Aittokallio.
    Wrapper-based selection of genetic features in genome-wide association studies through fast matrix operations.
    Algorithms for Molecular Biology, 7(1):11, 2012.
    
    [3] Pekka Naula, Antti Airola, Tapio Salakoski, and Tapio Pahikkala.
    Multi-label learning under feature extraction budgets.
    Pattern Recognition Letters, 40:56--65, April 2014. 
    """
    
    def __init__(self, X, Y, subsetsize, regparam = 1.0, bias=1.0, callbackfun=None, **kwargs):
        self.callbackfun = callbackfun
        self.regparam = regparam
        if isinstance(X, sp.base.spmatrix):
            self.X = X.todense()
        else:
            self.X = X
        self.X = self.X.T
        self.X = self.X.astype("float64", copy=False)
        self.Y = np.mat(array_tools.as_2d_array(Y))
        #Number of training examples
        self.size = self.Y.shape[0]
        self.bias = bias
        self.measure = None
        fsize = X.shape[1]
        self.desiredfcount = subsetsize
        if not fsize >= self.desiredfcount:
            raise Exception('The overall number of features ' + str(fsize) + ' is smaller than the desired number ' + str(self.desiredfcount) + ' of features to be selected.')
        self.results = {}
        if 'use_default_callback' in kwargs and bool(kwargs['use_default_callback']):
            self.callbackfun = DefaultCallback(**kwargs)
        #The current version works only with the squared error measure
        self._solve_cython(self.regparam)
    
    
    def _solve_cython(self, regparam):
        self.regparam = regparam
        X = self.X
        Y = self.Y
        bias_slice = np.sqrt(self.bias)*np.mat(np.ones((1,X.shape[1]),dtype=np.float64))
        
        tsize = self.size
        fsize = X.shape[0]
        assert X.shape[1] == tsize
        self.A = np.mat(np.zeros((fsize, Y.shape[1])))
        
        rp = regparam
        rpinv = 1. / rp
        
        #Biaz
        cv = np.sqrt(self.bias)*np.mat(np.ones((1, tsize)))
        ca = rpinv * (1. / (1. + cv * rpinv * cv.T)) * (cv * rpinv)
        
        self.dualvec = rpinv * Y - cv.T * rpinv * (1. / (1. + cv * rpinv * cv.T)) * (cv * rpinv * Y)
        
        XT = X.T
        GXT = rpinv * XT - cv.T * rpinv * (1. / (1. + cv * rpinv * cv.T)) * ((cv * rpinv) * XT)
        diagG = []
        for i in range(tsize):
            diagGi = rpinv - cv.T[i, 0] * ca[0, i]
            diagG.append(diagGi)
        diagG = np.array(diagG)
        
        listX = []
        for ci in range(fsize):
            listX.append(X[ci])
        
        self.selected = []
        
        currentfcount = 0
        self.performances = []
        selectedvec = np.zeros(fsize, dtype = np.int16)
        tempvec1, tempvec2, tempvec3 = np.zeros(tsize), np.zeros(Y.shape[1]), np.zeros((tsize, Y.shape[1]))
        while currentfcount < self.desiredfcount:
            
            if not self.measure is None:
                self.bestlooperf = None
            else:
                self.bestlooperf = 9999999999.
            
            self.looperf = np.ones(fsize) * float('Inf')
            #'''
            bestcind = _greedy_rls.find_optimal_feature(np.array(Y),
                                                              np.array(X),
                                                              np.array(GXT),
                                                              diagG,
                                                              np.array(self.dualvec),
                                                              self.looperf,
                                                              fsize,
                                                              tsize,
                                                              Y.shape[1],
                                                              selectedvec,
                                                              tempvec1,
                                                              tempvec2,
                                                              tempvec3)
            #Reference code
            '''
            diagG = np.mat(diagG).T
            for ci in allinds:
                if ci in self.selected: continue
                cv = listX[ci_mapped]
                GXT_ci = GXT[:, ci_mapped]
                ca = GXT_ci * (1. / (1. + cv * GXT_ci))
                updA = self.dualvec - ca * (cv * self.dualvec)
                invupddiagG = 1. / (diagG - np.multiply(ca, GXT_ci))
                
                if not self.measure is None:
                    loopred = Y - np.multiply(invupddiagG, updA)
                    looperf_i = self.measure.multiOutputPerformance(Y, loopred)
                    if self.bestlooperf is None:
                        self.bestlooperf = looperf_i
                        bestcind = ci
                    if self.measure.comparePerformances(looperf_i, self.bestlooperf) > 0:
                        bestcind = ci
                        self.bestlooperf = looperf_i
                else:
                    #This default squared performance is a bit faster to compute than the one loaded separately.
                    loodiff = np.multiply(invupddiagG, updA)
                    print loodiff
                    foo
                    looperf_i = mean(np.multiply(loodiff, loodiff))
                    if looperf_i < self.bestlooperf:
                        bestcind = ci
                        self.bestlooperf = looperf_i
                self.looperf[ci] = looperf_i
            '''
            #'''
            self.bestlooperf = self.looperf[bestcind]
            self.looperf = np.mat(self.looperf)
            self.performances.append(self.bestlooperf)
            ci_mapped = bestcind
            cv = listX[ci_mapped]
            GXT_bci = GXT[:, ci_mapped]
            ca = GXT_bci * (1. / (1. + cv * GXT_bci))
            self.dualvec = self.dualvec - ca * (cv * self.dualvec)
            diagG = diagG - np.array(np.multiply(ca, GXT_bci)).reshape((self.size))
            GXT = GXT - ca * (cv * GXT)
            self.selected.append(bestcind)
            currentfcount += 1
            
            #Linear predictor with bias
            self.A[self.selected] = X[self.selected] * self.dualvec
            self.b = bias_slice * self.dualvec * np.sqrt(self.bias)
            self.predictor = predictor.LinearPredictor(self.A, self.b)
            
            if not self.callbackfun is None:
                self.callbackfun.callback(self)
        if not self.callbackfun is None:
            self.callbackfun.finished(self)
        self.A[self.selected] = X[self.selected] * self.dualvec
        self.b = bias_slice * self.dualvec * np.sqrt(self.bias)
        self.results[SELECTED_FEATURES] = self.selected
        self.results[GREEDYRLS_LOO_PERFORMANCES] = self.performances
        self.predictor = predictor.LinearPredictor(self.A, self.b)
    
    
    def _solve_new(self, regparam, floattype = np.float64):
        
        #Legacy code. Works only with a single output but can work with given performance measures and is faster than _solve_bu
        
        if not self.Y.shape[1] == 1:
            raise Exception('This variation of GreedyRLS supports only one output at a time. The output matrix is now of shape ' + str(self.Y.shape) + '.')
        self.regparam = regparam
        X = self.X
        Y = np.mat(self.Y, dtype=floattype)
        
        bias_slice = np.sqrt(self.bias)*np.mat(np.ones((1,X.shape[1]), dtype=floattype))
        
        tsize = self.size
        fsize = X.shape[0]
        assert X.shape[1] == tsize
        self.A = np.mat(np.zeros((fsize,1),dtype=floattype))
        
        rp = regparam
        rpinv = 1. / rp
        
        
        #Biaz
        cv = np.sqrt(self.bias)*np.mat(np.ones((1, tsize), dtype=floattype))
        ca = np.mat(rpinv * (1. / (1. + cv * rpinv * cv.T)) * (cv * rpinv), dtype=floattype)
        
        
        self.dualvec = rpinv * Y - cv.T * rpinv * (1. / (1. + cv * rpinv * cv.T)) * (cv * rpinv * Y)
        
        GXT = cv.T * np.mat((rpinv * (1. / (1. + cv * rpinv * cv.T)) * (cv * rpinv)) * X.T, dtype=floattype)
        tempmatrix = np.mat(np.zeros(X.T.shape, dtype=floattype))
        np.multiply(X.T, rpinv, tempmatrix)
        #tempmatrix = rpinv * X.T
        np.subtract(tempmatrix, GXT, GXT)
        
        diagG = []
        for i in range(tsize):
            diagGi = rpinv - cv.T[i, 0] * ca[0, i]
            diagG.append(diagGi)
        diagG = np.mat(diagG, dtype=floattype).T
        
        self.selected = []
        self.performances = []
        currentfcount = 0
        
        temp2 = np.mat(np.zeros(tempmatrix.shape, dtype=floattype))
        
        while currentfcount < self.desiredfcount:
            
            np.multiply(X.T, GXT, tempmatrix)
            XGXTdiag = np.sum(tempmatrix, axis = 0)
            
            XGXTdiag = 1. / (1. + XGXTdiag)
            np.multiply(GXT, XGXTdiag, tempmatrix)
            
            tempvec1 = np.multiply((X * self.dualvec).T, XGXTdiag)
            np.multiply(GXT, tempvec1, temp2)
            np.subtract(self.dualvec, temp2, temp2)
            
            np.multiply(tempmatrix, GXT, tempmatrix)
            np.subtract(diagG, tempmatrix, tempmatrix)
            np.divide(1, tempmatrix, tempmatrix)
            np.multiply(tempmatrix, temp2, tempmatrix)
            
            
            if not self.measure is None:
                np.subtract(Y, tempmatrix, tempmatrix)
                np.multiply(temp2, 0, temp2)
                np.add(temp2, Y, temp2)
                looperf = self.measure.multiTaskPerformance(temp2, tempmatrix)
                looperf = np.mat(looperf, dtype=floattype)
                if self.measure.iserror:
                    looperf[0, self.selected] = float('inf')
                    bestcind = np.argmin(looperf)
                    self.bestlooperf = np.amin(looperf)
                else:
                    looperf[0, self.selected] = - float('inf')
                    bestcind = np.argmax(looperf)
                    self.bestlooperf = np.amax(looperf)
            else:
                np.multiply(tempmatrix, tempmatrix, temp2)
                looperf = np.sum(temp2, axis = 0)
                looperf[0, self.selected] = float('inf')
                bestcind = np.argmin(looperf)
                self.bestlooperf = np.amin(looperf)
                self.loo_predictions = Y - tempmatrix[:, bestcind]
            
            self.looperf = looperf   #Needed in test_GreedyRLS module
            
            self.performances.append(self.bestlooperf)
            cv = X[bestcind]
            GXT_bci = GXT[:, bestcind]
            ca = GXT_bci * (1. / (1. + cv * GXT_bci))
            self.dualvec = self.dualvec - ca * (cv * self.dualvec)
            diagG = diagG - np.multiply(ca, GXT_bci)
            np.multiply(tempmatrix, 0, tempmatrix)
            np.add(tempmatrix, ca, tempmatrix)
            tempvec1 = cv * GXT
            np.multiply(tempmatrix, tempvec1, tempmatrix)
            np.subtract(GXT, tempmatrix, GXT)
            self.selected.append(bestcind)
            currentfcount += 1
            
            #Linear predictor with bias
            self.A[self.selected] = X[self.selected] * self.dualvec
            self.b = bias_slice * self.dualvec * np.sqrt(self.bias)
            self.predictor = predictor.LinearPredictor(self.A, self.b)
            
            if not self.callbackfun is None:
                self.callbackfun.callback(self)
        if not self.callbackfun is None:
            self.callbackfun.finished(self)
        self.A[self.selected] = X[self.selected] * self.dualvec
        self.b = bias_slice * self.dualvec * np.sqrt(self.bias)
        self.results[SELECTED_FEATURES] = self.selected
        self.results[GREEDYRLS_LOO_PERFORMANCES] = self.performances
        #self.results['predictor'] = self.getModel()
        self.predictor = predictor.LinearPredictor(self.A, self.b)
    
    
    def _solve_bu(self, regparam):
        self.regparam = regparam
        X = self.X
        Y = self.Y
        
        bias_slice = np.sqrt(self.bias)*np.mat(np.ones((1,X.shape[1]),dtype=np.float64))
        
        su = np.sum(X, axis = 1)
        cc = 0
        
        tsize = self.size
        fsize = X.shape[0]
        assert X.shape[1] == tsize
        self.A = np.mat(np.zeros((fsize, Y.shape[1])))
        
        rp = regparam
        rpinv = 1. / rp
        
        #Biaz
        cv = np.sqrt(self.bias)*np.mat(np.ones((1, tsize)))
        ca = rpinv * (1. / (1. + cv * rpinv * cv.T)) * (cv * rpinv)
        
        
        self.dualvec = rpinv * Y - cv.T * rpinv * (1. / (1. + cv * rpinv * cv.T)) * (cv * rpinv * Y)
        
        XT = X.T
        GXT = rpinv * XT - cv.T * rpinv * (1. / (1. + cv * rpinv * cv.T)) * ((cv * rpinv) * XT)
        diagG = []
        for i in range(tsize):
            diagGi = rpinv - cv.T[i, 0] * ca[0, i]
            diagG.append(diagGi)
        diagG = np.mat(diagG).T
        
        listX = []
        for ci in range(fsize):
            listX.append(X[ci])
        
        self.selected = []
        
        currentfcount = 0
        self.performances = []
        while currentfcount < self.desiredfcount:
            
            if not self.measure is None:
                bestlooperf = None
            else:
                bestlooperf = 9999999999.
            
            self.looperf = []
            for ci in range(fsize):
                ci_mapped = ci
                if ci in self.selected: continue
                cv = listX[ci_mapped]
                GXT_ci = GXT[:, ci_mapped]
                ca = GXT_ci * (1. / (1. + cv * GXT_ci))
                updA = self.dualvec - ca * (cv * self.dualvec)
                invupddiagG = 1. / (diagG - np.multiply(ca, GXT_ci))
                
                if not self.measure is None:
                    loopred = Y - np.multiply(invupddiagG, updA)
                    looperf_i = self.measure(Y, loopred)
                    if bestlooperf is None:
                        bestlooperf = looperf_i
                        bestcind = ci
                    if looperf_i < bestlooperf:
                        bestcind = ci
                        bestlooperf = looperf_i
                else:
                    #This default squared performance is a bit faster to compute than the one loaded separately.
                    loodiff = np.multiply(invupddiagG, updA)
                    looperf_i = np.mean(np.multiply(loodiff, loodiff))
                    if looperf_i < bestlooperf:
                        bestcind = ci
                        bestlooperf = looperf_i
                self.looperf.append(looperf_i)
            self.looperf = np.mat(self.looperf)
            self.bestlooperf = bestlooperf
            self.performances.append(bestlooperf)
            ci_mapped = bestcind
            cv = listX[ci_mapped]
            GXT_bci = GXT[:, ci_mapped]
            ca = GXT_bci * (1. / (1. + cv * GXT_bci))
            self.dualvec = self.dualvec - ca * (cv * self.dualvec)
            diagG = diagG - np.multiply(ca, GXT_bci)
            GXT = GXT - ca * (cv * GXT)
            self.selected.append(bestcind)
            #print self.selected
            #print bestlooperf
            currentfcount += 1
            
            #Linear predictor with bias
            self.A[self.selected] = X[self.selected] * self.dualvec
            self.b = bias_slice * self.dualvec * np.sqrt(self.bias)
            self.predictor = predictor.LinearPredictor(self.A, self.b)            
            if not self.callbackfun is None:
                self.callbackfun.callback(self)
        if not self.callbackfun is None:
            self.callbackfun.finished(self)
        self.A[self.selected] = X[self.selected] * self.dualvec
        self.b = bias_slice * self.dualvec * np.sqrt(self.bias)
        self.results[SELECTED_FEATURES] = self.selected
        self.results[GREEDYRLS_LOO_PERFORMANCES] = self.performances
        self.predictor = predictor.LinearPredictor(self.A, self.b)



class DefaultCallback(object):
    
    
    def __init__(self, **kwargs):
        if 'test_features' in kwargs and 'test_labels' in kwargs:
            self.test_features = kwargs['test_features']
            self.test_labels = kwargs['test_labels']
            if 'test_measure' in kwargs:
                self.test_measure = kwargs['test_measure']
                if isinstance(self.test_measure, str):
                    exec("from rlscore.measure import " + self.test_measure)
                    exec("self.test_measure = " + self.test_measure)
            else:
                self.test_measure = None
        else:
            self.test_features = None
    
    
    def callback(self, learner):
        print('')
        print('LOOCV mean squared error %f' %learner.bestlooperf)
        print('The indices of selected features %s' %str(learner.selected))
        if not self.test_features is None:
            mod = learner.predictor
            tpreds = mod.predict(self.test_features)
            if not self.test_measure is None:
                test_perf = self.test_measure(self.test_labels, tpreds)
            else:
                testdiff = self.test_labels - tpreds
                test_perf = np.mean(np.multiply(testdiff, testdiff))
            print('Test performance %f' %test_perf)
    
    def finished(self, learner):
        pass

