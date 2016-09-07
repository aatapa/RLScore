#
# The MIT License (MIT)
#
# This file is part of RLScore 
#
# Copyright (c) 2013 - 2016 Tapio Pahikkala, Antti Airola
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

from math import sqrt

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import cg
from scipy import sparse
from numpy import ones

from rlscore.utilities import array_tools
from rlscore import predictor
from rlscore.measure import sqerror
from rlscore.predictor import PredictorInterface

class CGRLS(PredictorInterface):
    """Conjugate gradient RLS.
    
    Trains linear RLS using the conjugate gradient training algorithm. Suitable for
    large high-dimensional but sparse data.

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape = [n_samples, n_features]
        Data matrix
    regparam : float (regparam > 0)
        regularization parameter
    Y : {array-like}, shape = [n_samples] or [n_samples, 1]
        Training set labels
    bias : float, optional
        value of constant feature added to each data point (default 0)
        
    References
    ----------
    
    For an overview of regularized least-squares, and the conjugate gradient based optimization
    scheme see  [1].
    
    [1] Ryan Rifkin
    Everything old is new again : a fresh look at historical approaches in machine learning
    PhD Thesis, Massachusetts Institute of Technology, 2002
    """

    def __init__(self, X, Y, regparam = 1.0, bias = 1.0, callbackfun = None, **kwargs):
        self.Y = array_tools.as_2d_array(Y)
        self.X = csc_matrix(X.T)
        self.bias = bias
        self.regparam = regparam
        if self.bias != 0.:
            bias_slice = sqrt(self.bias)*np.mat(ones((1,self.X.shape[1]),dtype=np.float64))
            self.X = sparse.vstack([self.X,bias_slice]).tocsc()
        self.X_csr = self.X.tocsr()
        self.callbackfun = callbackfun
        self.results = {}
        regparam = self.regparam
        Y = self.Y
        X = self.X
        X_csr = self.X_csr
        def mv(v):
            return X.T*(X_csr*v)+regparam*v
        G = LinearOperator((X.shape[1],X.shape[1]), matvec=mv, dtype=np.float64)
        self.AA = []
        if not self.callbackfun is None:
            def cb(v):
                self.A = np.mat(v).T
                self.callbackfun.callback(self)
        else:
            cb = None
        try:
            self.A = np.mat(cg(G, Y, callback=cb)[0]).T
        except Finished:
            pass
        if self.callbackfun is not None:
            self.callbackfun.finished(self)
        self.A = X_csr*self.A
        if self.bias == 0.:
            self.b = np.mat(np.zeros((1,1)))
        else:
            self.b = sqrt(self.bias)*self.A[-1]
            self.A = self.A[:-1]
        #self.results['predictor'] = self.getModel()
        self.predictor = predictor.LinearPredictor(self.A, self.b)   

class EarlyStopCB(object):
    
    def __init__(self, X_valid, Y_valid, measure=sqerror, maxiter=10):
        self.X_valid = array_tools.as_matrix(X_valid)
        self.Y_valid = array_tools.as_2d_array(Y_valid)
        self.measure = measure
        self.bestperf = None
        self.bestA = None
        self.iter = 0
        self.last_update = 0
        self.maxiter = maxiter
    
    
    def callback(self, learner):
        A = learner.A
        b = learner.bias
        A = learner.X_csr*A
        if b == 0:
            b = np.mat(np.zeros((1,1)))
        else:
            b = sqrt(b)*A[-1]
            A = A[:-1]
        m = predictor.LinearPredictor(A,b)
        P = m.predict(self.X_valid)
        perf = self.measure(self.Y_valid,P)
        if self.bestperf is None or (self.measure.iserror == (perf < self.bestperf)):
            self.bestperf = perf
            self.bestA = learner.A
            self.last_update = 0
        else:
            self.iter += 1
            self.last_update += 1
        if self.last_update == self.maxiter:
            learner.A = np.mat(self.bestA)
            raise Finished("Done")
    
    
    def finished(self, learner):
        pass


class Finished(Exception):
    """Used to indicate that the optimization is finished and should
    be terminated."""

    def __init__(self, value):
        """Initialization
        
        @param value: the error message
        @type value: string"""
        self.value = value
    
    
    def __str__(self):
        return repr(self.value)  

    