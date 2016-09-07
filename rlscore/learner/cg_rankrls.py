#
# The MIT License (MIT)
#
# This file is part of RLScore 
#
# Copyright (c) 2012 - 2016 Tapio Pahikkala, Antti Airola
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
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import cg
import scipy.sparse as sp

from rlscore import predictor
from rlscore.utilities import array_tools
from rlscore.measure import sqmprank
from rlscore.measure.measure_utilities import UndefinedPerformance
from rlscore.predictor import PredictorInterface
from rlscore.learner.query_rankrls import map_qids
from rlscore.measure.measure_utilities import qids_to_splits

class CGRankRLS(PredictorInterface):
    """Conjugate gradient RankRLS.
    
    Trains linear RankRLS using the conjugate gradient training algorithm. Suitable for
    large high-dimensional but sparse data.
    
    Parameters
    ----------
    X : {array-like, sparse matrix}, shape = [n_samples, n_features]
        Data matrix
    regparam : float (regparam > 0)
        regularization parameter
    Y : {array-like}, shape = [n_samples] or [n_samples, 1], optional
        Training set labels (alternative to: 'train_preferences')
    qids : list of n_queries index lists, optional
        Training set qids,  (can be supplied with 'Y')
 
       
    References
    ----------
    
    RankRLS algorithm is described in [1], using the conjugate gradient optimization
    together with early stopping was considered in detail in [2]. 
    
    [1] Tapio Pahikkala, Evgeni Tsivtsivadze, Antti Airola, Jouni Jarvinen, and Jorma Boberg.
    An efficient algorithm for learning to rank from preference graphs.
    Machine Learning, 75(1):129-165, 2009.
    
    [2] Antti Airola, Tapio Pahikkala, and Tapio Salakoski.
    Large Scale Training Methods for Linear RankRLS
    ECML/PKDD-10 Workshop on Preference Learning, 2010.
    """

    def __init__(self, X, Y, regparam = 1.0, qids = None, callbackfun=None, **kwargs):
        self.regparam = regparam
        self.callbackfun = None
        self.Y = array_tools.as_2d_array(Y)
        #Number of training examples
        self.size = Y.shape[0]
        if self.Y.shape[1] > 1:
            raise Exception('CGRankRLS does not currently work in multi-label mode')
        self.learn_from_labels = True
        self.callbackfun = callbackfun
        self.X = csc_matrix(X.T)
        if qids is not None:
            self.qids = map_qids(qids)
            self.splits = qids_to_splits(self.qids)
        else:
            self.qids = None
        regparam = self.regparam
        qids = self.qids
        if qids is not None:
            P = sp.lil_matrix((self.size, len(set(qids))))
            for qidind in range(len(self.splits)):
                inds = self.splits[qidind]
                qsize = len(inds)
                for i in inds:
                    P[i, qidind] = 1. / sqrt(qsize)
            P = P.tocsr()
            PT = P.tocsc().T
        else:
            P = 1./sqrt(self.size)*(np.mat(np.ones((self.size,1), dtype=np.float64)))
            PT = P.T
        X = self.X.tocsc()
        X_csr = X.tocsr()
        def mv(v):
            v = np.mat(v).T
            return X_csr*(X.T*v)-X_csr*(P*(PT*(X.T*v)))+regparam*v
        G = LinearOperator((X.shape[0],X.shape[0]), matvec=mv, dtype=np.float64)
        Y = self.Y
        if not self.callbackfun is None:
            def cb(v):
                self.A = np.mat(v).T
                self.b = np.mat(np.zeros((1,1)))
                self.callbackfun.callback(self)
        else:
            cb = None
        XLY = X_csr*Y-X_csr*(P*(PT*Y))
        try:
            self.A = np.mat(cg(G, XLY, callback=cb)[0]).T
        except Finished:
            pass
        self.b = np.mat(np.zeros((1,1)))
        self.predictor = predictor.LinearPredictor(self.A, self.b)
    
class PCGRankRLS(PredictorInterface):
    """Conjugate gradient RankRLS with pairwise preferences.
    
    Trains linear RankRLS using the conjugate gradient training algorithm. Suitable for
    large high-dimensional but sparse data.

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape = [n_samples, n_features]
        Data matrix
    regparam : float (regparam > 0)
        regularization parameter
    train_preferences : {array-like}, shape = [n_preferences, 2], optional
        Pairwise preference indices (alternative to: 'Y')
        The array contains pairwise preferences one pair per row, i.e. the data point
        corresponding to the first index is preferred over the data point corresponding
        to the second index.

 
       
    References
    ----------
    
    RankRLS algorithm is described in [1], using the conjugate gradient optimization
    together with early stopping was considered in detail in [2]. 
    
    [1] Tapio Pahikkala, Evgeni Tsivtsivadze, Antti Airola, Jouni Jarvinen, and Jorma Boberg.
    An efficient algorithm for learning to rank from preference graphs.
    Machine Learning, 75(1):129-165, 2009.
    
    [2] Antti Airola, Tapio Pahikkala, and Tapio Salakoski.
    Large Scale Training Methods for Linear RankRLS
    ECML/PKDD-10 Workshop on Preference Learning, 2010.
    """

    def __init__(self, X, train_preferences, regparam = 1., **kwargs):
        self.regparam = regparam
        self.callbackfun = None
        self.pairs = train_preferences
        self.X = csc_matrix(X.T)
        regparam = self.regparam
        X = self.X.tocsc()
        X_csr = X.tocsr()
        vals = np.concatenate([np.ones((self.pairs.shape[0]), dtype=np.float64), -np.ones((self.pairs.shape[0]), dtype=np.float64)])
        row = np.concatenate([np.arange(self.pairs.shape[0]),np.arange(self.pairs.shape[0])])
        col = np.concatenate([self.pairs[:,0], self.pairs[:,1]])
        coo = coo_matrix((vals, (row, col)), shape=(self.pairs.shape[0], X.shape[1]))
        pairs_csr = coo.tocsr()
        pairs_csc = coo.tocsc()
        def mv(v):
            vmat = np.mat(v).T
            ret = np.array(X_csr * (pairs_csc.T * (pairs_csr * (X.T * vmat))))+regparam*vmat
            return ret
        G = LinearOperator((X.shape[0], X.shape[0]), matvec=mv, dtype=np.float64)
        self.As = []
        M = np.mat(np.ones((self.pairs.shape[0], 1)))
        if not self.callbackfun is None:
            def cb(v):
                self.A = np.mat(v).T
                self.b = np.mat(np.zeros((1,1)))
                self.callbackfun.callback()
        else:
            cb = None
        XLY = X_csr * (pairs_csc.T * M)
        self.A = np.mat(cg(G, XLY, callback=cb)[0]).T
        self.b = np.mat(np.zeros((1,self.A.shape[1])))
        self.predictor = predictor.LinearPredictor(self.A, self.b)
    


class EarlyStopCB(object):
    
    def __init__(self, X_valid, Y_valid, qids_valid = None, measure=sqmprank, maxiter=10):
        self.X_valid = array_tools.as_matrix(X_valid)
        self.Y_valid = array_tools.as_2d_array(Y_valid)
        self.qids_valid = qids_to_splits(qids_valid)
        self.measure = measure
        self.bestperf = None
        self.bestA = None
        self.iter = 0
        self.last_update = 0
        self.maxiter = maxiter
    
    def callback(self, learner):
        m = predictor.LinearPredictor(learner.A, learner.b)
        P = m.predict(self.X_valid)
        if self.qids_valid:
            perfs = []
            for query in self.qids_valid:
                try:
                    perf = self.measure(self.Y_valid[query], P[query])
                    perfs.append(perf)
                except UndefinedPerformance:
                    pass
            perf = np.mean(perfs)
        else:
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

    