#
# The MIT License (MIT)
#
# This file is part of RLScore 
#
# Copyright (c) 2016 Tapio Pahikkala, Antti Airola
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

import numpy as np
from rlscore.utilities import sampled_kronecker_products

class PairwisePredictorInterface(object):
    
    """Computes predictions for test examples.

    Parameters
    ----------
    X1 : {array-like}, shape = [n_samples1, n_features1]
        first test data matrix
    X2 : {array-like}, shape = [n_samples2, n_features2]
        second test data matrix
    row_inds_X1pred : array of indices, optional
        rows of X1, for which predictions are needed
    row_inds_X2pred : array of indices, optional
        rows of X2, for which predictions are needed
        
    Note
    ----
    
    If using kernels, give kernel matrices K1 and K2 as arguments instead of X1 and X2   
    """
        
    def predict(self, X1, X2, row_inds_X1pred = None, row_inds_X2pred = None):
        return self.predictor.predict(X1, X2, row_inds_X1pred, row_inds_X2pred)

class KernelPairwisePredictor(object):
    
    """Pairwise kernel predictor

    Parameters
    ----------
    A : {array-like}, shape = [n_train_pairs]
        dual coefficients
    row_inds_K1training : list of indices, shape = [n_train_pairs], optional
        maps dual coefficients to rows of K1, not needed if learning from complete data (i.e. n_train_pairs = n_samples1*n_samples2)
    row_inds_K2training : list of indices, shape = [n_train_pairs], optional
        maps dual coefficients to rows of K2, not needed if learning from complete data (i.e. n_train_pairs = n_samples1*n_samples2)
    weights : {list, tuple, array-like}, shape = [n_kernels], optional
         weights used by multiple pairwise kernel predictors  
    
    Attributes
    ----------
    A : {array-like}, shape = [n_train_pairs]
        dual coefficients
    row_inds_K1training : list of indices, shape = [n_train_pairs] or None
        maps dual coefficients to rows of K1, not needed if learning from complete data (i.e. n_train_pairs = n_samples1*n_samples2)
    row_inds_K2training : list of indices, shape = [n_train_pairs] or None
        maps dual coefficients to rows of K2, not needed if learning from complete data (i.e. n_train_pairs = n_samples1*n_samples2)
    weights : {list, tuple, array-like}, shape = [n_kernels], optional
         weights used by multiple pairwise kernel predictors  
    """
    
    def __init__(self, A, row_inds_K1training = None, row_inds_K2training = None, weights = None):
        self.A = A
        self.row_inds_K1training, self.row_inds_K2training = row_inds_K1training, row_inds_K2training
        if not weights == None: self.weights = weights
    
    
    def predict(self, K1pred, K2pred, row_inds_K1pred = None, row_inds_K2pred = None):
        """Computes predictions for test examples.

        Parameters
        ----------
        K1pred : {array-like, list of equally shaped array-likes}, shape = [n_samples1, n_train_pairs]
            the first part of the test data matrix
        K2pred : {array-like, list of equally shaped array-likes}, shape = [n_samples2, n_train_pairs]
            the second part of the test data matrix
        row_inds_K1pred : list of indices, shape = [n_test_pairs], optional
            maps rows of K1pred to vector of predictions P. If not supplied, predictions are computed for all possible test pair combinations.
        row_inds_K2pred : list of indices, shape = [n_test_pairs], optional
            maps rows of K2pred to vector of predictions P. If not supplied, predictions are computed for all possible test pair combinations.
            
        Returns
        ----------
        P : array, shape = [n_test_pairs] or [n_samples1*n_samples2]
            predictions, either ordered according to the supplied row indices, or if no such are supplied by default
            prediction for (K1[i], K2[j]) maps to P[i + j*n_samples1].
        """
        def inner_predict(K1pred, K2pred, row_inds_K1pred = None, row_inds_K2pred = None):
            if len(K1pred.shape) == 1:
                K1pred = K1pred.reshape(1, K1pred.shape[0])
            if len(K2pred.shape) == 1:
                K2pred = K2pred.reshape(1, K2pred.shape[0])
            if row_inds_K1pred is not None:
                row_inds_K1pred = np.array(row_inds_K1pred, dtype = np.int32)
                row_inds_K2pred = np.array(row_inds_K2pred, dtype = np.int32)
                P = sampled_kronecker_products.sampled_vec_trick(
                    self.A,
                    K2pred,
                    K1pred,
                    row_inds_K2pred,
                    row_inds_K1pred,
                    self.row_inds_K2training,
                    self.row_inds_K1training)
            else:
                P = sampled_kronecker_products.sampled_vec_trick(
                    self.A,
                    K2pred,
                    K1pred,
                    None,
                    None,
                    self.row_inds_K2training,
                    self.row_inds_K1training)
                
                #P = P.reshape((K1pred.shape[0], K2pred.shape[0]), order = 'F')
            P = np.array(P)
            return P
        
        if isinstance(K1pred, (list, tuple)):
            P = None
            for i in range(len(K1pred)):
                K1i = K1pred[i]
                K2i = K2pred[i]
                Pi = inner_predict(K1i, K2i, row_inds_K1pred, row_inds_K2pred)
                if P == None: P = self.weights[i] * Pi
                else: P = P + self.weights[i] * Pi
            return P
        else:
            return inner_predict(K1pred, K2pred, row_inds_K1pred, row_inds_K2pred)


class LinearPairwisePredictor(object):
    
    """Linear pairwise predictor.
    
    Parameters
    ----------
    W : {array-like}, shape = [n_features1, n_features2]
        primal coefficients for the Kronecker product features
        
    Attributes
    ----------
    W : {array-like}, shape = [n_features1, n_features2]
        primal coefficients for the Kronecker product features
    
    """
    
    def __init__(self, W):
        self.W = W
    
    
    def predict(self, X1pred, X2pred, row_inds_X1pred = None, row_inds_X2pred = None):
        """Computes predictions for test examples.

        Parameters
        ----------
        X1pred : array-like, shape = [n_samples1, n_features1]
            the first part of the test data matrix
        X2pred : array-like, shape = [n_samples2, n_features2]
            the second part of the test data matrix
        row_inds_X1pred : list of indices, shape = [n_test_pairs], optional
            maps rows of X1pred to vector of predictions P. If not supplied, predictions are computed for all possible test pair combinations.
        row_inds_X2pred : list of indices, shape = [n_test_pairs], optional
            maps rows of X2pred to vector of predictions P. If not supplied, predictions are computed for all possible test pair combinations.
            
        Returns
        ----------
        P : array, shape = [n_test_pairs] or [n_samples1*n_samples2]
            predictions, either ordered according to the supplied row indices, or if no such are supplied by default
            prediction for (X1[i], X2[j]) maps to P[i + j*n_samples1].
        """
        if len(X1pred.shape) == 1:
            if self.W.shape[0] > 1:
                X1pred = X1pred[np.newaxis, ...]
            else:
                X1pred = X1pred[..., np.newaxis]
        if len(X2pred.shape) == 1:
            if self.W.shape[1] > 1:
                X2pred = X2pred[np.newaxis, ...]
            else:
                X2pred = X2pred[..., np.newaxis]
        if row_inds_X1pred is None:
            P = np.dot(np.dot(X1pred, self.W), X2pred.T)
        else:
            P = sampled_kronecker_products.sampled_vec_trick(
                    self.W.reshape((self.W.shape[0] * self.W.shape[1]), order = 'F'),
                    X2pred,
                    X1pred,
                    np.array(row_inds_X2pred, dtype = np.int32),
                    np.array(row_inds_X1pred, dtype = np.int32),
                    None,
                    None)
        return P.ravel(order = 'F')

