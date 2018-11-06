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
from rlscore.utilities import pairwise_kernel_operator

class PairwisePredictorInterface(object):
    
    """Computes predictions for test examples.

    Parameters
    ----------
    X1 : {array-like}, shape = [n_samples1, n_features1]
        first test data matrix
    X2 : {array-like}, shape = [n_samples2, n_features2]
        second test data matrix
    inds_X1pred : array of indices, optional
        rows of X1, for which predictions are needed
    inds_X2pred : array of indices, optional
        rows of X2, for which predictions are needed
        
    Note
    ----
    
    If using kernels, give kernel matrices K1 and K2 as arguments instead of X1 and X2   
    """
        
    def predict(self, X1 = None, X2 = None, inds_X1pred = None, inds_X2pred = None, pko = None):
        """Computes predictions for test examples.
    
        Parameters
        ----------
        X1 : {array-like}, shape = [n_samples1, n_features1]
            first test data matrix
        X2 : {array-like}, shape = [n_samples2, n_features2]
            second test data matrix
        inds_X1pred : array of indices, optional
            rows of X1, for which predictions are needed
        inds_X2pred : array of indices, optional
            rows of X2, for which predictions are needed
            
        Notes
        -----
        
        If using kernels, give kernel matrices K1 and K2 as arguments instead of X1 and X2   
        """
        return self.predictor.predict(X1, X2, inds_X1pred, inds_X2pred, pko)

class KernelPairwisePredictor(object):
    
    """Pairwise kernel predictor

    Parameters
    ----------
    A : {array-like}, shape = [n_train_pairs]
        dual coefficients
    inds_K1training : list of indices, shape = [n_train_pairs], optional
        maps dual coefficients to instances of of type 1, not needed if learning from complete data (i.e. n_train_pairs = n_samples1*n_samples2)
    inds_K2training : list of indices, shape = [n_train_pairs], optional
        maps dual coefficients to instances of of type 2, not needed if learning from complete data (i.e. n_train_pairs = n_samples1*n_samples2)
    weights : {list, tuple, array-like}, shape = [n_kernels], optional
         weights used by multiple pairwise kernel predictors  
    
    Attributes
    ----------
    A : {array-like}, shape = [n_train_pairs]
        dual coefficients
    inds_K1training : list of indices, shape = [n_train_pairs] or None
        maps dual coefficients to instances of of type 1, not needed if learning from complete data (i.e. n_train_pairs = n_samples1*n_samples2)
    inds_K2training : list of indices, shape = [n_train_pairs] or None
        maps dual coefficients to instances of of type 2, not needed if learning from complete data (i.e. n_train_pairs = n_samples1*n_samples2)
    weights : {list, tuple, array-like}, shape = [n_kernels], optional
         weights used by multiple pairwise kernel predictors  
    """
    
    def __init__(self, A, inds_K1training = None, inds_K2training = None, weights = None):
        self.A = A
        self.inds_K1training, self.inds_K2training = inds_K1training, inds_K2training
        if weights is not None: self.weights = weights
        else: self.weights = None
    
    
    def predict(self, K1pred = None, K2pred = None, inds_K1pred = None, inds_K2pred = None, pko = None):
        """Computes predictions for test examples.

        Parameters
        ----------
        K1pred : {array-like, list of equally shaped array-likes}, shape = [n_samples1, n_train_pairs]
            the first part of the test data matrix
        K2pred : {array-like, list of equally shaped array-likes}, shape = [n_samples2, n_train_pairs]
            the second part of the test data matrix
        inds_K1pred : list of indices, shape = [n_test_pairs], optional
            maps rows of K1pred to vector of predictions P. If not supplied, predictions are computed for all possible test pair combinations.
        inds_K2pred : list of indices, shape = [n_test_pairs], optional
            maps rows of K2pred to vector of predictions P. If not supplied, predictions are computed for all possible test pair combinations.
            
        Returns
        ----------
        P : array, shape = [n_test_pairs] or [n_samples1*n_samples2]
            predictions, either ordered according to the supplied row indices, or if no such are supplied by default
            prediction for (K1[i], K2[j]) maps to P[i + j*n_samples1].
        """
        if pko == None:
            pko = pairwise_kernel_operator.PairwiseKernelOperator(K1pred,
                                                                  K2pred,
                                                                  inds_K1pred,
                                                                  inds_K2pred,
                                                                  self.inds_K1training,
                                                                  self.inds_K2training,
                                                                  self.weights)
        return pko.matvec(self.A)


class LinearPairwisePredictor(object):
    
    """Linear pairwise predictor.
    
    Parameters
    ----------
    W : {array-like}, shape = [n_features1 * n_features2]
        primal coefficients for the Kronecker product features
        
    Attributes
    ----------
    W : {array-like}, shape = [n_features1 * n_features2]
        primal coefficients for the Kronecker product features
    
    """
    
    def __init__(self, W, inds_X1training = None, inds_X2training = None, weights = None):
        self.W = W
        self.inds_X1training, self.inds_X2training = inds_X1training, inds_X2training
        if weights is not None: self.weights = weights
        else: self.weights = None
    
    
    def predict(self, X1pred, X2pred, inds_X1pred = None, inds_X2pred = None, pko = None):
        """Computes predictions for test examples.

        Parameters
        ----------
        X1pred : array-like, shape = [n_samples1, n_features1]
            the first part of the test data matrix
        X2pred : array-like, shape = [n_samples2, n_features2]
            the second part of the test data matrix
        inds_X1pred : list of indices, shape = [n_test_pairs], optional
            maps rows of X1pred to vector of predictions P. If not supplied, predictions are computed for all possible test pair combinations.
        inds_X2pred : list of indices, shape = [n_test_pairs], optional
            maps rows of X2pred to vector of predictions P. If not supplied, predictions are computed for all possible test pair combinations.
            
        Returns
        ----------
        P : array, shape = [n_test_pairs] or [n_samples1*n_samples2]
            predictions, either ordered according to the supplied row indices, or if no such are supplied by default
            prediction for (X1[i], X2[j]) maps to P[i + j*n_samples1].
        """
        
        if pko == None:
            pko = pairwise_kernel_operator.PairwiseKernelOperator(X1pred,
                                                                  X2pred,
                                                                  inds_X1pred,
                                                                  inds_X2pred,
                                                                  None,
                                                                  None,
                                                                  self.weights)
        return pko.matvec(self.W)

