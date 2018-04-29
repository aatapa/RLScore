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

import warnings

import numpy as np
from scipy.sparse.linalg import LinearOperator

from rlscore.utilities import sampled_kronecker_products

class PairwiseKernelOperator(LinearOperator):
    
    """Operator consisting of weighted sums of Kronecker product kernels for possibly incomplete data set.
    Usable for training pairwise (dyadic) kernel models with iterative solvers, such as conjugate gradient.
    

    Parameters
    ----------
    
    K1 : {array-like, list of array-likes}, shape = [n_samples1, n_samples1]
        Kernel matrix 1

    K2 : {array-like, list of array-likes}, shape = [n_samples1, n_samples1]
        Kernel matrix 2
        
    row_inds_K1 : {array-like, list of equal length array-likes}, shape = [n_rows_of_operator]
        maps rows of the operator to rows of K1
        
    row_inds_K2 : {array-like, list of equal length array-likes}, shape = [n_rows_of_operator]
        maps rows of the operator to rows of K2
    
    col_inds_K1 : {array-like, list of equal length array-likes}, shape = [n_columns_of_operator]
        maps columns of the operator to columns of K1
    
    col_inds_K2 : {array-like, list of equal length array-likes}, shape = [n_columns_of_operator]
        maps columns of the operator to columns of K2
    
    weights : {list, tuple, array-like}, shape = [n_kernels], optional
        weights used by multiple pairwise kernel predictors
    """
    
    def __init__(self, K1, K2, row_inds_K1 = None, row_inds_K2 = None, col_inds_K1 = None, col_inds_K2 = None, weights = None):
        
        if isinstance(K1, (list, tuple)):
            #This should not happen if interface is used according to requirements
            if len(K1[0].shape) == 1:
                raise Exception("Initialized PairwiseKernelOperator object with 1D kernel matrix K1 in a list which is not supported")
            if len(K2[0].shape) == 1:
                raise Exception("Initialized PairwiseKernelOperator object with 1D kernel matrix K2 in a list which is not supported")
            
            if row_inds_K1 is None: rows = K1[0].shape[0] * K2[0].shape[0]
            else: rows = len(row_inds_K1[0])
            if col_inds_K1 is None: cols = K1[0].shape[1] * K2[0].shape[1]
            else: cols = len(col_inds_K1[0])
            self.dtype = K1[0].dtype
        else:
            #This should not happen if interface is used according to requirements
            if len(K1.shape) == 1:
                warnings.warn("Initialized PairwiseKernelOperator object with 1D kernel matrix K1")
                K1 = K1.reshape(1, K1.shape[0])
            if len(K2.shape) == 1:
                warnings.warn("Initialized PairwiseKernelOperator object with 1D kernel matrix K2")
                K2 = K2.reshape(1, K2.shape[0])
            
            if row_inds_K1 is None: rows = K1.shape[0] * K2.shape[0]
            else: rows = len(row_inds_K1)
            if col_inds_K1 is None: cols = K1.shape[1] * K2.shape[1]
            else: cols = len(col_inds_K1)
            self.dtype = K1.dtype
        self.shape = rows, cols
        self.K1, self.K2 = K1, K2
        self.row_inds_K1, self.row_inds_K2 = row_inds_K1, row_inds_K2
        self.col_inds_K1, self.col_inds_K2 = col_inds_K1, col_inds_K2
        self.weights = weights if not weights is None else np.ones(len(K1))
    
    def _matvec(self, v):
        
        if len(v.shape) > 1:
            v = np.squeeze(v)
        def inner_mv(v, K1i, K2i, col_inds_K1i, col_inds_K2i, row_inds_K1i = None, row_inds_K2i = None):
            if len(K1i.shape) == 1:
                K1i = K1i.reshape(1, K1i.shape[0])
            if len(K2i.shape) == 1:
                K2i = K2i.reshape(1, K2i.shape[0])
            if row_inds_K1i is not None:
                P = sampled_kronecker_products.sampled_vec_trick(
                    v,
                    K2i,
                    K1i,
                    row_inds_K2i,
                    row_inds_K1i,
                    col_inds_K2i,
                    col_inds_K1i)
            else:
                P = sampled_kronecker_products.sampled_vec_trick(
                    v,
                    K2i,
                    K1i,
                    None,
                    None,
                    col_inds_K2i,
                    col_inds_K1i)
                
                #P = P.reshape((K1i.shape[0], K2i.shape[0]), order = 'F')
            P = np.array(P)
            return P
        
        if isinstance(self.K1, (list, tuple)):
            P = None
            for i in range(len(self.K1)):
                K1i = self.K1[i]
                K2i = self.K2[i]
                col_inds_K1i = self.col_inds_K1[i]
                col_inds_K2i = self.col_inds_K2[i]
                if self.row_inds_K1 is not None:
                    row_inds_K1i = self.row_inds_K1[i]
                    row_inds_K2i = self.row_inds_K2[i]
                    Pi = inner_mv(v, K1i, K2i, col_inds_K1i, col_inds_K2i, row_inds_K1i, row_inds_K2i)
                else:
                    Pi = inner_mv(v, K1i, K2i, col_inds_K1i, col_inds_K2i, None, None)
                if P is None: P = self.weights[i] * Pi
                else: P = P + self.weights[i] * Pi
        else:
            P = inner_mv(v, self.K1, self.K2, self.col_inds_K1, self.col_inds_K2, self.row_inds_K1, self.row_inds_K2)
        #if len(origvshape) > 1:
        #    P = np.expand_dims(P, axis=1)
        return P
        