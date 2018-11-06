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

def create_ind_vecs(rows, columns):
    rowstimescols = rows * columns
    indmatrix = np.arange(rowstimescols).reshape(rows, columns, order = 'C')
    row_inds, col_inds = np.unravel_index(indmatrix, (rows, columns))
    row_inds, col_inds = np.array(row_inds.ravel(order = 'F'), dtype = np.int32), np.array(col_inds.ravel(order = 'F'), dtype = np.int32)
    return row_inds, col_inds



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
        
        self.pko_adj = None
        self.original_row_inds_K1 = row_inds_K1
        self.original_row_inds_K2 = row_inds_K2
        self.original_col_inds_K1 = col_inds_K1
        self.original_col_inds_K2 = col_inds_K2
        
        def slice_off_unnecessarities(K1, K2, row_inds_K1 = None, row_inds_K2 = None, col_inds_K1 = None, col_inds_K2 = None):
            
            if len(K1.shape) == 1: K1 = K1.reshape(1, K1.shape[0])
            if len(K2.shape) == 1: K2 = K2.reshape(1, K2.shape[0])
            rc_m, cc_m = K2.shape
            rc_n, cc_n = K1.shape
            
            if row_inds_K1 is None:
                row_inds_K1, row_inds_K2 = create_ind_vecs(rc_n, rc_m)
            if col_inds_K1 is None:
                col_inds_K1, col_inds_K2 = create_ind_vecs(cc_n, cc_m)
            
            v_len = len(col_inds_K1)
            u_len = len(row_inds_K1)
            
            row_inds_K1 = np.atleast_1d(np.squeeze(np.asarray(row_inds_K1, dtype=np.int32)))
            row_inds_K2 = np.atleast_1d(np.squeeze(np.asarray(row_inds_K2, dtype=np.int32)))
            assert len(row_inds_K1) == len(row_inds_K2)
            assert np.min(row_inds_K1) >= 0
            assert np.min(row_inds_K2) >= 0
            assert np.max(row_inds_K1) < rc_n
            assert np.max(row_inds_K2) < rc_m
            
            col_inds_K1 = np.atleast_1d(np.squeeze(np.asarray(col_inds_K1, dtype=np.int32)))
            col_inds_K2 = np.atleast_1d(np.squeeze(np.asarray(col_inds_K2, dtype=np.int32)))
            assert len(col_inds_K1) == len(col_inds_K2)
            assert np.min(col_inds_K1) >= 0
            assert np.min(col_inds_K2) >= 0
            assert np.max(col_inds_K1) < cc_n
            assert np.max(col_inds_K2) < cc_m
            
            ui, ii = np.unique(row_inds_K1, return_inverse=True)
            nui = np.arange(len(ui))
            K1 = K1[ui]
            row_inds_K1 = nui[ii]
            
            ui, ii = np.unique(col_inds_K1, return_inverse=True)
            nui = np.arange(len(ui))
            K1 = K1[:, ui]
            col_inds_K1 = nui[ii]
            
            ui, ii = np.unique(row_inds_K2, return_inverse=True)
            nui = np.arange(len(ui))
            K2 = K2[ui]
            row_inds_K2 = nui[ii]
            
            ui, ii = np.unique(col_inds_K2, return_inverse=True)
            nui = np.arange(len(ui))
            K2 = K2[:, ui]
            col_inds_K2 = nui[ii]
            
            if len(K1.shape) == 1:
                K1 = K1[..., np.newaxis]
            if len(K2.shape) == 1:
                K2 = K2[..., np.newaxis]
            
            #These have to be re-assigned due to the above shape changes
            rc_m, cc_m = K2.shape
            rc_n, cc_n = K1.shape
            
            K1 = np.array(K1, order = 'C')
            K2 = np.array(K2, order = 'C')
            
            #if rc_m * v_len + cc_n * u_len < rc_n * v_len + cc_m * u_len:
            #    temp = np.zeros((cc_n, rc_m), order = 'F')
            #else:
            #    temp = np.zeros((rc_n, cc_m), order = 'C')
            temp = None
            
            row_inds_K1 = np.atleast_1d(np.squeeze(np.asarray(row_inds_K1, dtype=np.int32)))
            row_inds_K2 = np.atleast_1d(np.squeeze(np.asarray(row_inds_K2, dtype=np.int32)))
            col_inds_K1 = np.atleast_1d(np.squeeze(np.asarray(col_inds_K1, dtype=np.int32)))
            col_inds_K2 = np.atleast_1d(np.squeeze(np.asarray(col_inds_K2, dtype=np.int32)))
            return K1, K2, row_inds_K1, row_inds_K2, col_inds_K1, col_inds_K2, temp
        
        if isinstance(K1, (list, tuple)):
            #These must be copied because the operator may modify the matrices
            K1 = K1.copy()
            K2 = K2.copy()
            self.temp = [None for i in range(len(K1))]
            if row_inds_K1 is None:
                row_inds_K1 = [None for i in range(len(K1))]
                row_inds_K2 = [None for i in range(len(K2))]
            else:
                row_inds_K1 = row_inds_K1.copy()
            if col_inds_K1 is None:
                col_inds_K1 = [None for i in range(len(K1))]
                col_inds_K2 = [None for i in range(len(K2))]
            else:
                row_inds_K2 = row_inds_K2.copy()
            col_inds_K1 = col_inds_K1.copy()
            col_inds_K2 = col_inds_K2.copy()
            for i in range(len(K1)):
                K1i = K1[i]
                K2i = K2[i]
                col_inds_K1i = col_inds_K1[i]
                col_inds_K2i = col_inds_K2[i]
                row_inds_K1i = row_inds_K1[i]
                row_inds_K2i = row_inds_K2[i]
                K1[i], K2[i], row_inds_K1[i], row_inds_K2[i], col_inds_K1[i], col_inds_K2[i], self.temp[i] = slice_off_unnecessarities(K1i, K2i, row_inds_K1i, row_inds_K2i, col_inds_K1i, col_inds_K2i)
            self.dtype = K1[0].dtype
            self.shape = len(row_inds_K1[0]), len(col_inds_K1[0])
        else:
            K1, K2, row_inds_K1, row_inds_K2, col_inds_K1, col_inds_K2, self.temp = slice_off_unnecessarities(K1, K2, row_inds_K1, row_inds_K2, col_inds_K1, col_inds_K2)
            self.dtype = K1.dtype
            self.shape = len(row_inds_K1), len(col_inds_K1)
        
        self.K1, self.K2 = K1, K2
        self.row_inds_K1, self.row_inds_K2 = row_inds_K1, row_inds_K2
        self.col_inds_K1, self.col_inds_K2 = col_inds_K1, col_inds_K2
        self.weights = weights if not weights is None else np.ones(len(K1))
        self.x_after = None#np.zeros((self.shape[0]))
    
    def _matvec(self, v):
        
        if len(v.shape) > 1:
            v = np.squeeze(v)
        
        def inner_mv(v, K1i, K2i, col_inds_K1i, col_inds_K2i, row_inds_K1i, row_inds_K2i, temp):
            P = sampled_kronecker_products.sampled_vec_trick(
                v,
                K2i,
                K1i,
                row_inds_K2i,
                row_inds_K1i,
                col_inds_K2i,
                col_inds_K1i,
                temp,
                self.x_after)
            P = np.array(P)
            return P
        
        if isinstance(self.K1, (list, tuple)):
            P = None
            for i in range(len(self.K1)):
                K1i = self.K1[i]
                K2i = self.K2[i]
                col_inds_K1i = self.col_inds_K1[i]
                col_inds_K2i = self.col_inds_K2[i]
                row_inds_K1i = self.row_inds_K1[i]
                row_inds_K2i = self.row_inds_K2[i]
                tempi = self.temp[i]
                Pi = inner_mv(v, K1i, K2i, col_inds_K1i, col_inds_K2i, row_inds_K1i, row_inds_K2i, tempi)
                if P is None: P = self.weights[i] * Pi
                else: P = P + self.weights[i] * Pi
        else:
            P = inner_mv(v, self.K1, self.K2, self.col_inds_K1, self.col_inds_K2, self.row_inds_K1, self.row_inds_K2, self.temp)
        #if len(origvshape) > 1:
        #    P = np.expand_dims(P, axis=1)
        return P
    
    def _adjoint(self):
        if self.pko_adj is None:
            if isinstance(self.K1, (list, tuple)):
                K1_adj = [K1.T for K1 in self.K1]
                K2_adj = [K2.T for K2 in self.K2]
                self.pko_adj = PairwiseKernelOperator(K1_adj, K2_adj, self.col_inds_K1, self.col_inds_K2, self.row_inds_K1, self.row_inds_K2, self.weights)
            else:
                K1_adj = self.K1.T
                K2_adj = self.K2.T
                self.pko_adj = PairwiseKernelOperator(K1_adj, K2_adj, self.col_inds_K1, self.col_inds_K2, self.row_inds_K1, self.row_inds_K2, self.weights)
        return self.pko_adj
        