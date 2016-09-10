#
# The MIT License (MIT)
#
# This file is part of RLScore 
#
# Copyright (c) 2012, 2015, 2016 Tapio Pahikkala, Antti Airola
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
from scipy import sparse as sp


def as_2d_array(A, allow_sparse = False):
    #Interprets the input as 2d-array
    if allow_sparse and sp.issparse(A):
        s = np.sum(A.data)
        if s == np.inf or s == -np.inf:
            raise ValueError("Sparse matrix contains infinity")
        elif np.isnan(s):
            raise ValueError("Sparse matrix contains NaN")
        return A        
    if not allow_sparse and sp.issparse(A):
        A = A.todense()
    A = np.array(A, copy = False)
    shape = A.shape
    if not np.issubdtype(A.dtype, int) and not np.issubdtype(A.dtype, float):
        raise ValueError("Argument array contains non-numerical data") 
    if not len(shape) < 3:
        raise ValueError("Argument array of incorrect shape: expected 1D or 2D array, got %d dimensions" %len(shape))
    s = np.sum(A)
    if s == np.inf or s == -np.inf:
        raise ValueError("Array contains infinity")
    elif np.isnan(s):
        raise ValueError("Array contains NaN")
    if len(A.shape) == 1:
        A = A.reshape((A.shape[0], 1))
    elif len(A.shape) == 0:
        A = A.reshape((1,1))
    return A

def as_index_list(I, maxind):
    I = np.array(I, dtype=np.long, copy=False)
    if len(I.shape) != 1:
        raise ValueError("Index list should be one dimensional")
    if len(I) == 0:
        raise ValueError("Index list cannot be empty")
    minval = np.min(I)
    maxval = np.max(I)
    if minval < 0 or maxval >= maxind:
        raise IndexError("Index outside allowed range %d ... %d" %(0, maxind-1))
    return I
    
    

def as_dense_matrix(A):
    """Returns the input as matrix

    Parameters
    ----------
    A: {array-like, sparse matrix}, shape = 2D
    
    Returns
    -------
    A : np.matrix
    """
    if sp.issparse(A):
        return A.todense()
    else:
        return np.mat(A)
    
def as_matrix(A):
    """Returns the input as matrix or sparse matrix

    Parameters
    ----------
    A: {array-like, sparse matrix}, shape = 2D
    
    Returns
    -------
    A : {matrix, sparse matrix}
    """
    if sp.issparse(A):
        return A
    else:
        return np.mat(A)
    
def as_array(A):
    """Returns the input as dense array

    Parameters
    ----------
    A: {array-like, sparse matrix}, shape = 2D
    
    Returns
    -------
    A : {array}
    """
    if sp.issparse(A):
        A = A.todense()
    return np.asarray(A)

def spmat_resize(A, fdim):
    """Resizes the number of columns in sparse matrix to fdim, either removing or adding columns.

    Parameters
    ----------
    A: sparse matrix, size = [n_rows, n_cols]
    
    Returns
    -------
    A : csr_matrix, size = [n_rows, fdim]
    """
    if fdim < A.shape[1]:
        #Row slicing is efficient only for csr_matrix
        A = sp.csc_matrix(A)[:,:fdim]
    elif fdim > A.shape[1]:
        diff = fdim - A.shape[1]
        A = sp.hstack([A,sp.lil_matrix((A.shape[0],diff), dtype=np.float64)])
    A = sp.csr_matrix(A)
    return A
        
    