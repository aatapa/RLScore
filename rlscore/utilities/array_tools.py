import numpy as np
from scipy import sparse as sp

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

def as_labelmatrix(A):
    """Returns the input as a matrix, a 1-dimensional array is treated as
    column vector.

    Parameters
    ----------
    A: {array-like}, shape = 1D or 2D
    
    Returns
    -------
    A : matrix
    """
    shape = A.shape
    assert 0 < len(shape) < 3
    if len(A.shape) == 1:
        A = np.mat(A).T
    else:
        A = np.mat(A)
    return A

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
        
    