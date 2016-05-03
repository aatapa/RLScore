

#import pyximport; pyximport.install()

import numpy as np

from rlscore.utilities import _sampled_kronecker_products


def _compute_subset_of_matprod_entries(*args):
    _sampled_kronecker_products.compute_subset_of_matprod_entries(*args)

def _sparse_mat_from_left(*args):
    _sampled_kronecker_products.sparse_mat_from_left(*args)

def _sparse_mat_from_right(*args):
    _sampled_kronecker_products.sparse_mat_from_right(*args)

# u  <- R * (M x N) * C * v
def sampled_vec_trick(v, M, N, row_inds_M = None, row_inds_N = None, col_inds_M = None, col_inds_N = None):
    
    assert len(v.shape) == 1
    
    if len(M.shape) == 1:
        M = M[..., np.newaxis]
    rc_m, cc_m = M.shape
    if len(N.shape) == 1:
        N = N[..., np.newaxis]
    rc_n, cc_n = N.shape
    
    if row_inds_N == None:
        u_len = rc_m * rc_n
        #row_inds_N, row_inds_M = np.unravel_index(np.arange(rc_m * rc_n), (rc_n, rc_m), order = 'F')
        #Not sure why the next row is necessary
        #row_inds_N, row_inds_M = np.array(row_inds_N, dtype = np.int32), np.array(row_inds_M, dtype = np.int32)
    else:
        u_len = len(row_inds_N)
        assert len(row_inds_N) == len(row_inds_M)
        assert np.min(row_inds_N) >= 0
        assert np.min(row_inds_M) >= 0
        assert np.max(row_inds_N) < rc_n
        assert np.max(row_inds_M) < rc_m
    if col_inds_N == None:
        v_len = cc_m * cc_n
        #col_inds_N, col_inds_M = np.unravel_index(np.arange(cc_m * cc_n), (cc_n, cc_m), order = 'F')
        #Not sure why the next row is necessary
        #col_inds_N, col_inds_M = np.array(col_inds_N, dtype = np.int32), np.array(col_inds_M, dtype = np.int32)
    else:
        v_len = len(col_inds_N)
        assert len(col_inds_N) == len(col_inds_M)
        assert len(col_inds_N) == v.shape[0]
        assert np.min(col_inds_N) >= 0
        assert np.min(col_inds_M) >= 0
        assert np.max(col_inds_N) < cc_n
        assert np.max(col_inds_M) < cc_m
    
    if rc_m * v_len + cc_n * u_len < rc_n * v_len + cc_m * u_len:
        if col_inds_N == None:
            temp = np.dot(v.reshape((cc_n, cc_m), order = 'F'), M.T)
        else:
            temp = np.zeros((cc_n, rc_m), order='C')
            M = np.array(M, order = 'FORTRAN')
            _sampled_kronecker_products.sparse_mat_from_left(temp, v, M.T, col_inds_N, col_inds_M, v_len, rc_m)
        if row_inds_N == None:
            x_after = np.dot(N, temp)
            x_after = x_after.reshape((u_len,), order = 'F')
        else:
            temp = np.array(temp, order = 'FORTRAN')
            N = np.array(N, order = 'C')
            x_after = np.zeros((u_len))
            _sampled_kronecker_products.compute_subset_of_matprod_entries(x_after, N, temp, row_inds_N, row_inds_M, u_len, cc_n)
    else:
        if col_inds_N == None:
            temp = np.dot(N, v.reshape((cc_n, cc_m), order = 'F'))
        else:
            temp = np.zeros((rc_n, cc_m), order = 'FORTRAN')
            N = np.array(N, order = 'FORTRAN')            
            _sampled_kronecker_products.sparse_mat_from_right(temp, N, v, col_inds_N, col_inds_M, v_len, rc_n)
        if row_inds_N == None:
            x_after = np.dot(temp, M.T)
            x_after = x_after.reshape((u_len,), order = 'F')
        else:
            temp = np.array(temp, order = 'C')
            M = np.array(M, order = 'C')  
            x_after = np.zeros((u_len))
            _sampled_kronecker_products.compute_subset_of_matprod_entries(x_after, temp, M.T, row_inds_N, row_inds_M, u_len, cc_m)
    return x_after


def _x_gets_A_kron_B_times_sparse_v(v, M, N, row_inds, col_inds): #MVN=(N.T x M)v
    if len(M.shape) == 1:
        M = M[..., np.newaxis]
    rc_m, cc_m = M.shape
    if len(N.shape) == 1:
        N = N[..., np.newaxis]
    rc_n, cc_n = N.shape
    nzc_v = len(row_inds)
    len_c = rc_m * cc_n
    
    if rc_m * cc_m * cc_n + cc_n * nzc_v < rc_n * cc_m * cc_n + cc_m * nzc_v:
        temp = np.zeros((cc_m, cc_n))
        _sampled_kronecker_products.sparse_mat_from_left(temp, v, N, row_inds, col_inds, nzc_v, cc_n)
        temp = np.dot(M, temp)
        return temp.reshape((len_c,), order = 'F')
    else:
        temp = np.zeros((rc_m, rc_n))
        _sampled_kronecker_products.sparse_mat_from_right(temp, M, v, row_inds, col_inds, nzc_v, rc_m)
        temp = np.dot(temp, N)
        return temp.reshape((len_c,), order = 'F')


def _x_gets_subset_of_A_kron_B_times_v(v, M, N, row_inds, col_inds):
    if len(M.shape) == 1:
        M = M[..., np.newaxis]
    rc_m, cc_m = M.shape
    if len(N.shape) == 1:
        N = N[..., np.newaxis]
    rc_n, cc_n = N.shape
    nzc_x = len(row_inds)
    
    x_after = np.zeros(nzc_x)
    temp = v.reshape((cc_m, rc_n), order = 'F')
    
    if rc_m * cc_m * cc_n + cc_n * nzc_x < rc_n * cc_m * cc_n + cc_m * nzc_x:
        temp = np.dot(M, temp)
        _sampled_kronecker_products.compute_subset_of_matprod_entries(x_after, temp, N, row_inds, col_inds, nzc_x, rc_n)
        return x_after
    else:
        temp = np.dot(temp, N)
        _sampled_kronecker_products.compute_subset_of_matprod_entries(x_after, M, temp, row_inds, col_inds, nzc_x, cc_m)
        return x_after


