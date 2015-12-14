

#import pyximport; pyximport.install()

import random as pyrandom

import numpy as np

from scipy.sparse import lil_matrix

from rlscore.utilities import _sampled_kronecker_products


def compute_subset_of_matprod_entries(*args):
    _sampled_kronecker_products.compute_subset_of_matprod_entries(*args)

def sparse_mat_from_left(*args):
    _sampled_kronecker_products.sparse_mat_from_left(*args)

def sparse_mat_from_right(*args):
    _sampled_kronecker_products.sparse_mat_from_right(*args)

def sampled_vec_trick(v, M, N, row_inds_R = None, col_inds_R = None, row_inds_C = None, col_inds_C = None):
    if len(M.shape) == 1:
        M = M[..., np.newaxis]
    rc_m, cc_m = M.shape
    if len(N.shape) == 1:
        N = N[..., np.newaxis]
    rc_n, cc_n = N.shape
    
    if row_inds_R == None:
        row_inds_R, col_inds_R = np.unravel_index(np.arange(rc_m * rc_n), (rc_n, rc_m), order = 'F')
        #Not sure why the next row is necessary
        row_inds_R, col_inds_R = np.array(row_inds_R, dtype = np.int32), np.array(col_inds_R, dtype = np.int32)
    else:
        assert len(row_inds_R) == len(col_inds_R)
    if row_inds_C == None:
        row_inds_C, col_inds_C = np.unravel_index(np.arange(cc_m * cc_n), (cc_n, cc_m), order = 'F')
        #Not sure why the next row is necessary
        row_inds_C, col_inds_C = np.array(row_inds_C, dtype = np.int32), np.array(col_inds_C, dtype = np.int32)
    else:
        assert len(row_inds_C) == len(col_inds_C)

    u_len = len(row_inds_R)
    v_len = len(row_inds_C)
    
    if rc_m * v_len + cc_n * u_len < rc_n * v_len + cc_m * u_len:
        temp = np.zeros((cc_n, rc_m))
        _sampled_kronecker_products.sparse_mat_from_left(temp, v, M.T, row_inds_C, col_inds_C, v_len, rc_m)
        x_after = np.zeros((u_len))
        _sampled_kronecker_products.compute_subset_of_matprod_entries(x_after, N, temp, row_inds_R, col_inds_R, u_len, cc_n)
    else:
        temp = np.zeros((rc_n, cc_m))
        _sampled_kronecker_products.sparse_mat_from_right(temp, N, v, row_inds_C, col_inds_C, v_len, rc_n)
        x_after = np.zeros((u_len))
        _sampled_kronecker_products.compute_subset_of_matprod_entries(x_after, temp, M.T, row_inds_R, col_inds_R, u_len, cc_m)
    return x_after


def x_gets_A_kron_B_times_sparse_v(v, M, N, row_inds, col_inds): #MVN=(N.T x M)v
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


def x_gets_subset_of_A_kron_B_times_v(v, M, N, row_inds, col_inds):
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




if __name__=="__main__":
    
    # u  <- R * (M x N) * C * v
    
    np.random.seed(100)
    pyrandom.seed(100)
    
    def create_ind_vecs(rows, columns, incinds):
        rowstimescols = rows * columns
        indmatrix = np.arange(rowstimescols).T.reshape(rows, columns)
        row_inds, col_inds = np.unravel_index(indmatrix, (rows, columns))
        row_inds, col_inds = np.array(row_inds.ravel(order = 'F'), dtype = np.int32), np.array(col_inds.ravel(order = 'F'), dtype = np.int32)
        row_inds, col_inds = row_inds[incinds], col_inds[incinds]
        incidencemat = lil_matrix((len(row_inds), rowstimescols))
        for ind in range(len(row_inds)):
            i, j = row_inds[ind], col_inds[ind]
            incidencemat[ind, j * rows + i] = 1.
        return row_inds, col_inds, incidencemat
    
    #V is a sparse matrix and v is a vector containing the known entries of V in an arbitrary order
    V_rows, V_columns = 300, 500
    v_len = V_rows * V_columns / 5
    v = np.random.rand(v_len)
    v_incinds = pyrandom.sample(range(V_rows * V_columns), v_len)
    v_row_inds, v_col_inds, C = create_ind_vecs(V_rows, V_columns, v_incinds)
    V = float('nan') * np.zeros((V_rows, V_columns))
    V[v_row_inds, v_col_inds] = v
    
    U_rows, U_columns = 400, 600
    u_len = U_rows * U_columns / 5
    u_incinds = pyrandom.sample(range(U_rows * U_columns), u_len)
    u_row_inds, u_col_inds, R = create_ind_vecs(U_rows, U_columns, u_incinds)
    
    M = np.random.rand(U_columns, V_columns)
    N = np.random.rand(U_rows, V_rows)
    
    V_replace_nans_with_zeros = (C.T * (C * V.ravel(order = 'F'))).reshape(V_rows, V_columns, order='F')
    #V_replace_nans_with_zeros = (C.T * v).reshape(V_rows, V_columns, order='F')
    print 
    print (R * (N * np.mat(V_replace_nans_with_zeros * np.mat(M).T)).ravel(order = 'F').T).T
    print sampled_vec_trick(v, M, N, u_row_inds, u_col_inds, v_row_inds, v_col_inds)

