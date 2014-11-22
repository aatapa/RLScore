

import pyximport; pyximport.install()

import numpy as np

from rlscore.utilities import sparse_kronecker_multiplication_tools


def compute_subset_of_matprod_entries(*args):
    sparse_kronecker_multiplication_tools.compute_subset_of_matprod_entries(*args)

def sparse_mat_from_left(*args):
    sparse_kronecker_multiplication_tools.sparse_mat_from_left(*args)

def sparse_mat_from_right(*args):
    sparse_kronecker_multiplication_tools.sparse_mat_from_right(*args)

def x_gets_C_times_M_kron_N_times_B_times_v(v, M, N, row_inds_C, col_inds_C, row_inds_B, col_inds_B):
    rc_m, cc_m = M.shape
    rc_n, cc_n = N.shape
    u_len = len(row_inds_C)
    v_len = len(row_inds_B)
    
    if rc_m * v_len + cc_n * u_len < rc_n * v_len + cc_m * u_len:
    #if False:
    #if True:
        #print 'foo'
        temp = np.zeros((cc_n, rc_m))
        sparse_kronecker_multiplication_tools.sparse_mat_from_left(temp, v, M.T, row_inds_B, col_inds_B, v_len, rc_m)
        x_after = np.zeros((u_len))
        sparse_kronecker_multiplication_tools.compute_subset_of_matprod_entries(x_after, N, temp, row_inds_C, col_inds_C, u_len, cc_n)
        return x_after
    else:
        #print 'bar'
        temp = np.zeros((rc_n, cc_m))
        sparse_kronecker_multiplication_tools.sparse_mat_from_right(temp, N, v, row_inds_B, col_inds_B, v_len, rc_n)
        x_after = np.zeros((u_len))
        sparse_kronecker_multiplication_tools.compute_subset_of_matprod_entries(x_after, temp, M.T, row_inds_C, col_inds_C, u_len, cc_m)
        return x_after

def x_gets_A_kron_B_times_sparse_v(v, A, B, row_inds, col_inds):
    rc_a, cc_a = A.shape
    rc_b, cc_b = B.shape
    nzc_v = len(row_inds)
    len_c = rc_a * cc_b
    
    if rc_a * cc_a * cc_b + cc_b * nzc_v < rc_b * cc_a * cc_b + cc_a * nzc_v:
    #if False:
    #if True:
        #print 'foo'
        temp = np.zeros((cc_a, cc_b))
        sparse_kronecker_multiplication_tools.sparse_mat_from_left(temp, v, B, row_inds, col_inds, nzc_v, cc_b)
        temp = np.dot(A, temp)
        return temp.reshape((len_c,), order = 'F')
    else:
        #print 'bar'
        temp = np.zeros((rc_a, rc_b))
        sparse_kronecker_multiplication_tools.sparse_mat_from_right(temp, A, v, row_inds, col_inds, nzc_v, rc_a)
        temp = np.dot(temp, B)
        return temp.reshape((len_c,), order = 'F')


def x_gets_subset_of_A_kron_B_times_v(v, A, B, row_inds, col_inds):
    rc_a, cc_a = A.shape
    rc_b, cc_b = B.shape
    nzc_x = len(row_inds)
    
    x_after = np.zeros(nzc_x)
    temp = v.reshape((cc_a, rc_b), order = 'F')
    
    if rc_a * cc_a * cc_b + cc_b * nzc_x < rc_b * cc_a * cc_b + cc_a * nzc_x:
    #if False:
    #if True:
        temp = np.dot(A, temp)
        sparse_kronecker_multiplication_tools.compute_subset_of_matprod_entries(x_after, temp, B, row_inds, col_inds, nzc_x, rc_b)
        return x_after
    else:
        temp = np.dot(temp, B)
        sparse_kronecker_multiplication_tools.compute_subset_of_matprod_entries(x_after, A, temp, row_inds, col_inds, nzc_x, cc_a)
        return x_after




if __name__=="__main__":
    np.random.seed(100)
    v_rows, v_columns = 3, 5
    u_rows, u_columns = 4, 5
    M = np.random.rand(u_columns, v_columns)
    N = np.random.rand(u_rows, v_rows)
    Y_train = np.random.rand(v_rows, v_columns)
    
    def create_ind_vecs(rows, columns, incinds):
        rowstimescols = rows * columns
        indmatrix = np.arange(rowstimescols).T.reshape(rows, columns)
        #print indmatrix
        row_inds, col_inds = np.unravel_index(indmatrix, (rows, columns))
        row_inds, col_inds = np.array(row_inds.ravel(order = 'F'), dtype = np.int32), np.array(col_inds.ravel(order = 'F'), dtype = np.int32)
        #print row_inds, col_inds
        row_inds, col_inds = row_inds[incinds], col_inds[incinds]
        incidencemat = np.zeros((len(row_inds), rowstimescols))
        for ind in range(len(row_inds)):
            i, j = row_inds[ind], col_inds[ind]
            #Y_train_nonzeros.append(Y_train[i, j])
            #Y_alt.append(Y_train[i, j])
            #B[ind, i * columns + j] = 1.
            incidencemat[ind, j * rows + i] = 1.
            #incidencemat[ind, i * columns + j] = 1.
        return row_inds, col_inds, incidencemat
    
    #v_incinds = range(v_rows*v_columns)
    v_incinds = [0, 1, 3, 4, 7, 8, 9, 14]
    v_row_inds, v_col_inds, B = create_ind_vecs(v_rows, v_columns, v_incinds)
    #u_incinds = range(u_rows*u_columns)
    u_incinds = [0, 1, 4, 5, 6, 10, 12]
    u_row_inds, u_col_inds, C = create_ind_vecs(u_rows, u_columns, u_incinds)
    #print B
    #print C
    
    
    Y_sparsified = np.dot(B.T, np.dot(B, Y_train.ravel(order = 'F'))).reshape(v_rows, v_columns, order = 'F')
    vec_Y_cut = Y_train[v_row_inds, v_col_inds]
    Y_sparsified = np.dot(B.T, vec_Y_cut).reshape(v_rows, v_columns, order='F')
    print
    print np.dot(C, np.dot(N, np.dot(Y_sparsified, M.T)).ravel(order = 'F'))
    foo = x_gets_C_times_M_kron_N_times_B_times_v(vec_Y_cut, M, N, u_row_inds, u_col_inds, v_row_inds, v_col_inds)
    print foo

