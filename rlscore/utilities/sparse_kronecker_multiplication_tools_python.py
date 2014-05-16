

import pyximport; pyximport.install()

import numpy as np

from rlscore.utilities import sparse_kronecker_multiplication_tools


def compute_subset_of_matprod_entries(*args):
    sparse_kronecker_multiplication_tools.compute_subset_of_matprod_entries(*args)

def sparse_mat_from_left(*args):
    sparse_kronecker_multiplication_tools.sparse_mat_from_left(*args)

def sparse_mat_from_right(*args):
    sparse_kronecker_multiplication_tools.sparse_mat_from_right(*args)

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
