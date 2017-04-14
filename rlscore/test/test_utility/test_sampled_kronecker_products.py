import unittest
import random as pyrandom

import numpy as np

from scipy.sparse import lil_matrix

from rlscore.utilities import sampled_kronecker_products

class Test(unittest.TestCase):
    
    def setUp(self):
        np.random.seed(100)
        pyrandom.seed(100)

    def test_sampled_vec_trick(self):
        
        # u  <- R * (M x N) * C * v
        
        def create_ind_vecs(rows, columns, incinds):
            rowstimescols = rows * columns
            indmatrix = np.arange(rowstimescols).reshape(rows, columns)
            row_inds, col_inds = np.unravel_index(indmatrix, (rows, columns))
            row_inds, col_inds = np.array(row_inds.ravel(order = 'F'), dtype = np.int32), np.array(col_inds.ravel(order = 'F'), dtype = np.int32)
            row_inds, col_inds = row_inds[incinds], col_inds[incinds]
            incidencemat = lil_matrix((len(row_inds), rowstimescols))
            for ind in range(len(row_inds)):
                i, j = row_inds[ind], col_inds[ind]
                incidencemat[ind, j * rows + i] = 1.
            return row_inds, col_inds, incidencemat
        
        def verbosity_wrapper(v, M, N, row_inds_N = None, row_inds_M = None, col_inds_N = None, col_inds_M = None):
            rc_m, cc_m = M.shape
            rc_n, cc_n = N.shape
    
            if row_inds_N is None:
                u_len = rc_m * rc_n
            else:
                u_len = len(row_inds_N)
            if col_inds_N is None:
                v_len = cc_m * cc_n
            else:
                v_len = len(col_inds_N)
            
            ss = ''
            
            if rc_m * v_len + cc_n * u_len < rc_n * v_len + cc_m * u_len:
                ss += 'rc_m * v_len + cc_n * u_len < rc_n * v_len + cc_m * u_len\n'
                if col_inds_N is None:
                    ss += 'col_inds_N is None\n'
                if row_inds_N is None:
                    ss += 'row_inds_N is None\n'
            else:
                ss += 'rc_m * v_len + cc_n * u_len >= rc_n * v_len + cc_m * u_len\n'
                if col_inds_N is None:
                    ss += 'col_inds_N is None\n'
                if row_inds_N is None:
                    ss += 'row_inds_N is None\n'
            print(ss)
            return sampled_kronecker_products.sampled_vec_trick(v, M, N, row_inds_N, row_inds_M, col_inds_N, col_inds_M)
        
        def dd(M_rows, M_columns, N_rows, N_columns):
            #V is a sparse matrix and v is a vector containing the known entries of V in an arbitrary order
            v_len = int(N_columns * M_columns / 5)
            v = np.random.rand(v_len)
            v_incinds = pyrandom.sample(range(N_columns * M_columns), v_len)
            v_row_inds, v_col_inds, C = create_ind_vecs(N_columns, M_columns, v_incinds)
            V = float('nan') * np.zeros((N_columns, M_columns))
            V[v_row_inds, v_col_inds] = v
            
            u_len = int(N_rows * M_rows / 5)
            u_incinds = pyrandom.sample(range(N_rows * M_rows), u_len)
            u_row_inds, u_col_inds, R = create_ind_vecs(N_rows, M_rows, u_incinds)
            
            M = np.random.rand(M_rows, M_columns)
            N = np.random.rand(N_rows, N_columns)
            
            V_replace_nans_with_zeros = (C.T * (C * V.ravel(order = 'F'))).reshape(N_columns, M_columns, order='F')
            res = verbosity_wrapper(v, M, N, u_col_inds, u_row_inds, v_col_inds, v_row_inds)
            contr = (R * (N * np.mat(V_replace_nans_with_zeros * np.mat(M).T)).ravel(order = 'F').T).T
            np.testing.assert_almost_equal(res.squeeze(), np.array(contr).squeeze())
            
            res = verbosity_wrapper(v, M, N, None, None, v_col_inds, v_row_inds)
            contr = ((N * np.mat(V_replace_nans_with_zeros * np.mat(M).T)).ravel(order = 'F').T).T
            np.testing.assert_almost_equal(res.squeeze(), np.array(contr).squeeze())
            
            v_full = np.random.rand(M_columns * N_columns)
            res = verbosity_wrapper(v_full, M, N, u_col_inds, u_row_inds, None, None)
            contr = (R * (N * np.mat(v_full.reshape(N_columns, M_columns, order='F') * np.mat(M).T)).ravel(order = 'F').T).T
            np.testing.assert_almost_equal(res.squeeze(), np.array(contr).squeeze())
            
            v_full = np.random.rand(M_columns * N_columns)
            res = verbosity_wrapper(v_full, M, N, None, None, None, None)
            contr = ((N * np.mat(v_full.reshape(N_columns, M_columns, order='F') * np.mat(M).T)).ravel(order = 'F').T).T
            np.testing.assert_almost_equal(res.squeeze(), np.array(contr).squeeze())
        
        print('\n')
        M_rows, M_columns = 30, 500
        N_rows, N_columns = 400, 60
        dd(M_rows, M_columns, N_rows, N_columns)
        
        M_rows, M_columns = 300, 50
        N_rows, N_columns = 40, 600
        dd(M_rows, M_columns, N_rows, N_columns)
