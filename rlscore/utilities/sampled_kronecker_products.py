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

import numpy as np

from . import _sampled_kronecker_products


# u  <- R * (M x N) * C * v
def sampled_vec_trick(v, M, N, row_inds_M = None, row_inds_N = None, col_inds_M = None, col_inds_N = None):
    
    assert len(v.shape) == 1
    
    if len(M.shape) == 1:
        M = M[..., np.newaxis]
    rc_m, cc_m = M.shape
    if len(N.shape) == 1:
        N = N[..., np.newaxis]
    rc_n, cc_n = N.shape
    
    if row_inds_N is None:
        u_len = rc_m * rc_n
    else:
        row_inds_N = np.atleast_1d(np.squeeze(np.asarray(row_inds_N, dtype=np.int32)))
        row_inds_M = np.atleast_1d(np.squeeze(np.asarray(row_inds_M, dtype=np.int32)))
        u_len = len(row_inds_N)
        assert len(row_inds_N) == len(row_inds_M)
        assert np.min(row_inds_N) >= 0
        assert np.min(row_inds_M) >= 0
        assert np.max(row_inds_N) < rc_n
        assert np.max(row_inds_M) < rc_m
    if col_inds_N is None:
        v_len = cc_m * cc_n
    else:
        col_inds_N = np.atleast_1d(np.squeeze(np.asarray(col_inds_N, dtype=np.int32)))
        col_inds_M = np.atleast_1d(np.squeeze(np.asarray(col_inds_M, dtype=np.int32)))
        v_len = len(col_inds_N)
        assert len(col_inds_N) == len(col_inds_M)
        assert len(col_inds_N) == v.shape[0]
        assert np.min(col_inds_N) >= 0
        assert np.min(col_inds_M) >= 0
        assert np.max(col_inds_N) < cc_n
        assert np.max(col_inds_M) < cc_m
    
    if rc_m * v_len + cc_n * u_len < rc_n * v_len + cc_m * u_len:
        if col_inds_N is None:
            temp = np.array(np.dot(v.reshape((cc_n, cc_m), order = 'F'), M.T), order = 'F')
        else:
            temp = np.zeros((cc_n, rc_m), order='F')
            _sampled_kronecker_products.sparse_mat_from_left(temp, v, M.T, col_inds_N, col_inds_M, v_len, rc_m)
        if row_inds_N is None:
            x_after = np.dot(N, temp)
            x_after = x_after.reshape((u_len,), order = 'F')
        else:
            x_after = np.zeros((u_len))
            _sampled_kronecker_products.compute_subset_of_matprod_entries(x_after, N, temp, row_inds_N, row_inds_M, u_len, cc_n)
    else:
        if col_inds_N is None:
            temp = np.dot(N, v.reshape((cc_n, cc_m), order = 'F'))
        else:
            temp = np.zeros((rc_n, cc_m), order = 'C')
            _sampled_kronecker_products.sparse_mat_from_right(temp, N, v, col_inds_N, col_inds_M, v_len, rc_n)
        if row_inds_N is None:
            x_after = np.dot(temp, M.T)
            x_after = x_after.reshape((u_len,), order = 'F')
        else:
            x_after = np.zeros((u_len))
            _sampled_kronecker_products.compute_subset_of_matprod_entries(x_after, temp, M.T, row_inds_N, row_inds_M, u_len, cc_m)
    return x_after

