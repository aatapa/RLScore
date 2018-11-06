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
def sampled_vec_trick(v, M, N, row_inds_M, row_inds_N, col_inds_M, col_inds_N, temp = None, x_after = None):
    
    assert len(v.shape) == 1
    assert len(col_inds_N) == v.shape[0]
    assert row_inds_N is not None
    assert col_inds_N is not None
    rc_m, cc_m = M.shape
    rc_n, cc_n = N.shape
    u_len = len(row_inds_N)
    v_len = len(col_inds_N)
    
    if x_after is None: x_after = np.zeros((u_len))
    else: x_after.fill(0)
    if rc_m * v_len + cc_n * u_len < rc_n * v_len + cc_m * u_len:
        if temp is None: temp = np.zeros((cc_n, rc_m), order = 'F')
        else: temp.fill(0)
        _sampled_kronecker_products.sparse_mat_from_left(temp, v, M.T, col_inds_N, col_inds_M, v_len)
        _sampled_kronecker_products.compute_subset_of_matprod_entries(x_after, N, temp, row_inds_N, row_inds_M, u_len)
    else:
        if temp is None: temp = np.zeros((rc_n, cc_m), order = 'C')
        else: temp.fill(0)
        _sampled_kronecker_products.sparse_mat_from_right(temp, N, v, col_inds_N, col_inds_M, v_len)
        _sampled_kronecker_products.compute_subset_of_matprod_entries(x_after, temp, M.T, row_inds_N, row_inds_M, u_len)
    return x_after

