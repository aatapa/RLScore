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

from rlscore.utilities import sampled_kronecker_products

class PairwiseKernelOperator(object):

    def __init__(self, K1, K2, row_inds_K1 = None, row_inds_K2 = None, col_inds_K1 = None, col_inds_K2 = None, weights = None):
        
        self.K1, self.K2 = K1, K2
        self.row_inds_K1, self.row_inds_K2 = row_inds_K1, row_inds_K2
        self.col_inds_K1, self.col_inds_K2 = col_inds_K1, col_inds_K2
        if weights is not None: self.weights = weights
    
    def mv(self, v):
        
        def inner_mv(v, K1i, K2i, col_inds_K1i, col_inds_K2i, row_inds_K1i = None, row_inds_K2i = None):
            if len(K1i.shape) == 1:
                K1i = K1i.reshape(1, K1i.shape[0])
            if len(K2i.shape) == 1:
                K2i = K2i.reshape(1, K2i.shape[0])
            if row_inds_K1i is not None:
                P = sampled_kronecker_products.sampled_vec_trick(
                    v,
                    K2i,
                    K1i,
                    row_inds_K2i,
                    row_inds_K1i,
                    col_inds_K2i,
                    col_inds_K1i)
            else:
                P = sampled_kronecker_products.sampled_vec_trick(
                    v,
                    K2i,
                    K1i,
                    None,
                    None,
                    col_inds_K2i,
                    col_inds_K1i)
                
                #P = P.reshape((K1i.shape[0], K2i.shape[0]), order = 'F')
            P = np.array(P)
            return P
        
        if isinstance(self.K1, (list, tuple)):
            P = None
            for i in range(len(self.K1)):
                K1i = self.K1[i]
                K2i = self.K2[i]
                col_inds_K1i = self.col_inds_K1[i]
                col_inds_K2i = self.col_inds_K2[i]
                if self.row_inds_K1 is not None:
                    row_inds_K1i = self.row_inds_K1[i]
                    row_inds_K2i = self.row_inds_K2[i]
                    Pi = inner_mv(v, K1i, K2i, col_inds_K1i, col_inds_K2i, row_inds_K1i, row_inds_K2i)
                else:
                    Pi = inner_mv(v, K1i, K2i, col_inds_K1i, col_inds_K2i, None, None)
                if P is None: P = self.weights[i] * Pi
                else: P = P + self.weights[i] * Pi
            return P
        else:
            return inner_mv(v, self.K1, self.K2, self.col_inds_K1, self.col_inds_K2, self.row_inds_K1, self.row_inds_K2)
        