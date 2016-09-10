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

import cython

@cython.boundscheck(False)
@cython.wraparound(False)



def leave_pair_out(int pairslen,
                      long [:] pairs_first,
                      long [:] pairs_second,
                      int outputlen,
                      double [:, :] Y,
                      #double [:, :] svecsbevalssvecsT,
                      double [:, :] svecs,
                      double [:] bevals,
                      int rank,
                      double [:] hatmatrixdiagonal,
                      double [:, :] svecsbevalssvecsTY,
                      double [:, :] results_first,
                      double [:, :] results_second):
    
    cdef int i, j, pair_ind, oind, rind
    cdef double RQY_i, RQY_j, a, b, d, det, ss_ii, ss_ij, ss_jj
    
    for pair_ind in range(pairslen):
        i = pairs_first[pair_ind]
        j = pairs_second[pair_ind]
        
        ss_ii = hatmatrixdiagonal[i]
        ss_ij = 0.
        ss_jj = hatmatrixdiagonal[j]
        for rind in range(rank):
            ss_ij += bevals[rind] * svecs[i, rind] * svecs[j, rind]
        #Invert a symmetric 2x2 matrix
        #a, b, d = 1. - svecsbevalssvecsT[i, i], - svecsbevalssvecsT[i, j], 1. - svecsbevalssvecsT[j, j]
        a, b, d = 1. - ss_ii, - ss_ij, 1. - ss_jj
        det = 1. / (a * d - b * b)
        ia, ib, id = det * d, - det * b, det * a
        
        for oind in range(outputlen):
            #RQY_i = svecsbevalssvecsTY[i, oind] - svecsbevalssvecsT[i, i] * Y[i, oind] - svecsbevalssvecsT[i, j] * Y[j, oind]
            #RQY_j = svecsbevalssvecsTY[j, oind] - svecsbevalssvecsT[j, i] * Y[i, oind] - svecsbevalssvecsT[j, j] * Y[j, oind]
            RQY_i = svecsbevalssvecsTY[i, oind] - ss_ii * Y[i, oind] - ss_ij * Y[j, oind]
            RQY_j = svecsbevalssvecsTY[j, oind] - ss_ij * Y[i, oind] - ss_jj * Y[j, oind]
            lpo_i = ia * RQY_i + ib * RQY_j
            lpo_j = ib * RQY_i + id * RQY_j
            results_first[pair_ind, oind] = lpo_i
            results_second[pair_ind, oind] = lpo_j

