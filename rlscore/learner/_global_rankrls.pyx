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
from libc.math cimport sqrt

@cython.boundscheck(False)
@cython.wraparound(False)



def leave_pair_out(int pairslen,
                      int data_len,
                      long [:] pairs_first,
                      long [:] pairs_second,
                      int outputlen,
                      double [:, :] Y,
                      double [:, :] svecs,
                      double [:] modevals,
                      int rank,
                      double [:] Gdiag,
                      double [:] GC,
                      double [:] sm2Gdiag,
                      double CTGC,
                      double [:, :] GDY,
                      double [:, :] BTY,
                      double [:, :] sqrtsm2GDY,
                      double [:, :] BTGBBTY,
                      double [:] CTY,
                      double [:] CTGDY,
                      double [:, :] results_first,
                      double [:, :] results_second):
    
    cdef int i, j, pair_ind, rind, col
    cdef double RQY_i, RQY_j, a, b, d, det, sqrtsm2
    cdef double ss_ii, ss_ij, ss_jj, GCi, GCj
    cdef double BTY0, BTY1, BTY2, GiipGij, GijpGjj, GCipGCj
    cdef double BTGB00, BTGB01, BTGB02, BTGB12
    cdef double BTGLY0, BTGLY1, BTGLY2
    cdef double BTGB00m1, BTGB11m1, BTGB22m1
    cdef double CF00, CF01, CF02, CF11, CF12, CF22
    cdef double invdeter
    cdef double b0, b1, b2, t1, t2, F0, F1
    cdef double sm2
    
    sm2 = data_len - 2.
    sqrtsm2 = sqrt(sm2)
    
    for i in range(data_len):
        for rind in range(rank):
            Gdiag[i] += modevals[rind] * svecs[i, rind] * svecs[i, rind]
        sm2Gdiag[i] = sm2 * Gdiag[i] - 1.
        for col in range(outputlen):
            BTY[i, col] = sqrtsm2 * Y[i, col]
            sqrtsm2GDY[i, col] = sqrtsm2 * GDY[i, col]
            BTGBBTY[i, col] = sm2 * Gdiag[i] * sqrtsm2 * Y[i, col]
    
    for pair_ind in range(pairslen):
        i = pairs_first[pair_ind]
        j = pairs_second[pair_ind]
        
        Gii = Gdiag[i]
        Gij = 0#G[i, j]
        Gjj = Gdiag[j]
        
        for rind in range(rank):
            Gij += modevals[rind] * svecs[i, rind] * svecs[j, rind]
        
        GCi = GC[i]
        GCj = GC[j]
        
        GiipGij = Gii + Gij
        GijpGjj = Gij + Gjj
        GCipGCj = GCi + GCj
        
        BTGB00 = GiipGij + GijpGjj + CTGC - GCipGCj - GCipGCj
        BTGB01 = sqrtsm2 * (GCi - GiipGij)
        BTGB02 = sqrtsm2 * (GCj - GijpGjj)
        BTGB12 = sm2 * Gij
        
        BTGB00m1 = BTGB00 - 1.
        BTGB11m1 = sm2Gdiag[i]
        BTGB22m1 = sm2Gdiag[j]
        
        CF00 = BTGB11m1 * BTGB22m1 - BTGB12 * BTGB12
        CF01 = -BTGB01 * BTGB22m1 + BTGB12 * BTGB02
        CF02 = BTGB01 * BTGB12 - BTGB11m1 * BTGB02
        CF11 = BTGB00m1 * BTGB22m1 - BTGB02 * BTGB02
        CF12 = -BTGB00m1 * BTGB12 + BTGB01 * BTGB02
        CF22 = BTGB00m1 * BTGB11m1 - BTGB01 * BTGB01
        
        invdeter = 1. / (BTGB00m1 * CF00 + BTGB01 * CF01 + BTGB02 * CF02)
        
        for col in range(outputlen):
            
            BTY0 = CTY[col] - Y[i, col] - Y[j, col]
            BTY1 = BTY[i, col]
            BTY2 = BTY[j, col]
            
            BTGLY0 = CTGDY[col] - (GDY[i, col] + GDY[j, col] + BTGB00 * BTY0 + BTGB01 * BTY1 + BTGB02 * BTY2)
            BTGLY1 = sqrtsm2GDY[i, col] - (BTGB01 * BTY0 + BTGBBTY[i, col] + BTGB12 * BTY2)
            BTGLY2 = sqrtsm2GDY[j, col] - (BTGB02 * BTY0 + BTGB12 * BTY1 + BTGBBTY[j, col])
            
            b0 = invdeter * (CF00 * BTGLY0 + CF01 * BTGLY1 + CF02 * BTGLY2) + BTY0
            b1 = invdeter * (CF01 * BTGLY0 + CF11 * BTGLY1 + CF12 * BTGLY2) + BTY1
            b2 = invdeter * (CF02 * BTGLY0 + CF12 * BTGLY1 + CF22 * BTGLY2) + BTY2
            
            t1 = -b0 + sqrtsm2 * b1
            t2 = -b0 + sqrtsm2 * b2
            
            F0 = GDY[i, col] - (Gii * t1 + Gij * t2 + GCi * b0)
            F1 = GDY[j, col] - (Gij * t1 + Gjj * t2 + GCj * b0)
            
            results_first[pair_ind, col] = F0
            results_second[pair_ind, col] = F1