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

    
    
cpdef int find_optimal_feature(double [:, :] Y,
                         double [:, :] X,
                         double [:, :] GXT,
                         double [:] diagG,
                         double [:, :] dualvec,
                         double [:] looperf,
                         int fsize,
                         int tsize,
                         int lsize,
                         short [:] selected,
                         double [:] tempvec1,
                         double [:] tempvec2,
                         double [:, :] tempvec3):
    
    cdef double inf, temp_d1, bestlooperf
    cdef int ci, bestcind, i, j
    
    inf = float('Inf')
    bestlooperf = inf
    
    for ci in range(fsize):
        if selected[ci] > 0: continue
        GXT_ci = GXT[:, ci]
        temp_d1 = 0.
        for i in range(tsize):
            temp_d1 += X[ci, i] * GXT[i, ci]
        temp_d1 = 1. / (1. + temp_d1)
        for i in range(tsize):
            tempvec1[i] = GXT[i, ci] * temp_d1
        
        for j in range(lsize):
            tempvec2[j] = 0.
            for i in range(tsize):
                tempvec2[j] += X[ci, i] * dualvec[i, j]
        for j in range(lsize):
            for i in range(tsize):
                tempvec3[i, j] = dualvec[i, j] - tempvec1[i] * tempvec2[j]
        
        for i in range(tsize):
            tempvec1[i] = 1. / (diagG[i] - tempvec1[i] * GXT[i, ci])
        
        for j in range(lsize):
            for i in range(tsize):
                tempvec3[i, j] = tempvec1[i] * tempvec3[i, j]
        
        temp_d1 = 0.
        for j in range(lsize):
            for i in range(tsize):
                temp_d1 += tempvec3[i, j] * tempvec3[i, j]
        temp_d1 = temp_d1 / (tsize * lsize)
        
        if temp_d1 < bestlooperf:
            bestcind = ci
            bestlooperf = temp_d1
        looperf[ci] = temp_d1
    selected[bestcind] = 1
    return bestcind



