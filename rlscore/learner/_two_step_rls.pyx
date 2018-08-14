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

def out_of_sample_loo_symmetric(double [:, :] G,
                                 double [:, :] Y,
                                 double [:, :] GY,
                                 double [:, :] YG,
                                 double [:, :] GYG,
                                 double [:, :] results,
                                 int rowlen,
                                 int collen):
    cdef int i, j
    cdef double G_ii, G_ij, G_jj, GYG_ii, GYG_ij, GYG_ji, GYG_jj, invdetGhh
    cdef double invGhh_ii, invGhh_ij, invGhh_jj, GY_ii, GY_ij, GY_ji, GY_jj
    cdef double YG_ii, YG_ij, YG_ji, YG_jj
    cdef double invGhhGYhh_ii, invGhhGYhh_ij, invGhhGYhh_ji, invGhhGYhh_jj
    cdef double YGhhinvGhh_ii, YGhhinvGhh_ij, YGhhinvGhh_ji, YGhhinvGhh_jj
    cdef double invGhhGYGhhinvGhh_ii, invGhhGYGhhinvGhh_ij, invGhhGYGhhinvGhh_ji, invGhhGYGhhinvGhh_jj
    
    for i in range(rowlen):
        for j in range(collen):
            
            if i == j: continue
            
            G_ii = G[i, i]
            G_ij = G[i, j]
            G_jj = G[j, j]
            
            GYG_ii = GYG[i, i]
            GYG_ij = GYG[i, j]
            GYG_ji = GYG[j, i]
            GYG_jj = GYG[j, j]
            
            invdetGhh = 1. / (G_ii * G_jj - G_ij * G_ij)
            
            invGhh_ii = invdetGhh * G_jj
            invGhh_ij = - invdetGhh * G_ij
            invGhh_jj = invdetGhh * G_ii
            #GY_ii = GY[i, i]
            GY_ij = GY[i, j]
            #GY_ji = GY[j, i]
            GY_jj = GY[j, j]
            YG_ii = YG[i, i]
            YG_ij = YG[i, j]
            #YG_ji = YG[j, i]
            #YG_jj = YG[j, j]
            
            #invGhhGYhh_ii = invGhh_ii * GY_ii + invGhh_ij * GY_ji
            invGhhGYhh_ij = invGhh_ii * GY_ij + invGhh_ij * GY_jj
            #invGhhGYhh_ji = invGhh_ij * GY_ii + invGhh_jj * GY_ji
            #invGhhGYhh_jj = invGhh_ij * GY_ij + invGhh_jj * GY_jj
            #YGhhinvGhh_ii = YG_ii * invGhh_ii + YG_ij * invGhh_ij
            YGhhinvGhh_ij = YG_ii * invGhh_ij + YG_ij * invGhh_jj
            #YGhhinvGhh_ji = YG_ji * invGhh_ii + YG_jj * invGhh_ij
            #YGhhinvGhh_jj = YG_ji * invGhh_ij + YG_jj * invGhh_jj
            invGhhGYGhh_ii = invGhh_ii * GYG_ii + invGhh_ij * GYG_ji
            invGhhGYGhh_ij = invGhh_ii * GYG_ij + invGhh_ij * GYG_jj
            #invGhhGYGhh_ji = invGhh_ij * GYG_ii + invGhh_jj * GYG_ji
            #invGhhGYGhh_jj = invGhh_ij * GYG_ij + invGhh_jj * GYG_jj
            #invGhhGYGhhinvGhh_ii = invGhhGYGhh_ii * invGhh_ii + invGhhGYGhh_ij * invGhh_ij
            invGhhGYGhhinvGhh_ij = invGhhGYGhh_ii * invGhh_ij + invGhhGYGhh_ij * invGhh_jj
            #invGhhGYGhhinvGhh_ji = invGhhGYGhh_ji * invGhh_ii + invGhhGYGhh_jj * invGhh_ij
            #invGhhGYGhhinvGhh_jj = invGhhGYGhh_ji * invGhh_ij + invGhhGYGhh_jj * invGhh_jj
            
            results[i, j] = Y[i, j] - invGhhGYhh_ij - YGhhinvGhh_ij + invGhhGYGhhinvGhh_ij



    