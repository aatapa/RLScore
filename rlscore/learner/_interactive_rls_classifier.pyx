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


def claim_all_points(double [:, :] Y,
                     int [:] classcounts,
                     int [:] classvec,
                     int size,
                     double [:, :] DVTY,
                     double [:, :] sqrtRx2,
                     int rank_R,
                     int newclazz):
    
    cdef int oldclazz, i, j
    
    for i in range(size):
        oldclazz = classvec[i]
        if oldclazz == newclazz: continue
        
        Y[i, oldclazz] = -1.
        Y[i, newclazz] = 1.
        classvec[i] = newclazz
        classcounts[oldclazz] -= 1
        classcounts[newclazz] += 1
        
        for j in range(rank_R):
            DVTY[j, newclazz] += sqrtRx2[i, j]
            DVTY[j, oldclazz] -= sqrtRx2[i, j]
    i = 1


def cyclic_desccent(double [:, :] Y,
                     int [:] classcounts,
                     int [:] classvec,
                     double [:] fitvec,
                     int size,
                     int globalsize,
                     double [:, :] DVTY,
                     double [:, :] sqrtRx2,
                     int rank_R,
                     int labelcount,
                     int maxbalancechange,
                     int [:] classcount_delta):
    
    cdef int oldclazz, candclazz, newclazz, i, j, changed, changecount
    cdef double temp_fitness_oc, temp_fitness_nc, temp_double
    
    for candclazz in range(labelcount):
        fitvec[candclazz] = globalsize
        for j in range(rank_R):
            fitvec[candclazz] -= DVTY[j, candclazz] * DVTY[j, candclazz]
    
    changed = 1
    changecount = 0
    while changed == 1:
        changed = 0
        for i in range(size):
            oldclazz = classvec[i]
            
            for candclazz in range(labelcount):
                if oldclazz == candclazz: continue
                temp_fitness_oc = globalsize
                temp_fitness_nc = globalsize
                for j in range(rank_R):
                    temp_double = DVTY[j, oldclazz] - sqrtRx2[i, j]
                    temp_fitness_oc -= temp_double * temp_double
                    temp_double = DVTY[j, candclazz] + sqrtRx2[i, j]
                    temp_fitness_nc -= temp_double * temp_double
                if temp_fitness_oc + temp_fitness_nc < fitvec[oldclazz] + fitvec[candclazz]:
                    Y[i, oldclazz] = -1.
                    Y[i, candclazz] = 1.
                    classvec[i] = candclazz
                    classcounts[oldclazz] -= 1
                    classcounts[candclazz] += 1
                    
                    for j in range(rank_R):
                        DVTY[j, candclazz] += sqrtRx2[i, j]
                        DVTY[j, oldclazz] -= sqrtRx2[i, j]
                    
                    fitvec[oldclazz] = temp_fitness_oc
                    fitvec[candclazz] = temp_fitness_nc
                    
                    changed = 1
                    newclazz = candclazz
            
            if changed == 1:
                changecount += 1
                classcount_delta[oldclazz] -= 1
                classcount_delta[newclazz] += 1
                if classcount_delta[oldclazz] == maxbalancechange \
                 or classcount_delta[oldclazz] == -maxbalancechange \
                 or classcount_delta[newclazz] == maxbalancechange \
                 or classcount_delta[newclazz] == -maxbalancechange:
                    changed = 0
                    break
    return changecount


def compute_gradient(double [:, :] Y,
                     double [:] gradient_vec,
                     int [:] classcounts,
                     int [:] classvec,
                     int size,
                     int globalsize,
                     double [:, :] DVTY,
                     double [:, :] sqrtRx2,
                     int rank_R,
                     int newclazz,
                     double [:] tempvec,
                     int tempveclen):
    
    cdef int oldclazz, i, j
    cdef double inf, foo, steepness
    
    inf = float('Inf')
    for i in range(size):
        oldclazz = classvec[i]
        if oldclazz == newclazz:
            gradient_vec[i] = inf
            continue
        steepness = 2 * globalsize
        for j in range(rank_R):
            foo = DVTY[j, newclazz] + sqrtRx2[i, j]
            steepness -= foo * foo
            foo = DVTY[j, oldclazz] - sqrtRx2[i, j]
            steepness -= foo * foo
        gradient_vec[i] = steepness
    return gradient_vec


def claim_n_points(double [:, :] Y,
                     double [:] gradient_vec,
                     int [:] classcounts,
                     int [:] classvec,
                     long [:] claiminds,
                     int nnn,
                     double [:, :] DVTY,
                     double [:, :] sqrtRx2,
                     int rank_R,
                     int newclazz,
                     int [:] clazzcountchanges):
    
    cdef int oldclazz, i, j, ind
    cdef double inf
    
    inf = float('Inf')
    
    for i in range(nnn):
        ind = claiminds[i]
        oldclazz = classvec[ind]
        
        Y[ind, oldclazz] = -1.
        Y[ind, newclazz] = 1.
        classvec[ind] = newclazz
        classcounts[oldclazz] -= 1
        classcounts[newclazz] += 1
        
        for j in range(rank_R):
            DVTY[j, newclazz] += sqrtRx2[ind, j]
            DVTY[j, oldclazz] -= sqrtRx2[ind, j]
        clazzcountchanges[oldclazz] -= 1



