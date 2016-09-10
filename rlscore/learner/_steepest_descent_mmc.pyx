#
# The MIT License (MIT)
#
# This file is part of RLScore 
#
# Copyright (c) 2013 - 2016 Tapio Pahikkala, Antti Airola
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

def findSteepestDirRotateClasses_(double [:, :] Y,
                                 double [:, :] R,
                                 double [:, :] RY,
                                 double [:, :] Y_Schur_RY,
                                 double [:] classFitnessRowVec,
                                 double [:] minus_diagRx2,
                                 int [:] classcounts,
                                 int [:] classvec,
                                 int size,
                                 int labelcount,
                                 int howmany,
                                 double [:, :] sqrtR,
                                 int rank_R):
    cdef int h, i, j
    cdef double [:] bar_mv, dirscc
    cdef double [:, :] dirsnegdiff
    cdef int takenum, oldclazz, steepestdir
    cdef double YTRY_oldclazz, YTRY_newclazz, dirsnegdiff_i, R_is_x2, inf
     
    inf = float('Inf')
     
    for newclazz in range(labelcount):
         
        #!!!!!!!!!!!!!!!
        takenum = (size / labelcount) - classcounts[newclazz] + howmany
         
        for h in range(takenum):
             
            steepness = inf
            for i in range(size):
                oldclazz = classvec[i]
                if oldclazz == newclazz: continue
                dirsnegdiff_i = minus_diagRx2[i] + Y_Schur_RY[i, oldclazz] + Y_Schur_RY[i, newclazz]
                if dirsnegdiff_i < steepness:
                    steepness = dirsnegdiff_i
                    steepestdir = i
            oldclazz = classvec[steepestdir]
             
            Y[steepestdir, oldclazz] = -1.
            Y[steepestdir, newclazz] = 1.
            classvec[steepestdir] = newclazz
            classcounts[oldclazz] -= 1
            classcounts[newclazz] += 1
             
            Y_Schur_RY[steepestdir, oldclazz] *= -1
            Y_Schur_RY[steepestdir, newclazz] *= -1
            for i in range(size):
                R_is_x2 = 2 * R[i, steepestdir]
                 
                #Space efficient variation
                #R_is_x2 = 0
                #for j in range(rank_R):
                #    R_is_x2 += sqrtR[i, j] * sqrtR[steepestdir, j]
                #R_is_x2 *= 2
                 
                #RY[i, oldclazz] -= R_is_x2
                #RY[i, newclazz] += R_is_x2
                 
                Y_Schur_RY[i, oldclazz] -= Y[i, oldclazz] * R_is_x2
                Y_Schur_RY[i, newclazz] += Y[i, newclazz] * R_is_x2
     
    for newclazz in range(labelcount):
        YTRY_newclazz = 0
        for i in range(size):
            YTRY_newclazz += Y_Schur_RY[i, newclazz]
        classFitnessRowVec[newclazz] = YTRY_newclazz
         
    return False


def findSteepestDirRotateClasses(double [:, :] Y,
                                 double [:, :] R,
                                 double [:, :] RY,
                                 double [:, :] Y_Schur_RY,
                                 double [:] classFitnessRowVec,
                                 double [:] minus_diagRx2,
                                 int [:] classcounts,
                                 int [:] classvec,
                                 int size,
                                 int labelcount,
                                 int howmany,
                                 double [:, :] sqrtR,
                                 int rank_R):
    cdef int h, takenum
    
    for newclazz in range(labelcount):
        
        #!!!!!!!!!!!!!!!
        takenum = (size / labelcount) - classcounts[newclazz] + howmany
        
        for h in range(takenum):
            
            claim_a_point(Y, R, RY, Y_Schur_RY, minus_diagRx2, classcounts, classvec, size, 1, None, None, rank_R, newclazz, None, 0)
    
    #Book keeping stuff
    for newclazz in range(labelcount):
        YTRY_newclazz = 0
        for i in range(size):
            YTRY_newclazz += Y_Schur_RY[i, newclazz]
        classFitnessRowVec[newclazz] = YTRY_newclazz
        
    return False

def claim_a_point(double [:, :] Y,
                     double [:, :] R,
                     double [:, :] RY,
                     double [:, :] Y_Schur_RY,
                     double [:] minus_diagRx2,
                     int [:] classcounts,
                     int [:] classvec,
                     int size,
                     int use_full_caches,
                     double [:, :] DVTY,
                     double [:, :] sqrtRx2,
                     int rank_R,
                     int newclazz,
                     double [:] tempvec,
                     int tempveclen):  #This function should be explicitly inlined to gain faster running speed.
    
    cdef int oldclazz, steepestdir, i, j
    cdef double YTRY_oldclazz, YTRY_newclazz, dirsnegdiff_base, dirsnegdiff_i, R_is_x2, inf, foo
    
    inf = float('Inf')
    steepness = inf
    steepestdir = 0
    if use_full_caches == 0:
        dirsnegdiff_base = 0.
        for clazzind in range(tempveclen):
            for j in range(rank_R):
                tempvec[clazzind] += DVTY[j, newclazz] * DVTY[j, newclazz] + DVTY[j, clazzind] * DVTY[j, clazzind]
        for i in range(size):
            oldclazz = classvec[i]
            if oldclazz == newclazz: continue
            dirsnegdiff_i = tempvec[oldclazz]
            for j in range(rank_R):
                foo = DVTY[j, newclazz] + sqrtRx2[i, j]
                dirsnegdiff_i -= foo * foo
                foo = DVTY[j, oldclazz] - sqrtRx2[i, j]
                dirsnegdiff_i -= foo * foo
            if dirsnegdiff_i < steepness:
                steepness = dirsnegdiff_i
                steepestdir = i
    else:
        for i in range(size):
            oldclazz = classvec[i]
            if oldclazz == newclazz: continue
            dirsnegdiff_i = minus_diagRx2[i] + Y_Schur_RY[i, oldclazz] + Y_Schur_RY[i, newclazz]
            if dirsnegdiff_i < steepness:
                steepness = dirsnegdiff_i
                steepestdir = i
    oldclazz = classvec[steepestdir]
    
    Y[steepestdir, oldclazz] = -1.
    Y[steepestdir, newclazz] = 1.
    classvec[steepestdir] = newclazz
    classcounts[oldclazz] -= 1
    classcounts[newclazz] += 1
    
    if use_full_caches == 0:
        for j in range(rank_R):
            DVTY[j, newclazz] += sqrtRx2[steepestdir, j]
            DVTY[j, oldclazz] -= sqrtRx2[steepestdir, j]
    else:
        Y_Schur_RY[steepestdir, oldclazz] *= -1
        Y_Schur_RY[steepestdir, newclazz] *= -1
        for i in range(size):
            R_is_x2 = 2 * R[i, steepestdir]
            
            #Space efficient variation
            #R_is_x2 = 0
            #for j in range(rank_R):
            #    R_is_x2 += sqrtR[i, j] * sqrtR[steepestdir, j]
            #R_is_x2 *= 2
            
            #RY[i, oldclazz] -= R_is_x2
            #RY[i, newclazz] += R_is_x2
            
            Y_Schur_RY[i, oldclazz] -= Y[i, oldclazz] * R_is_x2
            Y_Schur_RY[i, newclazz] += Y[i, newclazz] * R_is_x2
    return steepestdir, oldclazz