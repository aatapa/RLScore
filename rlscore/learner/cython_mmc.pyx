

import cython
import numpy as np
cimport numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)

    
    
def findSteepestDirRotateClasses(double [:, :] Y,
                                 double [:, :] R,
                                 double [:, :] RY,
                                 double [:, :] Y_Schur_RY,
                                 double [:] classFitnessRowVec,
                                 double [:] mdiagRx2,
                                 int [:] classcounts,
                                 int [:] classvec,
                                 int size,
                                 int labelcount,
                                 int howmany,
                                 double [:, :] sqrtR,
                                 int rank_R):
    cdef int h, i, j
    bar_numpy = np.zeros((size))
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
                #dirsnegdiff_i = mdiagRx2[i] + 4 * Y[i, oldclazz] * RY[i, oldclazz] + mdiagRx2[i] + 4 * Y[i, newclazz] * RY[i, newclazz]
                #dirsnegdiff_i = mdiagRx2[i] + Y[i, oldclazz] * RY[i, oldclazz] + Y[i, newclazz] * RY[i, newclazz]
                dirsnegdiff_i = mdiagRx2[i] + Y_Schur_RY[i, oldclazz] + Y_Schur_RY[i, newclazz]
                if dirsnegdiff_i < steepness:
                    steepness = dirsnegdiff_i # * 4
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



