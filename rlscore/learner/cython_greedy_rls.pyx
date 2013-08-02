

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
        #print np.dot(cv, GXT_ci), temp_d1
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
        #updA = dualvec - np.dot(tempvec1[:, None], np.dot(cv, dualvec)[None, :])
        
        for i in range(tsize):
            tempvec1[i] = 1. / (diagG[i] - tempvec1[i] * GXT[i, ci])
        #invupddiagG = 1. / (diagG - np.multiply(tempvec1, GXT_ci))
        
        for j in range(lsize):
            for i in range(tsize):
                tempvec3[i, j] = tempvec1[i] * tempvec3[i, j]
        #loodiff = np.multiply(invupddiagG[:, None], updA)
        
        temp_d1 = 0.
        for j in range(lsize):
            for i in range(tsize):
                temp_d1 += tempvec3[i, j] * tempvec3[i, j]
        temp_d1 = temp_d1 / (tsize * lsize)
        #looperf_i = np.mean(np.multiply(loodiff, loodiff))
        
        if temp_d1 < bestlooperf:
            bestcind = ci
            bestlooperf = temp_d1
        looperf[ci] = temp_d1
    selected[bestcind] = 1
    return bestcind



