

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
                #dirsnegdiff_i = minus_diagRx2[i] + 4 * Y[i, oldclazz] * RY[i, oldclazz] + minus_diagRx2[i] + 4 * Y[i, newclazz] * RY[i, newclazz]
                #dirsnegdiff_i = minus_diagRx2[i] + Y[i, oldclazz] * RY[i, oldclazz] + Y[i, newclazz] * RY[i, newclazz]
                dirsnegdiff_i = minus_diagRx2[i] + Y_Schur_RY[i, oldclazz] + Y_Schur_RY[i, newclazz]
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
            
            claim_a_point(Y, R, RY, Y_Schur_RY, minus_diagRx2, classcounts, classvec, size, 1, None, None, rank_R, newclazz, None, 0)#sqrtR, rank_R, newclazz)
    
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
            #dirsnegdiff_i = minus_diagRx2[i] + Y_Schur_RY[i, oldclazz] + Y_Schur_RY[i, newclazz]
            dirsnegdiff_i = tempvec[oldclazz]
            for j in range(rank_R):
                foo = DVTY[j, newclazz] + sqrtRx2[i, j]
                dirsnegdiff_i -= foo * foo
                foo = DVTY[j, oldclazz] - sqrtRx2[i, j]
                dirsnegdiff_i -= foo * foo
            #print minus_diagRx2[i] + Y_Schur_RY[i, oldclazz] + Y_Schur_RY[i, newclazz], dirsnegdiff_i / 4. #, minus_diagRx2[i] + Y[i, oldclazz] * RY[i, oldclazz] + Y[i, newclazz] * RY[i, newclazz]
            #dirsnegdiff_i = minus_diagRx2[i] + Y[i, oldclazz] * RY[i, oldclazz] + Y[i, newclazz] * RY[i, newclazz]
            if dirsnegdiff_i < steepness:
                steepness = dirsnegdiff_i # * 4
                steepestdir = i
    else:
        for i in range(size):
            oldclazz = classvec[i]
            if oldclazz == newclazz: continue
            #dirsnegdiff_i = minus_diagRx2[i] + 4 * Y[i, oldclazz] * RY[i, oldclazz] + minus_diagRx2[i] + 4 * Y[i, newclazz] * RY[i, newclazz]
            #dirsnegdiff_i = minus_diagRx2[i] + Y[i, oldclazz] * RY[i, oldclazz] + Y[i, newclazz] * RY[i, newclazz]
            dirsnegdiff_i = minus_diagRx2[i] + Y_Schur_RY[i, oldclazz] + Y_Schur_RY[i, newclazz]
            #if dirsnegdiff_i == steepness:
            #    print 'Found two equally steep directions'
            if dirsnegdiff_i < steepness:
                steepness = dirsnegdiff_i # * 4
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
#         for i in range(size):
#             #R_is_x2 = 2 * R[i, steepestdir]
#             
#             #Space efficient variation
#             R_is_x2 = 0
#             for j in range(rank_R):
#                 R_is_x2 += sqrtR[i, j] * sqrtR[steepestdir, j]
#             R_is_x2 *= 2
#             
#             #RY[i, oldclazz] -= R_is_x2
#             #RY[i, newclazz] += R_is_x2
#             
#             Y_Schur_RY[i, oldclazz] -= Y[i, oldclazz] * R_is_x2
#             Y_Schur_RY[i, newclazz] += Y[i, newclazz] * R_is_x2
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
                    #print temp_fitness_oc + temp_fitness_nc, fitvec[oldclazz] + fitvec[candclazz]
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
    #steepness = inf
    #for clazzind in range(tempveclen):
        #tempvec[clazzind] = 2 * globalsize
        #for j in range(rank_R):
        #    tempvec[clazzind] += DVTY[j, newclazz] * DVTY[j, newclazz] + DVTY[j, clazzind] * DVTY[j, clazzind]
    for i in range(size):
        oldclazz = classvec[i]
        if oldclazz == newclazz:
            gradient_vec[i] = inf
            continue
        steepness = 2 * globalsize#tempvec[oldclazz]
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



