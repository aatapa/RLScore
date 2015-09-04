
import cython

@cython.boundscheck(False)
@cython.wraparound(False)



def computePairwiseCV(int pairslen,
                      long [:] pairs_first,
                      long [:] pairs_second,
                      int outputlen,
                      double [:, :] Y,
                      double [:, :] svecsbevalssvecsT,
                      double [:, :] svecsbevalssvecsTY,
                      double [:, :] results_first,
                      double [:, :] results_second):
    
    cdef int i, j, pair_ind, oind
    cdef double RQY_i, RQY_j, a, b, d, det
    
    for pair_ind in range(pairslen):
        i = pairs_first[pair_ind]
        j = pairs_second[pair_ind]
        
        #Invert a symmetric 2x2 matrix
        a, b, d = 1. - svecsbevalssvecsT[i, i], - svecsbevalssvecsT[i, j], 1. - svecsbevalssvecsT[j, j]
        det = 1. / (a * d - b * b)
        ia, ib, id = det * d, - det * b, det * a
        
        for oind in range(outputlen):
            RQY_i = svecsbevalssvecsTY[i, oind] - svecsbevalssvecsT[i, i] * Y[i, oind] - svecsbevalssvecsT[i, j] * Y[j, oind]
            RQY_j = svecsbevalssvecsTY[j, oind] - svecsbevalssvecsT[j, i] * Y[i, oind] - svecsbevalssvecsT[j, j] * Y[j, oind]
            lpo_i = ia * RQY_i + ib * RQY_j
            lpo_j = ib * RQY_i + id * RQY_j
            results_first[pair_ind, oind] = lpo_i
            results_second[pair_ind, oind] = lpo_j

