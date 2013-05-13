import cython
import numpy as np
cimport numpy as np

cdef extern from "c_swapped.h":
    double swapped_pairs(int len1, double* s, int len2, double* f, int len3, int* o)

def count_swapped(np.ndarray[np.double_t,ndim=1] A, np.ndarray[np.double_t,ndim=1] B, np.ndarray[np.int32_t,ndim=1] C):
    result = swapped_pairs(A.shape[0], <double*> A.data, B.shape[0], <double*> B.data, C.shape[0], <int*> C.data )
    return result
