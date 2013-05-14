import cython
import numpy as np
cimport numpy as cnp

cdef extern from "c_swapped.h":
    double swapped_pairs(int len1, double* s, int len2, double* f)

def count_swapped(cnp.ndarray[cnp.double_t,ndim=1] A, cnp.ndarray[cnp.double_t,ndim=1] B):
    I = np.argsort(B)
    A = A[I]
    B = B[I]
    result = swapped_pairs(A.shape[0], <double*> A.data, B.shape[0], <double*> B.data)
    return result
