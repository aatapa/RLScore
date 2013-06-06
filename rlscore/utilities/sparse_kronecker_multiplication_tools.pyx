import cython
#import numpy as np
#cimport numpy as np




@cython.boundscheck(False)
@cython.wraparound(False)

def sparse_mat_from_left(double [:, :] dst, double [:] v, double [:, :] X2, int [:] label_row_inds, int [:] label_col_inds, int couplecount, int X2_width):
    
    cdef int i, j
    cdef double tempd
    
    for outerind in range(couplecount):
        i, j = label_row_inds[outerind], label_col_inds[outerind]
        tempd = v[outerind]
        for innerind in range(X2_width):
            dst[i, innerind] += tempd * X2[j, innerind]


def sparse_mat_from_right(double [:, :] dst, double [:] u, double [:, :] X1T, int [:] label_row_inds, int [:] label_col_inds, int couplecount, int X1T_height):
    
    cdef int i, j
    cdef double tempd
    
    for outerind in range(couplecount):
        i, j = label_row_inds[outerind], label_col_inds[outerind]
        tempd = u[outerind]
        for innerind in range(X1T_height):
            dst[innerind, j] = dst[innerind, j] + X1T[innerind, i] * tempd


def compute_subset_of_matprod_entries(double [:] dst, double [:, :] ML, double [:, :] MR, int [:] label_row_inds, int [:] label_col_inds, int subsetlen, int veclen):
    
    cdef int i, j
    cdef double tempd
    
    for outerind in range(subsetlen):
        i, j = label_row_inds[outerind], label_col_inds[outerind]
        tempd = 0.
        for innerind in range(veclen):
            tempd += ML[i, innerind] * MR[innerind, j]
        dst[outerind] = tempd


def cpy_reorder(dst,src, rowcount, colcount):
    cdef double [:, :] c_dst = dst
    cdef double [:, :] c_src = src

    cdef int i, j, h, k
    cdef int rows = rowcount
    cdef int cols = colcount 
        
    for i in range(rows):
        for j in range(cols):
            for h in range(rows):
                for k in range(cols):
                    c_dst[i * cols + j, h * cols + k] = c_src[i * rows + h, j * cols + k]



