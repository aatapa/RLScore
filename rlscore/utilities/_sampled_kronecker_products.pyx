import cython
#import numpy as np
#cimport numpy as np



@cython.boundscheck(False)
@cython.wraparound(False)

def sparse_mat_from_left(double [:, :] dst, double [:] sparse_matrix, double [:, :] dense_matrix, int [:] row_inds, int [:] col_inds, int entry_count, int dense_width):
    
    cdef int i, j
    cdef double entry
    
    for outerind in range(entry_count):
        i, j = row_inds[outerind], col_inds[outerind]
        entry = sparse_matrix[outerind]
        for innerind in range(dense_width):
            dst[i, innerind] += entry * dense_matrix[j, innerind]


def sparse_mat_from_right(double [:, :] dst, double [:, :] dense_matrix, double [:] sparse_matrix, int [:] row_inds, int [:] col_inds, int entry_count, int dense_height):
    
    cdef int i, j
    cdef double entry
    
    for outerind in range(entry_count):
        i, j = row_inds[outerind], col_inds[outerind]
        entry = sparse_matrix[outerind]
        for innerind in range(dense_height):
            dst[innerind, j] += dense_matrix[innerind, i] * entry


def compute_subset_of_matprod_entries(double [:] dst, double [:, :] matrix_left, double [:, :] matrix_right, int [:] row_inds, int [:] col_inds, int subsetlen, int veclen):
    
    cdef int i, j
    cdef double entry
    
    for outerind in range(subsetlen):
        i, j = row_inds[outerind], col_inds[outerind]
        entry = 0.
        for innerind in range(veclen):
            entry += matrix_left[i, innerind] * matrix_right[innerind, j]
        dst[outerind] = entry


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



