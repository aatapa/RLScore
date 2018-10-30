import cython
from cython.parallel import prange
#import numpy as np
#cimport numpy as np



@cython.boundscheck(False)
@cython.wraparound(False)
def sparse_mat_from_left(double [::1, :] dst, double [:] sparse_matrix, double [::1, :] dense_matrix, int [::1] row_inds, int [::1] col_inds, int entry_count, int dense_width):
    
    cdef Py_ssize_t innerind, outerind
    for innerind in prange(dst.shape[1], nogil=True):
    #for innerind in range(dst.shape[1]):
        for outerind in range(entry_count):
            dst[row_inds[outerind], innerind] += sparse_matrix[outerind] * dense_matrix[col_inds[outerind], innerind]


@cython.boundscheck(False)
@cython.wraparound(False)
def sparse_mat_from_left_old(double [:, ::1] dst, double [:] sparse_matrix, double [:, ::1] dense_matrix, int [:] row_inds, int [:] col_inds, int entry_count, int dense_width):
    
    cdef int i, j
    cdef double entry
    
    for outerind in range(entry_count):
        i, j = row_inds[outerind], col_inds[outerind]
        entry = sparse_matrix[outerind]
        for innerind in range(dense_width):
            dst[i, innerind] += entry * dense_matrix[j, innerind]

@cython.boundscheck(False)
@cython.wraparound(False)
def sparse_mat_from_right(double [:, ::1] dst, double [:, ::1] dense_matrix, double [:] sparse_matrix, int [::1] row_inds, int [::1] col_inds, int entry_count, int dense_height):
    
    cdef Py_ssize_t innerind, outerind
    #for innerind in range(dst.shape[0]):
    for innerind in prange(dst.shape[0], nogil=True):
        #for outerind in prange(entry_count, nogil=True):
        for outerind in range(entry_count):
            #i, j = row_inds[outerind], col_inds[outerind]
            #entry = sparse_matrix[outerind]
            dst[innerind, col_inds[outerind]] += dense_matrix[innerind, row_inds[outerind]] * sparse_matrix[outerind]


@cython.boundscheck(False)
@cython.wraparound(False)
def sparse_mat_from_right_old(double [::1, :] dst, double [::1, :] dense_matrix, double [:] sparse_matrix, int [:] row_inds, int [:] col_inds, int entry_count, int dense_height):
    
    cdef Py_ssize_t i, j
    cdef Py_ssize_t innerind, outerind
    cdef double entry
    
    #for outerind in prange(entry_count, nogil=True):
    for outerind in range(entry_count):
        i, j = row_inds[outerind], col_inds[outerind]
        entry = sparse_matrix[outerind]
        for innerind in range(dst.shape[0]):
        #for innerind in prange(dst.shape[0], nogil=True):
            dst[innerind, j] += dense_matrix[innerind, i] * entry



@cython.boundscheck(False)
@cython.wraparound(False)
def compute_subset_of_matprod_entries(double [::1] dst, double [:, ::1] matrix_left, double [::1, :] matrix_right, int [::1] row_inds, int [::1] col_inds, int subsetlen, int veclen):
    
    cdef Py_ssize_t i, j
    cdef Py_ssize_t innerind, outerind
    #cdef double entry
    
    #for outerind in range(subsetlen):
    for outerind in prange(subsetlen, nogil=True):
        #i, j = row_inds[outerind], col_inds[outerind]
        #entry = 0.
        for innerind in range(matrix_left.shape[1]):
            #entry += matrix_left[i, innerind] * matrix_right[innerind, j]
            #entry = entry + matrix_left[i, innerind] * matrix_right[innerind, j]
            dst[outerind] += matrix_left[row_inds[outerind], innerind] * matrix_right[innerind, col_inds[outerind]]
            #dst[outerind] += matrix_left[i, innerind] * matrix_right[innerind, j]
        #dst[outerind] = entry


@cython.boundscheck(False)
@cython.wraparound(False)
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



