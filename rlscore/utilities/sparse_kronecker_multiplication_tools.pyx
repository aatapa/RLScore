import cython

@cython.boundscheck(False)
@cython.wraparound(False)

def sparse_mat_from_left(dst, v, X2, label_row_inds, label_col_inds, couplecount, X2_width):
    cdef double [:, :] c_dst = dst
    cdef double [:] c_v = v
    cdef double [:, :] c_X2 = X2
    cdef int [:] c_label_row_inds = label_row_inds
    cdef int [:] c_label_col_inds = label_col_inds
    
    cdef int c_couplecount = couplecount
    cdef int c_X2_width = X2_width
    
    cdef int i, j
    
    for outerind in range(c_couplecount):
        i, j = c_label_col_inds[outerind], c_label_row_inds[outerind]
        for innerind in range(c_X2_width):
            c_dst[i, innerind] = c_dst[i, innerind] + c_v[outerind] * c_X2[j, innerind]


def sparse_mat_from_right(dst, u, X1T, label_row_inds, label_col_inds, couplecount, X1T_height):
    cdef double [:, :] c_dst = dst
    cdef double [:] c_u = u
    cdef double [:, :] c_X1T = X1T
    cdef int [:] c_label_row_inds = label_row_inds
    cdef int [:] c_label_col_inds = label_col_inds
    
    cdef int c_couplecount = couplecount
    cdef int c_X1T_height = X1T_height
    
    cdef int i, j
    
    #for ind in range(lsize):
    #    i, j = self.label_col_inds[ind], self.label_row_inds[ind]
    #    temp[:, j] = temp[:, j] + X1.T[:, i] * u[ind]
    
    for outerind in range(c_couplecount):
        i, j = c_label_col_inds[outerind], c_label_row_inds[outerind]
        for innerind in range(c_X1T_height):
            c_dst[innerind, j] = c_dst[innerind, j] + X1T[innerind, i] * c_u[outerind]


def compute_subset_of_matprod_entries(dst, ML, MR, label_row_inds, label_col_inds, subsetlen, veclen):
    cdef double [:] c_dst = dst
    cdef double [:, :] c_ML = ML
    cdef double [:, :] c_MR = MR
    cdef int [:] c_label_row_inds = label_row_inds
    cdef int [:] c_label_col_inds = label_col_inds
    
    cdef int c_subsetlen = subsetlen
    cdef int c_veclen = veclen
    
    cdef int i, j
    
    #for ind in range(lsize):
    #    i, j = self.label_col_inds[ind], self.label_row_inds[ind]
    #    v_after[ind] = X1[i] * temp[:, j]
    
    for outerind in range(c_subsetlen):
        i, j = c_label_col_inds[outerind], c_label_row_inds[outerind]
        for innerind in range(c_veclen):
            c_dst[outerind] = c_dst[outerind] + c_ML[i, innerind] * c_MR[innerind, j]


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



