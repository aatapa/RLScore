import numpy as np
import numpy.linalg as la
from numpy.linalg.linalg import LinAlgError

SMALLEST_EVAL = 0.0000000001

def decomposeDataMatrix(X):
    """Returns the reduced singular value decomposition of the data matrix X so that only the singular vectors corresponding to the nonzero singular values are returned.
    
    @param X: data matrix whose rows and columns correspond to the features and datumns, respectively.
    @type X: numpy matrix of floats
    @return: the nonzero singular values and the corresponding left and right singular vectors of X. The singular vectors are contained in a r*1-matrix, where r is the number of nonzero singular values. 
    @rtype: a tuple of three numpy matrices"""
    svecs, svals, U = la.svd(X.T, full_matrices=0)
    svals, evecs = np.mat(svals), np.mat(svecs)
    evals = np.multiply(svals, svals)
    
    maxnz = min(X.shape[0], X.shape[1])
    nz = 0
    for l in range(maxnz):
        if evals[0, l] > SMALLEST_EVAL:
            nz += 1
    rang = range(0, nz)
    evecs = evecs[:, rang]
    svals = svals[:, rang]
    U = U[rang]
    return svals, evecs, U


def decomposeKernelMatrix(K, trunc = None):
    """"Returns the reduced eigen decomposition of the kernel matrix K so that only the eigenvectors corresponding to the nonzero eigenvalues are returned.
    
    @param K: a positive semi-definite kernel matrix whose rows and columns are indexed by the datumns.
    @type K: numpy matrix of floats
    @param trunc: return only the 'trunc' largest eigenvalues and the corresponding eigenvectors
    @type trunc: int
    @return: the square roots of the nonzero eigenvalues and the corresponding eigenvectors of K. The square roots of the eigenvectors are contained in a r*1-matrix, where r is the number of nonzero eigenvalues. 
    @rtype: a tuple of two numpy matrices"""
    evals, evecs = la.eigh(K)
    evals, evecs = np.mat(evals), np.mat(evecs)
    nz = 0
    maxnz = K.shape[0]
    for l in range(maxnz):
        if evals[0, l] > SMALLEST_EVAL:
            nz += 1
    if trunc != None: nz = min(nz, trunc)
    rang = range(maxnz - nz, maxnz)
    evecs = evecs[:, rang]
    evals = evals[:, rang]
    svals = np.sqrt(evals)
    return svals, evecs


#def decomposeSubsetKM(K_r, basis_vectors):
def decomposeSubsetKM(K_r, K_rr):
    """decomposes r*m kernel matrix, where r is the number of basis vectors and m the
    number of training examples
    
    @param K_r: r*m kernel matrix, where only the lines corresponding to basis vectors are present
    @type K_r: numpy matrix
    @param basis_vectors: the indices of the basis vectors
    @type basis_vectors: list of integers
    @return svals, evecs, U, C_T_inv
    @rtype tuple of numpy matrices"""
    #K_rr = K_r[:, basis_vectors]
    #C = la.cholesky(K_rr)
    try:
        C = la.cholesky(K_rr)
    except LinAlgError:
        print "Warning: chosen basis vectors not linearly independent"
        print "Shifting the diagonal of kernel matrix"
        #__shiftKmatrix(K_r, basis_vectors)
        #K_rr = K_r[:, basis_vectors]
        #C = la.cholesky(K_rr)
        C = la.cholesky(K_rr+0.000000001 * np.eye(K_rr.shape[0]))
    C_T_inv = la.inv(C.T)
    H = np.dot(K_r.T, C_T_inv)
    svals, evecs, U = decomposeDataMatrix(H.T)
    return svals, evecs, U, C_T_inv

def __shiftKmatrix(K_r, basis_vectors, shift=0.000000001):
    """Diagonal shift for the basis vector kernel evaluations
    
    @param K_r: r*m kernel matrix, where only the lines corresponding to basis vectors are present
    @type K_r: numpy matrix
    @param basis_vectors: indices of the basis vectors
    @type basis_vectors: list of integers
    @param shift: magnitude of the shift (default 0.000000001)
    @type shift: float
    """
    #If the chosen subset is not linearly independent, we
    #enforce this with shifting the kernel matrix
    for i, j in enumerate(basis_vectors):
        K_r[i, j] += shift    

