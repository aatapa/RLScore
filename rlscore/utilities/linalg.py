import numpy as np
import numpy.linalg as la
from numpy.linalg.linalg import LinAlgError

SMALLEST_EVAL = 0.0000000001

def svd_economy_sized(X):
    """Returns the reduced singular value decomposition of the data matrix X so that only the singular vectors corresponding to the nonzero singular values are returned.
    
    @param X: data matrix whose rows and columns correspond to the data and features, respectively.
    @type X: numpy matrix of floats
    @return: the nonzero singular values and the corresponding left and right singular vectors of X. The singular vectors are contained in a r*1-matrix, where r is the number of nonzero singular values. 
    @rtype: a tuple of three numpy matrices"""
    evecs, svals, U = la.svd(X, full_matrices=0)
    evals = np.multiply(svals, svals)
    
    evecs = evecs[:, svals > 0]
    svals = svals[svals > 0]
    U = U[svals > 0]
    svals = np.mat(svals)
    return svals, evecs, U


def eig_psd(K):
    """"Returns the reduced eigen decomposition of the kernel matrix K so that only the eigenvectors corresponding to the nonzero eigenvalues are returned.
    
    @param K: a positive semi-definite kernel matrix whose rows and columns are indexed by the datumns.
    @type K: numpy matrix of floats
    @return: the square roots of the nonzero eigenvalues and the corresponding eigenvectors of K. The square roots of the eigenvectors are contained in a r*1-matrix, where r is the number of nonzero eigenvalues. 
    @rtype: a tuple of two numpy matrices"""
    try:
        evals, evecs = la.eigh(K)
    except LinAlgError, e:
        print 'Warning, caught a LinAlgError while eigen decomposing: ' + str(e)
        K = K + np.eye(K.shape[0]) * 0.0000000001
        evals, evecs = la.eigh(K)
    evecs = evecs[:, evals > 0.]
    evals = evals[evals > 0.]
    evals, evecs = np.mat(evals), np.mat(evecs)
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
    try:
        C = la.cholesky(K_rr)
    except LinAlgError:
        print "Warning: chosen basis vectors not linearly independent"
        print "Shifting the diagonal of kernel matrix"
        C = la.cholesky(K_rr+0.000000001 * np.eye(K_rr.shape[0]))
    C_T_inv = la.inv(C.T)
    H = np.dot(K_r.T, C_T_inv)
    svals, evecs, U = svd_economy_sized(H)
    return svals, evecs, U, C_T_inv
  

