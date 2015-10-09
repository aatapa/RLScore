import numpy as np
from rlscore.utilities import sparse_kronecker_multiplication_tools_python

class KernelPairwisePredictor(object):
    
    def __init__(self, A, label_row_inds, label_col_inds, kernel = None):
        """Initializes the dual model
        @param A: dual coefficient matrix
        @type A: numpy matrix"""
        self.A = A
        self.label_row_inds, self.label_col_inds = label_row_inds, label_col_inds
        self.kernel = kernel
    
    
    def predictWithKernelMatrices(self, K1pred, K2pred, row_inds = None, col_inds = None):
        """Computes predictions for test examples.

        Parameters
        ----------
        K1pred: {array-like, sparse matrix}, shape = [n_samples1, n_basis_functions1]
            the first part of the test data matrix
        K2pred: {array-like, sparse matrix}, shape = [n_samples2, n_basis_functions2]
            the second part of the test data matrix
        
        Returns
        ----------
        P: array, shape = [n_samples1, n_samples2]
            predictions
        """
        if row_inds == None:
            P = sparse_kronecker_multiplication_tools_python.x_gets_A_kron_B_times_sparse_v(self.A, K1pred, K2pred.T, self.label_row_inds, self.label_col_inds)
            P = P.reshape((K1pred.shape[0], K2pred.shape[0]), order='F')
        else:
            P = sparse_kronecker_multiplication_tools_python.x_gets_C_times_M_kron_N_times_B_times_v(
                self.A,
                K2pred,
                K1pred,
                np.array(row_inds, dtype = np.int32),
                np.array(col_inds, dtype = np.int32),
                self.label_row_inds,
                self.label_col_inds)
        P = np.array(P)
        print "p", P.shape
        return P


class LinearPairwisePredictor(object):
    
    def __init__(self, W):
        """Initializes the linear model
        @param W: primal coefficient matrix
        @type W: numpy matrix"""
        self.W = W
    
    
    def predictWithDataMatrices(self, X1pred, X2pred):
        """Computes predictions for test examples.
        
        Parameters
        ----------
        X1pred: {array-like, sparse matrix}, shape = [n_samples1, n_features1]
            the first part of the test data matrix
        X2pred: {array-like, sparse matrix}, shape = [n_samples2, n_features2]
            the second part of the test data matrix
        
        Returns
        ----------
        P: array, shape = [n_samples1, n_samples2]
            predictions
        """
        P = np.dot(np.dot(X1pred, self.W), X2pred.T)
        return P
    
    
    def predictWithDataMatricesAlt(self, X1pred, X2pred, row_inds = None, col_inds = None):
        if row_inds == None:
            P = np.dot(np.dot(X1pred, self.W), X2pred.T).ravel(order = 'F')
        else:
            P = sparse_kronecker_multiplication_tools_python.x_gets_subset_of_A_kron_B_times_v(
                    self.W.reshape((self.W.shape[0] * self.W.shape[1], 1), order = 'F'),
                    X1pred, X2pred.T, np.array(row_inds, dtype = np.int32),
                    np.array(col_inds, dtype = np.int32))
            #P = X1pred * self.W * X2pred.T
        return P
