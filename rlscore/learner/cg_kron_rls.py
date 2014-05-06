

import pyximport; pyximport.install()

# from numpy import *
import numpy as np
import numpy.linalg as la

from scipy import sparse
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import bicgstab

from rlscore.learner.abstract_learner import AbstractIterativeLearner
from rlscore import data_sources
from rlscore import model
from rlscore.utilities import array_tools
from rlscore.utilities import decomposition

from rlscore.utilities import sparse_kronecker_multiplication_tools


def c_gets_axb(x, A, B, label_row_inds, label_col_inds):
    rc_a, cc_a = A.shape
    rc_b, cc_b = B.shape
    nzc_x = len(label_row_inds)
    len_c = rc_a * cc_b
    
    if rc_a * cc_a * cc_b + cc_b * nzc_x < rc_b * cc_a * cc_b + cc_a * nzc_x:
    #if False:
    #if True:
        #print 'foo'
        temp = np.mat(np.zeros((cc_a, cc_b)))
        sparse_kronecker_multiplication_tools.sparse_mat_from_left(temp, x, B, label_row_inds, label_col_inds, nzc_x, cc_b)
        temp = A * temp
        return temp.reshape((len_c,), order = 'F')
    else:
        #print 'bar'
        temp = np.mat(np.zeros((rc_a, rc_b)))
        sparse_kronecker_multiplication_tools.sparse_mat_from_right(temp, x, A, label_row_inds, label_col_inds, nzc_x, rc_a)
        temp = temp * B
        return temp.reshape((len_c,), order = 'F')


def u_gets_axb(xx, A, B, label_row_inds, label_col_inds):
    rc_a, cc_a = A.shape
    rc_b, cc_b = B.shape
    nzc_u = len(label_row_inds)
    len_c = rc_a * cc_b
    
    x_after = np.zeros(nzc_u)
    temp = xx.reshape((cc_a, rc_b), order = 'F')
    
    if rc_a * cc_a * cc_b + cc_b * nzc_u < rc_b * cc_a * cc_b + cc_a * nzc_u:
    #if False:
    #if True:
        temp = A * temp
        #temp = mat(zeros((cc_a, cc_b)))
        #sparse_kronecker_multiplication_tools.sparse_mat_from_left(temp, x, B, label_row_inds, label_col_inds, nzc_x, cc_b)
        sparse_kronecker_multiplication_tools.compute_subset_of_matprod_entries(x_after, temp, B, label_row_inds, label_col_inds, nzc_u, rc_b)
        return x_after
    else:
        temp = temp * B
        sparse_kronecker_multiplication_tools.compute_subset_of_matprod_entries(x_after, A, temp, label_row_inds, label_col_inds, nzc_u, cc_a)
        return x_after


class CGKronRLS(AbstractIterativeLearner):
    
    '''def __init__(self, train_labels, label_row_inds, label_col_inds, regparam=1.0):
        self.Y = array_tools.as_labelmatrix(train_labels)
        self.regparam = regparam
        self.label_row_inds = label_row_inds
        self.label_col_inds = label_col_inds
        self.results = {}'''
    
    
    def loadResources(self):
        Y = self.resource_pool[data_sources.TRAIN_LABELS]
        self.label_row_inds = np.array(self.resource_pool["label_row_inds"], dtype = np.int32)
        self.label_col_inds = np.array(self.resource_pool["label_col_inds"], dtype = np.int32)
        Y = array_tools.as_labelmatrix(Y)
        self.Y = Y
        self.trained = False
        if self.resource_pool.has_key(data_sources.CALLBACK_FUNCTION):
            self.callbackfun = self.resource_pool[data_sources.CALLBACK_FUNCTION]
        else:
            self.callbackfun = None
    
    
    def train(self):
        regparam = self.resource_pool['regparam']
        if self.resource_pool.has_key('kmatrix1'):
            self.solve_kernel(regparam)
        else:
            self.solve_linear(regparam)
    
    
    def solve_kernel(self, regparam):
        self.regparam = regparam
        K1 = np.mat(self.resource_pool['kmatrix1'])
        K2 = np.mat(self.resource_pool['kmatrix2'])
        lsize = len(self.label_row_inds) #n
        
        if 'maxiter' in self.resource_pool: maxiter = int(self.resource_pool['maxiter'])
        else: maxiter = None
        
        label_row_inds = self.label_row_inds
        label_col_inds = self.label_col_inds
        
        temp = np.zeros((K1.shape[1], K2.shape[0]))
        v_after = np.zeros((len(self.label_row_inds),))
        #Y = self.Y
        #self.itercount = 0
        def mv(v):
            assert v.shape[0] == len(self.label_row_inds)
            temp = np.zeros((K1.shape[1], K2.shape[0]))
            sparse_kronecker_multiplication_tools.sparse_mat_from_left(temp, v, K2, label_row_inds, label_col_inds, lsize, K2.shape[0])
            v_after = np.zeros(v.shape[0])
            #print K1.shape, temp.shape
            sparse_kronecker_multiplication_tools.compute_subset_of_matprod_entries(v_after, K1, temp, label_row_inds, label_col_inds, lsize, K1.shape[0])
            return v_after + regparam * v
        
        def mvr(v):
            foofoo
            raise Exception('You should not be here!')
            return None
        
        G = LinearOperator((len(self.label_row_inds), len(self.label_row_inds)), matvec = mv, rmatvec = mvr, dtype = np.float64)
        self.A = bicgstab(G, self.Y, maxiter = maxiter)[0]
        self.model = KernelPairwiseModel(self.A, self.label_row_inds, self.label_col_inds)
    
    
    def solve_linear(self, regparam):
        self.regparam = regparam
        X1 = np.mat(self.resource_pool['xmatrix1'])
        X2 = np.mat(self.resource_pool['xmatrix2'])
        self.X1, self.X2 = X1, X2
        
        if 'maxiter' in self.resource_pool: maxiter = int(self.resource_pool['maxiter'])
        else: maxiter = None
        
        x1tsize, x1fsize = X1.shape #m, d
        x2tsize, x2fsize = X2.shape #q, r
        lsize = len(self.label_row_inds) #n
        
        kronfcount = x1fsize * x2fsize
        
        label_row_inds = np.array(self.label_row_inds, dtype = np.int32)
        label_col_inds = np.array(self.label_col_inds, dtype = np.int32)
        
        def mv(v):
            v_after = u_gets_axb(v, X1, X2.T, label_row_inds, label_col_inds)
            v_after = c_gets_axb(v_after, X1.T, X2, label_row_inds, label_col_inds) + regparam * v
            return v_after
        
        def mvr(v):
            raise Exception('You should not be here!')
            return None
        
        def cgcb(v):
            self.W = np.mat(v).T.reshape((x1fsize, x2fsize), order = 'F')
            self.callback()
            
        G = LinearOperator((kronfcount, kronfcount), matvec = mv, rmatvec = mvr, dtype = np.float64)
        
        v_init = np.array(self.Y).reshape(self.Y.shape[0])
        v_init = c_gets_axb(v_init, X1.T, X2, label_row_inds, label_col_inds)
        v_init = np.array(v_init).reshape(kronfcount)
        if self.resource_pool.has_key('warm_start'):
            x0 = np.array(self.resource_pool['warm_start']).reshape(kronfcount, order = 'F')
        else:
            x0 = None
        self.W = np.mat(bicgstab(G, v_init, x0 = x0, maxiter = maxiter, callback = cgcb)[0]).T.reshape((x1fsize, x2fsize), order='F')
        self.model = LinearPairwiseModel(self.W, X1.shape[1], X2.shape[1])
        self.finished()
    
    
    
    def getModel(self):
        if not hasattr(self, "model"):
            self.model = LinearPairwiseModel(self.W, self.X1.shape[1], self.X2.shape[1])
        return self.model

    
class KernelPairwiseModel(object):
    
    def __init__(self, A, label_row_inds, label_col_inds, kernel = None):
        """Initializes the dual model
        @param A: dual coefficient matrix
        @type A: numpy matrix"""
        self.A = A
        self.label_row_inds, self.label_col_inds = label_row_inds, label_col_inds
        self.kernel = kernel
    
    
    def predictWithKernelMatrices(self, K1pred, K2pred):
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
        P = c_gets_axb(self.A, K1pred, K2pred.T, self.label_row_inds, self.label_col_inds)
        P = np.mat(P).reshape((K1pred.shape[0], K2pred.shape[0]), order='F')
        return P


class LinearPairwiseModel(object):
    
    def __init__(self, W, dim1, dim2):
        """Initializes the linear model
        @param W: primal coefficient matrix
        @type W: numpy matrix"""
        self.W = W
        self.dim1, self.dim2 = dim1, dim2
    
    
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
        P = X1pred * self.W * X2pred.T
        return P
    
    
    def predictWithDataMatricesAlt(self, X1pred, X2pred, row_inds = None, col_inds = None):
        if row_inds == None:
            P = X1pred * self.W * X2pred.T
            P = P.reshape(X1pred.shape[0] * X2pred.shape[0], 1, order = 'F')
        else:
            P = u_gets_axb(self.W.reshape((self.W.shape[0] * self.W.shape[1], 1), order = 'F'), X1pred, X2pred.T, array(row_inds, dtype=int32), array(col_inds, dtype=int32))
            #P = X1pred * self.W * X2pred.T
        return P


