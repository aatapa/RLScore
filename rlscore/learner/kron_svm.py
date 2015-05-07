
# from numpy import *
import numpy as np
import numpy.linalg as la

from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import cg

from rlscore.learner.abstract_learner import AbstractIterativeLearner
from rlscore import model
from rlscore.utilities import array_tools
from rlscore.utilities import decomposition
from rlscore.utilities import sparse_kronecker_multiplication_tools_python
from scipy.optimize import line_search
from scipy.optimize import check_grad

TRAIN_LABELS = 'train_labels'
CALLBACK_FUNCTION = 'callback'

class KronSVM(AbstractIterativeLearner):
        
    
    def __init__(self, **kwargs):
        self.resource_pool = kwargs
        Y = kwargs[TRAIN_LABELS]
        self.label_row_inds = np.array(kwargs["label_row_inds"], dtype = np.int32)
        self.label_col_inds = np.array(kwargs["label_col_inds"], dtype = np.int32)
        self.Y = Y
        self.trained = False
        if kwargs.has_key("regparam"):
            self.regparam = kwargs["regparam"]
        else:
            self.regparam = 0.
        if kwargs.has_key(CALLBACK_FUNCTION):
            self.callbackfun = kwargs[CALLBACK_FUNCTION]
        else:
            self.callbackfun = None
    
    
    def createLearner(cls, **kwargs):
        learner = cls(**kwargs)
        return learner
    createLearner = classmethod(createLearner)
    
    
    def train(self):
        self.solve_linear(self.regparam)
    
    
    def solve_linear(self, regparam):
        self.regparam = regparam
        X1 = self.resource_pool['xmatrix1']
        X2 = self.resource_pool['xmatrix2']
        self.X1, self.X2 = X1, X2
        
        if 'maxiter' in self.resource_pool: maxiter = int(self.resource_pool['maxiter'])
        else: maxiter = 1000

        if 'inneriter' in self.resource_pool: inneriter = int(self.resource_pool['inneriter'])
        else: inneriter = 50
        
        x1tsize, x1fsize = X1.shape #m, d
        x2tsize, x2fsize = X2.shape #q, r
        lsize = len(self.label_row_inds) #n
        
        kronfcount = x1fsize * x2fsize
        
        label_row_inds = np.array(self.label_row_inds, dtype = np.int32)
        label_col_inds = np.array(self.label_col_inds, dtype = np.int32)
        
        
        #def cgcb(v):
        #    self.W = v.reshape((x1fsize, x2fsize), order = 'F')
        #    self.callback()


        #OPERAATIOT:
        #Z = X1 kron X2 
        #Z * v -> vec-trick: 
        #Z[sv] * v 
        #v_after = sparse_kronecker_multiplication_tools_python.x_gets_subset_of_A_kron_B_times_v(v, X1, X2.T, label_row_inds, label_col_inds)
        #v_after = sparse_kronecker_multiplication_tools_python.x_gets_A_kron_B_times_sparse_v(v_after, X1.T, X2, label_row_inds, label_col_inds)
        #Z[sv].T * v
        #

        Y = self.Y
        rowind = label_row_inds
        colind = label_col_inds
        lamb = self.regparam
        rowind = np.array(rowind, dtype = np.int32)
        colind = np.array(colind, dtype = np.int32)
        fdim = X1.shape[1]*X2.shape[1]
        def func(v):
            #REPLACE
            #P = np.dot(X,v)
            P = sparse_kronecker_multiplication_tools_python.x_gets_subset_of_A_kron_B_times_v(v, X2, X1.T, colind, rowind)
            z = (1. - Y*P)
            z = np.where(z>0, z, 0)
            return np.dot(z,z)+lamb*np.dot(v,v)
        def gradient(v):
            #REPLACE
            #P = np.dot(X,v)
            #P = vecProd(X1, X2, v)
            P = sparse_kronecker_multiplication_tools_python.x_gets_subset_of_A_kron_B_times_v(v, X2, X1.T, colind, rowind)
            z = (1. - Y*P)
            z = np.where(z>0, z, 0)
            sv = np.nonzero(z)[0]
            #map to rows and cols
            rows = rowind[sv]
            cols = colind[sv]
            #A = -2*np.dot(X[sv].T, Y[sv])
            A = -2 * sparse_kronecker_multiplication_tools_python.x_gets_A_kron_B_times_sparse_v(Y[sv], X1.T, X2, rows, cols)
            A = A.reshape(X2.shape[1], X1.shape[1]).T.ravel()
            #B = 2 * np.dot(X[sv].T, np.dot(X[sv],v))
            v_after = sparse_kronecker_multiplication_tools_python.x_gets_subset_of_A_kron_B_times_v(v, X2, X1.T, cols, rows)
            v_after = 2 * sparse_kronecker_multiplication_tools_python.x_gets_A_kron_B_times_sparse_v(v_after, X1.T, X2, rows, cols)
            B = v_after.reshape(X2.shape[1], X1.shape[1]).T.ravel()
            #print "FOOOBAAR"
            return A + B + lamb*v
            #return -2*np.dot(X[sv].T,Y[sv]) + 2 * np.dot(X[sv].T, np.dot(X[sv],v)) + lamb*v
        def hessian(v, p):
            #P = np.dot(X,v)
            #P = vecProd(X1, X2, v)
            P = sparse_kronecker_multiplication_tools_python.x_gets_subset_of_A_kron_B_times_v(v, X2, X1.T, colind, rowind)
            z = (1. - Y*P)
            z = np.where(z>0, z, 0)
            sv = np.nonzero(z)[0]
            #map to rows and cols
            rows = rowind[sv]
            cols = colind[sv]
            p_after = sparse_kronecker_multiplication_tools_python.x_gets_subset_of_A_kron_B_times_v(p, X2, X1.T, cols, rows)
            p_after = sparse_kronecker_multiplication_tools_python.x_gets_A_kron_B_times_sparse_v(p_after, X1.T, X2, rows, cols)
            p_after = p_after.reshape(X2.shape[1], X1.shape[1]).T.ravel()
            return 2 * p_after + lamb*p    
            #return 2 * np.dot(X[sv].T, np.dot(X[sv],p)) + lamb*p
        w = np.zeros(fdim)
        #np.random.seed(1)
        #w = np.random.random(fdim)
        def mv(v):
            return hessian(w, v)
        for i in range(maxiter):
            g = gradient(w)
            G = LinearOperator((fdim, fdim), matvec=mv, dtype=np.float64)
            w_new = cg(G, g, maxiter=inneriter)[0]
            #r = G*w_new - g
            #e_rel = np.linalg.norm(r)/np.linalg.norm(g)
            #print e_rel, alpha
            #print "function value", func(w)
            w = w - w_new
            self.W = w.reshape((x1fsize, x2fsize), order='C')
            self.callback()
            #print i
        self.model = LinearPairwiseModel(self.W, X1.shape[1], X2.shape[1])
        self.finished()
    
    
    def getModel(self):
        #if not hasattr(self, "model"):
        self.model = LinearPairwiseModel(self.W, self.X1.shape[1], self.X2.shape[1])
        return self.model




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
        P = np.dot(np.dot(X1pred, self.W), X2pred.T)
        return P
    
    
    def predictWithDataMatricesAlt(self, X1pred, X2pred, row_inds = None, col_inds = None):
        if row_inds == None:
            P = np.dot(np.dot(X1pred, self.W), X2pred.T)
            P = P.reshape(X1pred.shape[0] * X2pred.shape[0], 1, order = 'F')
        else:
            P = sparse_kronecker_multiplication_tools_python.x_gets_subset_of_A_kron_B_times_v(self.W.reshape((self.W.shape[0] * self.W.shape[1], 1), order = 'F'), X1pred, X2pred.T, np.array(row_inds, dtype=np.int32), np.array(col_inds, dtype=np.int32))
            #P = X1pred * self.W * X2pred.T
        return P


