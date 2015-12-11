
# from numpy import *
import numpy as np

from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import minres

from rlscore.pairwise_predictor import LinearPairwisePredictor
from rlscore.pairwise_predictor import KernelPairwisePredictor
from rlscore.utilities import array_tools
from rlscore.utilities import sparse_kronecker_multiplication_tools_python
from rlscore.pairwise_predictor import PairwisePredictorInterface

CALLBACK_FUNCTION = 'callback'

def primal_rls_objective(w, X1, X2, Y, rowind, colind, lamb):
    #primal form of the objective function for regularized least squares
    #w: current primal solution
    #X1: samples x features data matrix for domain 1
    #X2: samples x features data matrix for domain 2
    #rowind: row indices for training pairs
    #colind: column indices for training pairs
    #lamb: regularization parameter
    P = sparse_kronecker_multiplication_tools_python.x_gets_subset_of_A_kron_B_times_v(w, X2, X1.T, colind, rowind)
    z = (Y - P)
    return 0.5*(np.dot(z,z)+lamb*np.dot(w,w))

class CGKronRLS(PairwisePredictorInterface):
    
    '''def __init__(self, Y, label_row_inds, label_col_inds, regparam=1.0):
        self.Y = array_tools.as_labelmatrix(Y)
        self.regparam = regparam
        self.label_row_inds = label_row_inds
        self.label_col_inds = label_col_inds
        self.results = {}'''
    
    
    def __init__(self, **kwargs):
        self.resource_pool = kwargs
        Y = kwargs["Y"]
        self.label_row_inds = np.array(kwargs["label_row_inds"], dtype = np.int32)
        self.label_col_inds = np.array(kwargs["label_col_inds"], dtype = np.int32)
        Y = array_tools.as_labelmatrix(Y)
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
        if kwargs.has_key("compute_risk"):
            self.compute_risk = kwargs["compute_risk"]
        else:
            self.compute_risk = False
        self.train()
    
    
    def train(self):
        if self.resource_pool.has_key('kmatrix1'):
            self.solve_kernel(self.regparam)
        else:
            self.solve_linear(self.regparam)
    
    
    def solve_kernel(self, regparam):
        self.regparam = regparam
        K1 = self.resource_pool['kmatrix1']
        K2 = self.resource_pool['kmatrix2']
        lsize = len(self.label_row_inds) #n
        
        if 'maxiter' in self.resource_pool: maxiter = int(self.resource_pool['maxiter'])
        else: maxiter = None
        
        label_row_inds = self.label_row_inds
        label_col_inds = self.label_col_inds
        
        #Y = self.Y
        #self.itercount = 0
        Y = np.array(self.Y).ravel()
        self.bestloss = float("inf")
        def mv(v):
            assert v.shape[0] == len(self.label_row_inds)
            temp = np.zeros((K1.shape[1], K2.shape[0]))
            #sparse_kronecker_multiplication_tools_python.sparse_mat_from_left(temp, v, K2, label_row_inds, label_col_inds, lsize, K2.shape[0])
            sparse_kronecker_multiplication_tools_python.sparse_mat_from_right(temp, K1, v, label_row_inds, label_col_inds, lsize, K1.shape[0])
            v_after = np.zeros(v.shape[0])
            #print K1.shape, temp.shape
            #sparse_kronecker_multiplication_tools_python.compute_subset_of_matprod_entries(v_after, K1, temp, label_row_inds, label_col_inds, lsize, K1.shape[1])
            sparse_kronecker_multiplication_tools_python.compute_subset_of_matprod_entries(v_after, temp, K2, label_row_inds, label_col_inds, lsize, K2.shape[0])
            return v_after + regparam * v
        
        def mvr(v):
            #foofoo
            raise Exception('You should not be here!')
            return None
        
        def cgcb(v):
            if self.compute_risk:
                P =  sparse_kronecker_multiplication_tools_python.compute_R_times_M_kron_N_times_C_times_v(v, K2, K1, label_row_inds, label_col_inds, label_row_inds,label_col_inds)
                z = (Y - P)
                Ka = sparse_kronecker_multiplication_tools_python.compute_R_times_M_kron_N_times_C_times_v(v, K2, K1, label_row_inds, label_col_inds, label_row_inds, label_col_inds)
                loss = (np.dot(z,z)+regparam*np.dot(v,Ka))
                print "loss", 0.5*loss
                if loss < self.bestloss:
                    self.A = v.copy()
                    self.bestloss = loss
            else:
                self.A = v
            if not self.callbackfun == None:
                self.callbackfun.callback(self)

        
        G = LinearOperator((len(self.label_row_inds), len(self.label_row_inds)), matvec = mv, rmatvec = mvr, dtype = np.float64)
        minres(G, self.Y, maxiter = maxiter, callback = cgcb, tol=1e-20)[0]
        self.predictor = KernelPairwisePredictor(self.A, self.label_row_inds, self.label_col_inds)
    
    
    def solve_linear(self, regparam):
        self.regparam = regparam
        X1 = self.resource_pool['xmatrix1']
        X2 = self.resource_pool['xmatrix2']
        self.X1, self.X2 = X1, X2
        
        if 'maxiter' in self.resource_pool: maxiter = int(self.resource_pool['maxiter'])
        else: maxiter = None
        
        x1tsize, x1fsize = X1.shape #m, d
        x2tsize, x2fsize = X2.shape #q, r
        
        kronfcount = x1fsize * x2fsize
        
        label_row_inds = np.array(self.label_row_inds, dtype = np.int32)
        label_col_inds = np.array(self.label_col_inds, dtype = np.int32)
        #Y = np.array(self.Y).ravel(order='F')
        Y = np.array(self.Y).ravel()
        self.bestloss = float("inf")
        def mv(v):
            v_after = sparse_kronecker_multiplication_tools_python.x_gets_subset_of_A_kron_B_times_v(v, X1, X2.T, label_row_inds, label_col_inds)
            v_after = sparse_kronecker_multiplication_tools_python.x_gets_A_kron_B_times_sparse_v(v_after, X1.T, X2, label_row_inds, label_col_inds) + regparam * v
            return v_after
        
        def mvr(v):
            raise Exception('You should not be here!')
            return None
        
        def cgcb(v):
            #self.W = v.reshape((x1fsize, x2fsize), order = 'F')
            if self.compute_risk:
                P = sparse_kronecker_multiplication_tools_python.x_gets_subset_of_A_kron_B_times_v(v, X1, X2.T, label_row_inds, label_col_inds)
                z = (Y - P)
                loss = (np.dot(z,z)+regparam*np.dot(v,v))
                if loss < self.bestloss:
                    self.W = v.copy().reshape((x1fsize, x2fsize), order = 'F')
                    self.bestloss = loss
            else:
                self.W = v.reshape((x1fsize, x2fsize), order = 'F')
            if not self.callbackfun == None:
                self.callbackfun.callback(self)
            
        G = LinearOperator((kronfcount, kronfcount), matvec = mv, rmatvec = mvr, dtype = np.float64)
        
        v_init = np.array(self.Y).reshape(self.Y.shape[0])
        v_init = sparse_kronecker_multiplication_tools_python.x_gets_A_kron_B_times_sparse_v(v_init, X1.T, X2, label_row_inds, label_col_inds)
        v_init = np.array(v_init).reshape(kronfcount)
        if self.resource_pool.has_key('warm_start'):
            x0 = np.array(self.resource_pool['warm_start']).reshape(kronfcount, order = 'F')
        else:
            x0 = None
        #self.W = bicgstab(G, v_init, x0 = x0, maxiter = maxiter, callback = cgcb)[0].reshape((x1fsize, x2fsize), order='F')
        minres(G, v_init, x0 = x0, maxiter = maxiter, callback = cgcb, tol=1e-20)[0].reshape((x1fsize, x2fsize), order='F')
        self.predictor = LinearPairwisePredictor(self.W)
        if not self.callbackfun == None:
                self.callbackfun.finished(self)

    



