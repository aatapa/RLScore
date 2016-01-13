
# from numpy import *
import numpy as np

from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import minres

from rlscore.pairwise_predictor import LinearPairwisePredictor
from rlscore.pairwise_predictor import KernelPairwisePredictor
from rlscore.utilities import array_tools
from rlscore.utilities import sampled_kronecker_products
from rlscore.pairwise_predictor import PairwisePredictorInterface

CALLBACK_FUNCTION = 'callback'


class CGKronRLS(PairwisePredictorInterface):
    
    
    def __init__(self, **kwargs):
        self.resource_pool = kwargs
        Y = kwargs["Y"]
        self.input1_inds = np.array(kwargs["label_row_inds"], dtype = np.int32)
        self.input2_inds = np.array(kwargs["label_col_inds"], dtype = np.int32)
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
        
        if 'maxiter' in self.resource_pool: maxiter = int(self.resource_pool['maxiter'])
        else: maxiter = None
        
        Y = np.array(self.Y).ravel(order = 'F')
        self.bestloss = float("inf")
        def mv(v):
            return sampled_kronecker_products.sampled_vec_trick(v, K2, K1, self.input2_inds, self.input1_inds, self.input2_inds, self.input1_inds) + regparam * v
        
        def mvr(v):
            raise Exception('You should not be here!')
        
        def cgcb(v):
            if self.compute_risk:
                P =  sampled_kronecker_products.sampled_vec_trick(v, K2, K1, self.input2_inds, self.input1_inds, self.input2_inds, self.input1_inds)
                z = (Y - P)
                Ka = sampled_kronecker_products.sampled_vec_trick(v, K2, K1, self.input2_inds, self.input1_inds, self.input2_inds, self.input1_inds)
                loss = (np.dot(z,z)+regparam*np.dot(v,Ka))
                print "loss", 0.5*loss
                if loss < self.bestloss:
                    self.A = v.copy()
                    self.bestloss = loss
            else:
                self.A = v
            if not self.callbackfun == None:
                self.callbackfun.callback(self)

        
        G = LinearOperator((len(self.input1_inds), len(self.input1_inds)), matvec = mv, rmatvec = mvr, dtype = np.float64)
        minres(G, self.Y, maxiter = maxiter, callback = cgcb, tol=1e-20)[0]
        self.predictor = KernelPairwisePredictor(self.A, self.input1_inds, self.input2_inds)
    
    
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
        
        Y = np.array(self.Y).ravel(order = 'F')
        self.bestloss = float("inf")
        def mv(v):
            #v_after = sampled_kronecker_products.x_gets_subset_of_A_kron_B_times_v(v, X1, X2.T, label_row_inds, label_col_inds)
            v_after = sampled_kronecker_products.sampled_vec_trick(v, X2, X1, self.input2_inds, self.input1_inds)
            #v_after = sampled_kronecker_products.x_gets_A_kron_B_times_sparse_v(v_after, X1.T, X2, label_row_inds, label_col_inds) + regparam * v
            v_after = sampled_kronecker_products.sampled_vec_trick(v_after, X2.T, X1.T, None, None, self.input2_inds, self.input1_inds) + regparam * v
            return v_after
        
        def mvr(v):
            raise Exception('You should not be here!')
            return None
        
        def cgcb(v):
            #self.W = v.reshape((x1fsize, x2fsize), order = 'F')
            if self.compute_risk:
                #P = sampled_kronecker_products.x_gets_subset_of_A_kron_B_times_v(v, X1, X2.T, label_row_inds, label_col_inds)
                P = sampled_kronecker_products.sampled_vec_trick(v, X2, X1, self.input2_inds, self.input1_inds)
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
        #v_init = sampled_kronecker_products.x_gets_A_kron_B_times_sparse_v(v_init, X1.T, X2, label_row_inds, label_col_inds)
        v_init = sampled_kronecker_products.sampled_vec_trick(v_init, X2.T, X1.T, None, None, self.input2_inds, self.input1_inds)
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

    



