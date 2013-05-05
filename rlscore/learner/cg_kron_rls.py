

import pyximport; pyximport.install()

from numpy import *
import numpy.linalg as la

from scipy import sparse
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import bicgstab

from rlscore.learner.abstract_learner import AbstractLearner
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
        #print 'foo'
        temp = mat(zeros((cc_a, cc_b)))
        sparse_kronecker_multiplication_tools.sparse_mat_from_left(temp, x, B, array(label_row_inds, dtype=int32), array(label_col_inds, dtype=int32), nzc_x, cc_b)
        temp = A * temp
        return temp.reshape((len_c,), order='F')
    else:
        #print 'bar'
        temp = mat(zeros((rc_a, rc_b)))
        sparse_kronecker_multiplication_tools.sparse_mat_from_right(temp, x, A, array(label_row_inds, dtype=int32), array(label_col_inds, dtype=int32), nzc_x, rc_a)
        temp = temp * B
        return temp.reshape((len_c,), order='F')

class CGKronRLS(AbstractLearner):
    
    '''def __init__(self, train_labels, label_row_inds, label_col_inds, regparam=1.0):
        self.Y = array_tools.as_labelmatrix(train_labels)
        self.regparam = regparam
        self.label_row_inds = label_row_inds
        self.label_col_inds = label_col_inds
        self.results = {}'''
    
    
    def loadResources(self):
        Y = self.resource_pool[data_sources.TRAIN_LABELS]
        #self.K1 = mat(self.resource_pool['kmatrix1'])
        #self.K2 = mat(self.resource_pool['kmatrix2'])
        self.label_row_inds = self.resource_pool["label_row_inds"]
        self.label_col_inds = self.resource_pool["label_col_inds"]
        Y = array_tools.as_labelmatrix(Y)
        #assert Y.shape == (self.K1.shape[0], self.K2.shape[0]), 'Y.shape!=(K1.shape[0],K2.shape[0]). Y.shape=='+str(Y.shape)+', K1.shape=='+str(self.K1.shape)+', K2.shape=='+str(self.K2.shape)
        self.Y = Y
        self.trained = False
    
    
    def train(self):
        regparam = self.resource_pool['regparam']
        if self.resource_pool.has_key('kmatrix1'):
            self.solve_kernel(regparam)
        else:
            self.solve_linear(regparam)
    
    
    def solve_kernel(self, regparam):
        self.regparam = regparam
        K1 = mat(self.resource_pool['kmatrix1'])
        K2 = mat(self.resource_pool['kmatrix2'])
        lsize = len(self.label_row_inds) #n
        
        #Y = self.Y
        
        def mv(v):
            assert v.shape[0] == len(self.label_row_inds)
            temp = mat(zeros((K1.shape[1], K2.shape[0])))
            sparse_kronecker_multiplication_tools.sparse_mat_from_left(temp, v, K2, array(self.label_row_inds, dtype=int32), array(self.label_col_inds, dtype=int32), lsize, K2.shape[0])
            #for ind in range(len(self.label_row_inds)):
            #    i, j = self.label_col_inds[ind], self.label_row_inds[ind]
            #    temp[i] = temp[i] + v[ind] * K2.T[j]
            v_after = zeros(v.shape[0])
            sparse_kronecker_multiplication_tools.compute_subset_of_matprod_entries(v_after, K1, temp, array(self.label_row_inds, dtype=int32), array(self.label_col_inds, dtype=int32), lsize, K1.shape[0])
            #for ind in range(len(self.label_row_inds)):
            #    i, j = self.label_col_inds[ind], self.label_row_inds[ind]
            #    v_after[ind] = K1[i] * temp[:, j] + regparam * v[ind]
            return v_after + regparam * v
        
        def mvr(v):
            foofoo
            raise Exception('You should not be here!')
            return None
        
        G = LinearOperator((len(self.label_row_inds), len(self.label_row_inds)), matvec=mv, rmatvec=mvr, dtype=float64)
        self.A = bicgstab(G, self.Y)[0]
        self.model = KernelPairwiseModel(self.A, self.label_row_inds, self.label_col_inds)
    
    
    def solve_linear(self, regparam):
        self.regparam = regparam
        X1 = mat(self.resource_pool['xmatrix1'])
        X2 = mat(self.resource_pool['xmatrix2'])
        
        x1tsize, x1fsize = X1.shape #m, d
        x2tsize, x2fsize = X2.shape #q, r
        lsize = len(self.label_row_inds) #n
        
        kronfcount = x1fsize * x2fsize
        
        if x1tsize * x1fsize * x2fsize + x2fsize * lsize < x2tsize * x1fsize * x2fsize + x1fsize * lsize:
            def u_gets_bxv(v):
                temp = v.reshape((x1fsize, x2fsize), order='F')
                temp = X1 * temp
                v_after = zeros(lsize)
                sparse_kronecker_multiplication_tools.compute_subset_of_matprod_entries(v_after, temp, X2.T, array(self.label_row_inds, dtype=int32), array(self.label_col_inds, dtype=int32), lsize, x2fsize)
                #for ind in range(lsize):
                #    i, j = self.label_col_inds[ind], self.label_row_inds[ind]
                #    v_after[ind] = temp[i] * X2.T[:, j]
                return v_after
            
            def v_gets_xbu(u):
                temp = mat(zeros((x1tsize, x2fsize)))
                sparse_kronecker_multiplication_tools.sparse_mat_from_left(temp, u, X2, array(self.label_row_inds, dtype=int32), array(self.label_col_inds, dtype=int32), lsize, x2fsize)
                #for ind in range(lsize):
                #    i, j = self.label_col_inds[ind], self.label_row_inds[ind]
                #    temp[i] = temp[i] + u[ind] * X2[j]
                temp = X1.T * temp
                return temp.reshape((kronfcount,),order='F')
        else:
            def u_gets_bxv(v):
                temp = v.reshape((x1fsize, x2fsize), order='F')
                temp = temp * X2.T
                v_after = zeros(lsize)
                sparse_kronecker_multiplication_tools.compute_subset_of_matprod_entries(v_after, X1, temp, array(self.label_row_inds, dtype=int32), array(self.label_col_inds, dtype=int32), lsize, x1fsize)
                #for ind in range(lsize):
                #    i, j = self.label_col_inds[ind], self.label_row_inds[ind]
                #    v_after[ind] = X1[i] * temp[:, j]
                return v_after
            
            def v_gets_xbu(u):
                temp = mat(zeros((x1fsize, x2tsize)))
                sparse_kronecker_multiplication_tools.sparse_mat_from_right(temp, u, X1.T, array(self.label_row_inds, dtype=int32), array(self.label_col_inds, dtype=int32), lsize, x1fsize)
                #for ind in range(lsize):
                #    i, j = self.label_col_inds[ind], self.label_row_inds[ind]
                #    temp[:, j] = temp[:, j] + X1.T[:, i] * u[ind]
                temp = temp * X2
                return temp.reshape((kronfcount,), order='F')
        
        def mv(v):
            '''temp = mat(zeros((x1tsize, x2fsize)))
            sparse_kronecker_multiplication_tools.kron_slice_multiply(temp, v, X2, array(self.label_row_inds, dtype=int32), array(self.label_col_inds, dtype=int32), lsize, x2fsize)
            #for ind in range(lsize):
            #    i, j = self.label_col_inds[ind], self.label_row_inds[ind]
            #    temp[i] = temp[i] + v[ind] * X2[j]
            temp = (X1.T * temp) * X2.T
            v_after = zeros(v.shape[0])
            for ind in range(lsize):
                i, j = self.label_col_inds[ind], self.label_row_inds[ind]
                v_after[ind] = X1[i] * temp[:, j] + regparam * v[ind]'''
            v_after = u_gets_bxv(v)
            #v_after = c_gets_axb(v, X1.T, X2, self.label_row_inds, self.label_col_inds)
            #v_after = v_gets_xbu(v_after) + regparam * v
            v_after = c_gets_axb(v_after, X1.T, X2, self.label_row_inds, self.label_col_inds) + regparam * v
            return v_after
        
        def mvr(v):
            raise Exception('You should not be here!')
            return None
        
        G = LinearOperator((kronfcount, kronfcount), matvec=mv, rmatvec=mvr, dtype=float64)
        
        v_init = array(self.Y).reshape(self.Y.shape[0])
        v_init = v_gets_xbu(v_init)
        v_init = array(v_init).reshape(kronfcount)
        self.W = mat(bicgstab(G, v_init)[0]).T.reshape((x1fsize, x2fsize),order='F')
        #self.A = self.A.reshape((K1.shape[1],K2.shape[0]),order='F')
        self.model = LinearPairwiseModel(self.W)
    
    
    def getModel(self):
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
        #P = K1pred.T * self.A * K2pred
        P = c_gets_axb(self.A, K1pred, K2pred.T, self.label_row_inds, self.label_col_inds)
        P = mat(P).reshape((K1pred.shape[0], K2pred.shape[0]), order='F')
        return P


class LinearPairwiseModel(object):
    
    def __init__(self, W):
        """Initializes the linear model
        @param W: primal coefficient matrix
        @type W: numpy matrix"""
        self.W = W
    
    
    def predictWithDataMatrices(self, X1pred, X2pred):
        P = X1pred * self.W * X2pred.T
        return P


