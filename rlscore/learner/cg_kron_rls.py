

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

class CGKronRLS(AbstractLearner):

    def loadResources(self):
        Y = self.resource_pool[data_sources.TRAIN_LABELS]
        #self.K1 = mat(self.resource_pool['kmatrix1'])
        #self.K2 = mat(self.resource_pool['kmatrix2'])
        self.nonzeros_x_coord = self.resource_pool["nonzeros_x_coord"]
        self.nonzeros_y_coord = self.resource_pool["nonzeros_y_coord"]
        self.B = self.resource_pool["B"]
        Y = array_tools.as_labelmatrix(Y)
        #assert Y.shape == (self.K1.shape[0], self.K2.shape[0]), 'Y.shape!=(K1.shape[0],K2.shape[0]). Y.shape=='+str(Y.shape)+', K1.shape=='+str(self.K1.shape)+', K2.shape=='+str(self.K2.shape)
        self.Y = Y
        self.trained = False
    
    
    def train(self):
        regparam = self.resource_pool['regparam']
        self.solve(regparam)
    
    
    def solve(self, regparam):
        self.regparam = regparam
        K1 = mat(self.resource_pool['kmatrix1'])
        K2 = mat(self.resource_pool['kmatrix2'])
        lsize = len(self.nonzeros_x_coord) #n
        
        #Y = self.Y
        
        def mv(v):
            assert v.shape[0] == len(self.nonzeros_x_coord)
            temp = mat(zeros((K1.shape[1], K2.shape[0])))
            sparse_kronecker_multiplication_tools.sparse_mat_from_left(temp, v, K2, array(self.nonzeros_x_coord, dtype=int32), array(self.nonzeros_y_coord, dtype=int32), lsize, K2.shape[0])
            #for ind in range(len(self.nonzeros_x_coord)):
            #    i, j = self.nonzeros_y_coord[ind], self.nonzeros_x_coord[ind]
            #    temp[i] = temp[i] + v[ind] * K2.T[j]
            v_after = zeros(v.shape[0])
            sparse_kronecker_multiplication_tools.compute_subset_of_matprod_entries(v_after, K1, temp, array(self.nonzeros_x_coord, dtype=int32), array(self.nonzeros_y_coord, dtype=int32), lsize, K1.shape[0])
            #for ind in range(len(self.nonzeros_x_coord)):
            #    i, j = self.nonzeros_y_coord[ind], self.nonzeros_x_coord[ind]
            #    v_after[ind] = K1[i] * temp[:, j] + regparam * v[ind]
            return v_after + regparam * v
        
        def mvr(v):
            foofoo
            raise Exception('You should not be here!')
            return None
        
        G = LinearOperator((len(self.nonzeros_x_coord), len(self.nonzeros_x_coord)), matvec=mv, rmatvec=mvr, dtype=float64)
        #self.A = (mat(bicgstab(G, self.Y, maxiter = 1000)[0]).T)
        self.A = (mat(bicgstab(G, self.Y)[0]).T)
        #self.A = self.A.reshape((K1.shape[1],K2.shape[0]),order='F')
        self.A = (self.B.T*self.A).reshape((K1.shape[1],K2.shape[0]))
        self.model = PairwiseModel(self.A)
    
    
    def solve_linear(self, regparam):
        self.regparam = regparam
        X1 = mat(self.resource_pool['xmatrix1'])
        X2 = mat(self.resource_pool['xmatrix2'])
        
        x1tsize, x1fsize = X1.shape #m, d
        x2tsize, x2fsize = X2.shape #q, r
        lsize = len(self.nonzeros_x_coord) #n
        
        kronfcount = x1fsize * x2fsize
        
        if x1tsize * x1fsize * x2fsize + x2fsize * lsize < x2tsize * x1fsize * x2fsize + x1fsize * lsize:
            def u_gets_bxv(v):
                temp = v.reshape((x1fsize, x2fsize), order='F')
                temp = X1 * temp
                v_after = zeros(lsize)
                sparse_kronecker_multiplication_tools.compute_subset_of_matprod_entries(v_after, temp, X2.T, array(self.nonzeros_x_coord, dtype=int32), array(self.nonzeros_y_coord, dtype=int32), lsize, x2fsize)
                #for ind in range(lsize):
                #    i, j = self.nonzeros_y_coord[ind], self.nonzeros_x_coord[ind]
                #    v_after[ind] = temp[i] * X2.T[:, j]
                return v_after
            
            def v_gets_xbu(u):
                temp = mat(zeros((x1tsize, x2fsize)))
                sparse_kronecker_multiplication_tools.sparse_mat_from_left(temp, u, X2, array(self.nonzeros_x_coord, dtype=int32), array(self.nonzeros_y_coord, dtype=int32), lsize, x2fsize)
                #for ind in range(lsize):
                #    i, j = self.nonzeros_y_coord[ind], self.nonzeros_x_coord[ind]
                #    temp[i] = temp[i] + u[ind] * X2[j]
                temp = X1.T * temp
                return temp.reshape((kronfcount,),order='F')
        else:
            def u_gets_bxv(v):
                temp = v.reshape((x1fsize, x2fsize), order='F')
                temp = temp * X2.T
                v_after = zeros(lsize)
                sparse_kronecker_multiplication_tools.compute_subset_of_matprod_entries(v_after, X1, temp, array(self.nonzeros_x_coord, dtype=int32), array(self.nonzeros_y_coord, dtype=int32), lsize, x1fsize)
                #for ind in range(lsize):
                #    i, j = self.nonzeros_y_coord[ind], self.nonzeros_x_coord[ind]
                #    v_after[ind] = X1[i] * temp[:, j]
                return v_after
            
            def v_gets_xbu(u):
                temp = mat(zeros((x1fsize, x2tsize)))
                sparse_kronecker_multiplication_tools.sparse_mat_from_right(temp, u, X1.T, array(self.nonzeros_x_coord, dtype=int32), array(self.nonzeros_y_coord, dtype=int32), lsize, x1fsize)
                #for ind in range(lsize):
                #    i, j = self.nonzeros_y_coord[ind], self.nonzeros_x_coord[ind]
                #    temp[:, j] = temp[:, j] + X1.T[:, i] * u[ind]
                temp = temp * X2
                return temp.reshape((kronfcount,), order='F')
        
        def mv(v):
            '''temp = mat(zeros((x1tsize, x2fsize)))
            sparse_kronecker_multiplication_tools.kron_slice_multiply(temp, v, X2, array(self.nonzeros_x_coord, dtype=int32), array(self.nonzeros_y_coord, dtype=int32), lsize, x2fsize)
            #for ind in range(lsize):
            #    i, j = self.nonzeros_y_coord[ind], self.nonzeros_x_coord[ind]
            #    temp[i] = temp[i] + v[ind] * X2[j]
            temp = (X1.T * temp) * X2.T
            v_after = zeros(v.shape[0])
            for ind in range(lsize):
                i, j = self.nonzeros_y_coord[ind], self.nonzeros_x_coord[ind]
                v_after[ind] = X1[i] * temp[:, j] + regparam * v[ind]'''
            v_after = u_gets_bxv(v)
            v_after = v_gets_xbu(v_after) + regparam * v
            return v_after
        
        def mvr(v):
            raise Exception('You should not be here!')
            return None
        
        G = LinearOperator((kronfcount, kronfcount), matvec=mv, rmatvec=mvr, dtype=float64)
        #self.A = (mat(bicgstab(G, self.Y, maxiter = 1000)[0]).T)
        
        v_init = array(self.Y).reshape(self.Y.shape[0])
        v_init = v_gets_xbu(v_init)
        v_init = array(v_init).reshape(kronfcount)
        #print G.shape, v_init.shape
        self.W = mat(bicgstab(G, v_init)[0]).T.reshape((x1fsize, x2fsize),order='F')
        #self.A = self.A.reshape((K1.shape[1],K2.shape[0]),order='F')
        #self.A = (self.B.T*self.A).reshape((K1.shape[1],K2.shape[0]))
        self.model = LinearPairwiseModel(self.W)
    
    
    def getModel(self):
        return self.model

    
class PairwiseModel(object):
    
    def __init__(self, A, kernel = None):
        """Initializes the dual model
        @param A: dual coefficient matrix
        @type A: numpy matrix"""
        self.A = A
        self.kernel = kernel
    
    
    def predictWithKernelMatrices(self, K1pred, K2pred):
        P = K1pred.T * self.A * K2pred
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


