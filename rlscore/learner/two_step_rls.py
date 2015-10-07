
import pyximport; pyximport.install()

import numpy as np
import numpy.linalg as la

from rlscore.utilities import array_tools
from rlscore.utilities import decomposition


import cython_two_step_rls_cv


class TwoStepRLS(object):
    
    
    def __init__(self, **kwargs):
        Y = kwargs["Y"]
        Y = array_tools.as_labelmatrix(Y)
        self.Y = Y
        if kwargs.has_key('kmatrix1'):
            K1 = np.mat(kwargs['kmatrix1'])
            K2 = np.mat(kwargs['kmatrix2'])
            self.K1, self.K2 = K1, K2
            self.kernelmode = True
        else:
            X1 = np.mat(kwargs['xmatrix1'])
            X2 = np.mat(kwargs['xmatrix2'])
            self.X1, self.X2 = X1, X2
            self.kernelmode = False
        if kwargs.has_key('regparam1'):
            self.regparam1 = kwargs["regparam1"]
        else:
            self.regparam1 = kwargs["regparam"]
        if kwargs.has_key('regparam2'):
            self.regparam2 = kwargs["regparam2"]
        else:
            self.regparam2 = kwargs["regparam"]
        self.trained = False
    
    
    def createLearner(cls, **kwargs):
        learner = cls(**kwargs)
        return learner
    createLearner = classmethod(createLearner)
    
    
    def train(self):
        if self.kernelmode:
            self.solve_kernel(self.regparam1, self.regparam2)
        else:
            self.solve_linear(self.regparam1, self.regparam2)
    
    
    def solve_kernel(self, regparam1, regparam2):
        self.regparam1 = regparam1
        self.regparam2 = regparam2
        K1, K2 = self.K1, self.K2
        Y = self.Y.reshape((K1.shape[0], K2.shape[0]), order='F')
        #assert self.Y.shape == (self.K1.shape[0], self.K2.shape[0]), 'Y.shape!=(K1.shape[0],K2.shape[0]). Y.shape=='+str(Y.shape)+', K1.shape=='+str(self.K1.shape)+', K2.shape=='+str(self.K2.shape)
        if not self.trained:
            self.trained = True
            evals1, V  = decomposition.decomposeKernelMatrix(K1)
            evals1 = np.mat(evals1).T
            evals1 = np.multiply(evals1, evals1)
            V = np.mat(V)
            self.evals1 = evals1
            self.V = V
            
            evals2, U = decomposition.decomposeKernelMatrix(K2)
            evals2 = np.mat(evals2).T
            evals2 = np.multiply(evals2, evals2)
            U = np.mat(U)
            self.evals2 = evals2
            self.U = U
            self.VTYU = V.T * self.Y * U
        
        #newevals = 1. / (self.evals1 * self.evals2.T + regparam)
        self.newevals1 = 1. / (self.evals1 + regparam1)
        self.newevals2 = 1. / (self.evals2 + regparam2)
        newevals = self.newevals1 * self.newevals2.T
        
        self.A = np.multiply(self.VTYU, newevals)
        self.A = self.V * self.A * self.U.T
        self.model = KernelPairwisePredictor(self.A)
        
        #self.dsikm1 = la.inv(K1 + regparam1 * (np.mat(np.eye(K1.shape[0]))))
        #self.dsikm2 = la.inv(K2 + regparam2 * (np.mat(np.eye(K2.shape[0]))))
    
    
    def solve_linear(self, regparam1, regparam2):
        self.regparam1 = regparam1
        self.regparam2 = regparam2
        X1, X2 = self.X1, self.X2
        Y = self.Y.reshape((X1.shape[0], X2.shape[0]), order='F')
        if not self.trained:
            self.trained = True
            svals1, V, rsvecs1 = decomposition.decomposeDataMatrix(X1.T)
            self.svals1 = svals1.T
            self.evals1 = np.multiply(self.svals1, self.svals1)
            self.V = V
            self.rsvecs1 = np.mat(rsvecs1)
            
            if X1.shape == X2.shape and (X1 == X2).all():
                svals2, U, rsvecs2 = svals1, V, rsvecs1
            else:
                svals2, U, rsvecs2 = decomposition.decomposeDataMatrix(X2.T)
            self.svals2 = svals2.T
            self.evals2 = np.multiply(self.svals2, self.svals2)
            self.U = U
            self.rsvecs2 = np.mat(rsvecs2)
            
            self.VTYU = V.T * Y * U
        
        self.newevals1 = 1. / (self.evals1 + regparam1)
        self.newevals2 = 1. / (self.evals2 + regparam2)
        newevals = np.multiply(self.svals1, self.newevals1) * np.multiply(self.svals2, self.newevals2).T
        
        self.W = np.multiply(self.VTYU, newevals)
        self.W = self.rsvecs1.T * self.W * self.rsvecs2
        self.model = LinearPairwisePredictor(self.W)
    
    
    def computeLOO(self):
        
        bevals_col = np.multiply(self.evals2, self.newevals2).T
        
        svecsm_col = np.multiply(bevals_col, self.U)
        #print rightall.shape, svecsm.shape, self.Y.shape
        #right = svecsm.T * self.Y - multiply(svecsm, self.Y).T
        RQR_col = np.sum(np.multiply(self.U, svecsm_col), axis = 1)
        #RQY = sum(multiply(self.svecs.T, right), axis = 0)
        #RQY = sum(multiply(self.svecs.T, svecsm.T * self.Y), axis = 0) - sum(multiply(RQRT.T, self.Y), axis = 1).T
        #RQY = self.svecs * (svecsm.T * self.Y) - sum(multiply(RQR, self.Y), axis = 1)
        LOO_ek_col = (1. / (1. - RQR_col))
        #LOO = multiply(LOO_ek, RQY)
        #print LOO_ek.shape, (self.svecs * (svecsm.T * self.Y)).shape, RQR.shape, self.Y.shape
        LOO_col = (np.multiply(LOO_ek_col, self.U * (svecsm_col.T * self.Y.T)) - np.multiply(LOO_ek_col, np.multiply(RQR_col, self.Y.T))).T
        #print 'LOO_col', LOO_col
        
        
        bevals_row = np.multiply(self.evals1, self.newevals1).T
        
        svecsm_row = np.multiply(bevals_row, self.V)
        #print rightall.shape, svecsm.shape, self.Y.shape
        #right = svecsm.T * self.Y - multiply(svecsm, self.Y).T
        RQR_row = np.sum(np.multiply(self.V, svecsm_row), axis = 1)
        #RQY = sum(multiply(self.svecs.T, right), axis = 0)
        #RQY = sum(multiply(self.svecs.T, svecsm.T * self.Y), axis = 0) - sum(multiply(RQRT.T, self.Y), axis = 1).T
        #RQY = self.svecs * (svecsm.T * self.Y) - sum(multiply(RQR, self.Y), axis = 1)
        LOO_ek_row = (1. / (1. - RQR_row))
        #LOO = multiply(LOO_ek, RQY)
        #print LOO_ek.shape, (self.svecs * (svecsm.T * self.Y)).shape, RQR.shape, self.Y.shape
        LOO_two_step = np.multiply(LOO_ek_row, self.V * (svecsm_row.T * LOO_col)) - np.multiply(LOO_ek_row, np.multiply(RQR_row, LOO_col))
        return LOO_two_step
    
    
    def compute_symmetric_double_LOO(self):
        
        #bevals_col = np.multiply(self.evals2, self.newevals2).T
        #multiplyright = self.U.T * self.Y.T
        #I = np.mat(np.identity(2))
        
        G = np.multiply((self.newevals1.T-(1./self.regparam1)), self.V) * self.V.T + (1./self.regparam1) * np.mat(np.identity(self.K1.shape[0]))
        #G2 = np.multiply((self.newevals2.T-(1./self.regparam)), self.U) * self.U.T + (1./self.regparam) * np.mat(np.identity(self.K2.shape[0]))
        GY = G * self.Y
        #YG2 = self.Y * G2
        GYG = GY * G
        #A2 = G2 * self.Y.T
        
        i, j = 2, 4
        inds = [i, j]
        
        #A = self.U[inds]
        #right = multiplyright - A.T * self.Y.T[inds]
        #RQY = A * np.multiply(bevals_col.T, right)
        #B = np.multiply(bevals_col.T, A.T)
        #HO_col = (la.inv(I - A * B) * RQY).T
        
        #HO_col = (self.Y.T[inds]-la.inv(G2[np.ix_(inds, inds)]) * A2[inds]).T
        #print HO_col.shape
        
        
        #bevals_row = np.multiply(self.evals1, self.newevals1).T
        #multiplyright = self.V.T * HO_col
        
        #A = self.V[inds]
        #right = multiplyright - A.T * HO_col[inds]
        #RQY = A * np.multiply(bevals_row.T, right)
        #B = np.multiply(bevals_col.T, A.T)
        #HO_row = la.inv(I - A * B) * RQY
        
        #A1 = G1[inds] * HO_col
        #HO_row = HO_col[inds]-la.inv(G1[np.ix_(inds, inds)]) * A1
        
        #HO_rowr = self.Y[np.ix_(inds, inds)] \
        #    - YG2[np.ix_(inds, inds)] * la.inv(G2[np.ix_(inds, inds)]) \
        #    - la.inv(G1[np.ix_(inds, inds)]) * G1Y[np.ix_(inds, inds)] \
        #    + la.inv(G1[np.ix_(inds, inds)]) * G1YG2[np.ix_(inds, inds)] * la.inv(G2[np.ix_(inds, inds)])
        
        invGii = la.inv(G[np.ix_(inds, inds)])
        GYii = GY[np.ix_(inds, inds)]
        invGiiGYii = invGii * GYii
        HO_rowr = self.Y[np.ix_(inds, inds)] \
            - invGiiGYii.T \
            - invGiiGYii \
            + invGii * GYG[np.ix_(inds, inds)] * invGii
        
        #II1 = np.mat(np.identity(self.Y.shape[0]))[inds]
        #II2 = np.mat(np.identity(self.Y.shape[1]))[:, inds]
        #HO_rowr = (II1 - la.inv(G1[np.ix_(inds, inds)]) * G1[inds]) * self.Y * (II2 - G2[:, inds] * la.inv(G2[np.ix_(inds, inds)]))
        
        #print HO_row.shape
        results = np.zeros((self.Y.shape[0], self.Y.shape[1]))
        cython_two_step_rls_cv.compute_symmetric_double_loo(G, self.Y, GY, GYG, results, self.Y.shape[0], self.Y.shape[1])
        return results
    
    
    def getModel(self):
        return self.model

    
class KernelPairwisePredictor(object):
    
    def __init__(self, A, kernel = None):
        """Initializes the dual model
        @param A: dual coefficient matrix
        @type A: numpy matrix"""
        self.A = A
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
        #print K1pred.shape, self.A.shape, K2pred.shape
        P = np.array(np.dot(np.dot(K1pred, self.A), K2pred.T))
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
        P = np.array(np.dot(np.dot(X1pred, self.W), X2pred.T))
        return P


