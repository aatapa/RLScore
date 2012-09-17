


from numpy import *
import numpy.linalg as la

from rlscore.learner.abstract_learner import AbstractLearner
from rlscore import data_sources
from rlscore import model
from rlscore.utilities import array_tools

class KronRLS(AbstractLearner):

    def loadResources(self):
        Y = self.resource_pool[data_sources.TRAIN_LABELS]
        self.K1 = self.resource_pool['kmatrix1']
        self.K2 = self.resource_pool['kmatrix2']
        Y = array_tools.as_labelmatrix(Y)
        assert Y.shape == (self.K1.shape[0], self.K2.shape[0]), 'Y.shape!=(K1.shape[0],K2.shape[0]). Y.shape=='+str(Y.shape)+', K1.shape=='+str(self.K1.shape)+', K2.shape=='+str(self.K2.shape)
        self.Y = Y
    
    
    def train(self):
        regparam = self.resource_pool['regparam']
        self.solve(regparam)
    
    def solve(self, regparam):
        self.regparam = regparam
        K1 = self.K1
        K2 = self.K2
        
        L, V  = la.eigh(K1)
        L = mat(L).T
        V = mat(V)
        self.L = L
        self.V = V
        
        S, U = la.eigh(K2)
        S = mat(S).T
        U = mat(U)
        self.S = S
        self.U = U
        
        newevals = 1. / (L * S.T + regparam)
        
        self.A = V.T * self.Y * U
        self.A = multiply(self.A, newevals)
        self.A = V * self.A * U.T
    
    
    def imputationLOO(self):
        P = self.K1.T * self.A * self.K2
        
        newevals = multiply(self.S * self.L.T, 1. / (self.S * self.L.T + self.regparam))
        Vsqr = multiply(self.V, self.V)
        Usqr = multiply(self.U, self.U)
        loopred = mat(zeros((self.V.shape[0], self.U.shape[0])))
        print self.U.shape[0], self.V.shape[0], self.Y.shape, loopred.shape, P.shape
        for i in range(self.V.shape[0]):
            cache = Vsqr[i] * newevals.T
            for j in range(self.U.shape[0]):
                ccc = (cache * Usqr[j].T)[0, 0]
                loopred[i, j] = (1. / (1. - ccc)) * (P[i, j] - ccc * self.Y[i, j])
                #loopred[i, j] = P[i, j]
        return loopred
    
    
    def computeBlockHO(self, hoinds1, hoinds2):
        '''
        #UNFINISHED CODE
        #P = ((K2.T * self.A * K1).T).reshape(self.Y.shape[0] * self.Y.shape[1],1)
        P = (self.K2.T * self.A * self.K1).T
        
        newevals = multiply(self.S * self.L.T, 1. / (self.S * self.L.T + self.regparam))
        Vsqr = multiply(self.V, self.V)
        Usqr = multiply(self.U, self.U)
        loopred = mat(zeros((self.U.shape[0], self.V.shape[0])))
        for i in hoinds1:
            cache = Usqr[i] * newevals
            for j in hoinds2:
                ccc = (cache * Vsqr[j].T)[0, 0]
                loopred[i, j] = (1. / (1. - ccc)) * (P[i, j] - ccc * self.Y[i, j])
        return loopred
        '''
    
    
    def getModel(self):
        model = PairwiseModel(self.A)
        return model

    
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


