
import pyximport; pyximport.install()

import numpy as np
import numpy.linalg as la

from rlscore.utilities import array_tools
from rlscore.utilities import decomposition


import cython_two_step_rls_cv

from rlscore.pairwise_predictor import PairwisePredictorInterface
from rlscore.pairwise_predictor import LinearPairwisePredictor
from rlscore.pairwise_predictor import KernelPairwisePredictor

class TwoStepRLS(PairwisePredictorInterface):
    
    
    def __init__(self, **kwargs):
        Y = kwargs["Y"]
        Y = array_tools.as_labelmatrix(Y)
        if kwargs.has_key('kmatrix1'):
            K1 = np.mat(kwargs['kmatrix1'])
            K2 = np.mat(kwargs['kmatrix2'])
            Y = Y.reshape((K1.shape[0], K2.shape[0]), order = 'F')
            self.K1, self.K2 = K1, K2
            self.kernelmode = True
        else:
            X1 = np.mat(kwargs['xmatrix1'])
            X2 = np.mat(kwargs['xmatrix2'])
            Y = Y.reshape((X1.shape[0], X2.shape[0]), order = 'F')
            self.X1, self.X2 = X1, X2
            self.kernelmode = False
        self.Y = Y
        if kwargs.has_key('regparam1'):
            self.regparam1 = kwargs["regparam1"]
        else:
            self.regparam1 = kwargs["regparam"]
        if kwargs.has_key('regparam2'):
            self.regparam2 = kwargs["regparam2"]
        else:
            self.regparam2 = kwargs["regparam"]
        self.trained = False
        self.solve(self.regparam1, self.regparam2)
    
    
    def solve(self, regparam1, regparam2):
        self.regparam1 = regparam1
        self.regparam2 = regparam2
        if self.kernelmode:
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
            self.A = np.array(self.A)
            #self.predictor = KernelPairwisePredictor(self.A)
            label_row_inds, label_col_inds = np.unravel_index(np.arange(K1.shape[0] * K2.shape[0]), (K1.shape[0],  K2.shape[0]))
            label_row_inds = np.array(label_row_inds, dtype = np.int32)
            label_col_inds = np.array(label_col_inds, dtype = np.int32)
            self.predictor = KernelPairwisePredictor(self.A.ravel(), label_row_inds, label_col_inds)
            
            #self.dsikm1 = la.inv(K1 + regparam1 * (np.mat(np.eye(K1.shape[0]))))
            #self.dsikm2 = la.inv(K2 + regparam2 * (np.mat(np.eye(K2.shape[0]))))
        else:
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
            #self.predictor = LinearPairwisePredictor(self.W)
            self.predictor = LinearPairwisePredictor(np.array(self.W))
    
    
    def in_sample_loo(self):
        if not self.kernelmode:
            X1, X2 = self.X1, self.X2
            P = X1 * self.W * X2.T
        else:
            P = self.K1 * self.A * self.K2.T
        
        newevals = np.multiply(self.evals2 * self.evals1.T, 1. / ((self.evals2 + self.regparam2) * (self.evals1.T + self.regparam1)))
        #newevals = np.multiply(self.evals2 * self.evals1.T + self.regparam1 * self.evals2 + self.regparam2 * self.evals1.T, 1. / ((self.evals2 + self.regparam2) * (self.evals1.T + self.regparam1)))
        #P = np.multiply(self.VTYU, newevals)
        #P = self.V * P * self.U.T
        #P = np.array(P)
        Vsqr = np.multiply(self.V, self.V)
        Usqr = np.multiply(self.U, self.U)
        #loopred = mat(zeros((self.V.shape[0], self.U.shape[0])))
        #print self.U.shape[0], self.V.shape[0], self.Y.shape, loopred.shape, P.shape
        #for i in range(self.V.shape[0]):
            #cache = Vsqr[i] * newevals.T
            #for j in range(self.U.shape[0]):
            #    ccc = (cache * Usqr[j].T)[0, 0]
            #    loopred[i, j] = (1. / (1. - ccc)) * (P[i, j] - ccc * self.Y[i, j])
            #    #loopred[i, j] = P[i, j]
        ccc = Vsqr * newevals.T * Usqr.T
        loopred = np.multiply(1. / (1. - ccc), P - np.multiply(ccc, self.Y))
        return np.asarray(loopred)
    
    
    def in_sample_loo_ref(self):
        if not self.kernelmode:
            X1, X2 = self.X1, self.X2
            P = X1 * self.W * X2.T
        else:
            P = self.K1 * self.A * self.K2.T
        
        ylen = self.Y.shape[0] * self.Y.shape[1]
        hocompl = [0] + range(2, ylen)
        kron = np.kron(self.K2, self.K1)
        regkron = np.kron(self.K2 + self.regparam2 * np.eye(self.K2.shape[0]), self.K1 + self.regparam1 * np.eye(self.K1.shape[0]))
        invregkron = la.inv(regkron)
        
        weirdkron = kron - self.regparam2 * self.regparam1 * np.eye(self.K2.shape[0] * self.K1.shape[0])
        invregweirdkron = la.inv(weirdkron + self.regparam2 * self.regparam1 * np.eye(self.K2.shape[0] * self.K1.shape[0]))
        #invregkron = np.kron(la.inv(self.K2 + self.regparam2 * np.eye(self.K2.shape[0])), la.inv(self.K1 + self.regparam1 * np.eye(self.K1.shape[0])))
        #invregkron_sampled = invregkron[hocompl][:, hocompl]
        #regkron_sampled = regkron[hocompl][:, hocompl]
        #invregkron_sampled = la.inv(regkron_sampled)
        #predkron = np.kron(self.K2, self.K1)[1][:, hocompl]
        #y_sampled = self.Y.ravel(order = 'F').T[hocompl]
        #print self.Y, self.Y.ravel(order = 'F'), len(y_sampled)
        YY = self.Y.ravel(order = 'F').T.copy()
        YY[1] = 5
        P = P.ravel(order = 'F').T
        PP = np.dot(np.dot(kron, invregkron), YY)
        #loopred = np.dot(predkron, np.dot(invregkron_sampled, y_sampled))
        #loopred_a = self.Y.ravel(order = 'F').T[1] - (1. / regkron[1, 1]) * np.dot(invregkron, self.Y.ravel(order = 'F').T)[1]
        #loopred_b = YY[1] - (1. / regkron[1, 1]) * np.dot(invregkron, YY)[1]
        #loopred_a = self.Y.ravel(order = 'F').T[1] - (1. / invregweirdkron[1, 1]) * np.dot(invregweirdkron, self.Y.ravel(order = 'F').T)[1]
        #loopred_b = YY[1] - (1. / invregweirdkron[1, 1]) * np.dot(invregweirdkron, YY)[1]
        loopred_a = (1. / (1. - (weirdkron * invregweirdkron)[1, 1])) * (P[1] - (weirdkron * invregweirdkron)[1, 1] * self.Y.ravel(order = 'F').T[1])
        loopred_b = (1. / (1. - (weirdkron * invregweirdkron)[1, 1])) * (PP[1] - (weirdkron * invregweirdkron)[1, 1] * YY[1])
        loopred = (1. / (1. - (kron * invregkron)[1, 1])) * (P[1] - (kron * invregkron)[1, 1] * self.Y.ravel(order = 'F').T[1])
        loopred2 = (1. / (1. - (kron * invregkron)[1, 1])) * (PP[1] - (kron * invregkron)[1, 1] * YY[1])
        foo = np.zeros(ylen)
        #foo[]
        return loopred, loopred2, loopred_a, loopred_b
    
    
    def out_of_sample_loo(self):
        
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
        return LOO_two_step.ravel(order = 'F')
    
    
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
        return results.ravel(order = 'F')


