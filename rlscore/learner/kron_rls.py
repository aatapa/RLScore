
import pyximport; pyximport.install()

import numpy as np
import numpy.linalg as la

from rlscore.utilities import array_tools
from rlscore.utilities import decomposition

from rlscore.utilities import sparse_kronecker_multiplication_tools

from rlscore.pairwise_predictor import LinearPairwisePredictor
from rlscore.pairwise_predictor import KernelPairwisePredictor

from rlscore.pairwise_predictor import PairwisePredictorInterface

class KronRLS(PairwisePredictorInterface):
    
    
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
        if kwargs.has_key("regparam"):
            self.regparam = kwargs["regparam"]
        else:
            self.regparam = 1.
        self.trained = False
        self.train()
    
    
    def train(self):
        if self.kernelmode:
            self.solve_kernel(self.regparam)
        else:
            self.solve_linear(self.regparam)
    
    
    def solve_kernel(self, regparam):
        self.regparam = regparam
        K1, K2 = self.K1, self.K2
        #assert self.Y.shape == (self.K1.shape[0], self.K2.shape[0]), 'Y.shape!=(K1.shape[0],K2.shape[0]). Y.shape=='+str(Y.shape)+', K1.shape=='+str(self.K1.shape)+', K2.shape=='+str(self.K2.shape)
        if not self.trained:
            self.trained = True
            evals1, V  = la.eigh(K1)
            evals1 = np.mat(evals1).T
            V = np.mat(V)
            self.evals1 = evals1
            self.V = V
            
            evals2, U = la.eigh(K2)
            evals2 = np.mat(evals2).T
            U = np.mat(U)
            self.evals2 = evals2
            self.U = U
            self.VTYU = V.T * self.Y * U
        
        newevals = 1. / (self.evals1 * self.evals2.T + regparam)
        
        self.A = np.multiply(self.VTYU, newevals)
        self.A = self.V * self.A * self.U.T
        self.A = np.asarray(self.A)
        label_row_inds, label_col_inds = np.unravel_index(np.arange(K1.shape[0] * K2.shape[0]), (K1.shape[0],  K2.shape[0]))
        label_row_inds = np.array(label_row_inds, dtype = np.int32)
        label_col_inds = np.array(label_col_inds, dtype = np.int32)
        self.predictor = KernelPairwisePredictor(self.A.ravel(), label_row_inds, label_col_inds)
    
    
    def solve_linear(self, regparam):
        self.regparam = regparam
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
        
        kronsvals = self.svals1 * self.svals2.T
        
        newevals = np.divide(kronsvals, np.multiply(kronsvals, kronsvals) + regparam)
        self.W = np.multiply(self.VTYU, newevals)
        self.W = self.rsvecs1.T * self.W * self.rsvecs2
        self.predictor = LinearPairwisePredictor(np.array(self.W))
    
    
    def solve_linear_conditional_ranking(self, regparam):
        self.regparam = regparam
        X1, X2 = self.X1, self.X2
        Y = self.Y.reshape((X1.shape[0], X2.shape[0]), order = 'F')
        
        svals1, V, rsvecs1 = decomposition.decomposeDataMatrix(X1.T)
        self.svals1 = svals1.T
        self.evals1 = np.multiply(self.svals1, self.svals1)
        self.V = V
        self.rsvecs1 = np.mat(rsvecs1)
        
        qlen = X2.shape[0]
        onevec = (1. / np.math.sqrt(qlen)) * np.mat(np.ones((qlen, 1)))
        C = np.mat(np.eye(qlen)) - onevec * onevec.T
        
        svals2, U, rsvecs2 = decomposition.decomposeDataMatrix(X2.T * C)
        self.svals2 = svals2.T
        self.evals2 = np.multiply(self.svals2, self.svals2)
        self.U = U
        self.rsvecs2 = np.mat(rsvecs2)
        
        self.VTYU = V.T * Y * C * U
        
        kronsvals = self.svals1 * self.svals2.T
        
        newevals = np.divide(kronsvals, np.multiply(kronsvals, kronsvals) + regparam)
        self.W = np.multiply(self.VTYU, newevals)
        self.W = self.rsvecs1.T * self.W * self.rsvecs2
        self.predictor = LinearPairwisePredictor(np.array(self.W))
    
    
    def in_sample_loo(self):
        if not self.kernelmode:
            X1, X2 = self.X1, self.X2
            P = X1 * self.W * X2.T
        else:
            P = self.K1 * self.A * self.K2.T
        
        newevals = np.multiply(self.evals2 * self.evals1.T, 1. / (self.evals2 * self.evals1.T + self.regparam))
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
        return np.asarray(loopred).ravel(order = 'F')
    
    
    def compute_ho(self, row_inds, col_inds):
        if not self.kernelmode:
            X1, X2 = self.X1, self.X2
            P_ho = X1[row_inds] * self.W * X2.T[:, col_inds]
        else:
            P_ho = self.K1[row_inds] * self.A * self.K2.T[:, col_inds]
        
        newevals = np.multiply(self.evals2 * self.evals1.T, 1. / (self.evals2 * self.evals1.T + self.regparam))
        
        rowcount = len(row_inds)
        colcount = len(col_inds)
        hosize = rowcount * colcount
        
        VV = np.mat(np.zeros((rowcount * rowcount, self.V.shape[1])))
        UU = np.mat(np.zeros((colcount * colcount, self.U.shape[1])))
        
        def bar():
            for i in range(len(row_inds)):
                ith_row = self.V[row_inds[i]]
                for h in range(len(row_inds)):
                    VV[i * rowcount + h] = np.multiply(ith_row, self.V[row_inds[h]])
            
            for j in range(len(col_inds)):
                jth_col = self.U[col_inds[j]]
                for k in range(len(col_inds)):
                    UU[j * colcount + k] = np.multiply(jth_col, self.U[col_inds[k]])
        
        def baz():
            #print VV.shape, newevals.shape, UU.T.shape
            B_in_wrong_order = VV * newevals.T * UU.T
            
            #B_in_right_order = mat(zeros((hosize, hosize)))
            
            #for i in range(len(row_inds)):
            #    for j in range(len(col_inds)):
            #        for h in range(len(row_inds)):
            #            for k in range(len(col_inds)):
            #                B_in_right_order[i * colcount + j, h * colcount + k] = B_in_wrong_order[i * rowcount + h, j * colcount + k]
            
            #print B_in_right_order
            #print B_in_right_order.shape, B_in_wrong_order.shape, rowcount, colcount
            B_in_right_order = np.mat(np.zeros((hosize, hosize)))
            sparse_kronecker_multiplication_tools.cpy_reorder(B_in_right_order, B_in_wrong_order, rowcount, colcount)
            #print B_in_right_order
            #print
            hopred = la.inv(np.mat(np.eye(hosize)) - B_in_right_order) * (P_ho.ravel().T - B_in_right_order * self.Y[np.ix_(row_inds, col_inds)].ravel().T)
            return hopred
        bar()
        hopred = baz()
        #print rowcount, colcount, hosize, hopred.shape
        return np.asarray(hopred.reshape(rowcount, colcount))
    
    
    def nested_in_sample_loo(self, outer_row_coord, outer_col_coord,):
        if not self.kernelmode:
            X1, X2 = self.X1, self.X2
            P = X1 * self.W * X2.T
        else:
            P = self.K1 * self.A * self.K2.T
        P_out = P[outer_row_coord, outer_col_coord]
        Y_out = self.Y[outer_row_coord, outer_col_coord]
        
        newevals = np.multiply(self.evals2 * self.evals1.T, 1. / (self.evals2 * self.evals1.T + self.regparam))
        Vsqr = np.multiply(self.V, self.V)
        Usqr = np.multiply(self.U, self.U)
        d = (Vsqr[outer_row_coord] * newevals.T * Usqr[outer_col_coord].T)[0, 0]
        dY_out = d * Y_out
        
        Vox = np.multiply(self.V, self.V[outer_row_coord])
        Uoy = np.multiply(self.U, self.U[outer_col_coord])
        
        cache = Vsqr * newevals.T * Usqr.T
        crosscache = Vox * newevals.T * Uoy.T
        
        #loopred = mat(zeros((self.V.shape[0], self.U.shape[0])))
        #for i in range(self.V.shape[0]):
        #    jinds = range(self.U.shape[0])
        #    if i == outer_row_coord:
        #        jinds.remove(outer_col_coord)
#            for j in jinds:
#                a = cache[i, j]
#                bc = crosscache[i, j]
#                invdetGshift = 1. / ((1. - a) * (1. - d) - bc * bc)
#                invGshift_1 = invdetGshift * (1. - d)
#                invGshift_2 = invdetGshift * bc
#                Yij = self.Y[i, j]
#                temp1 = P[i, j] - (a * Yij + bc * Y_out)
#                temp2 = P_out - (bc * Yij + dY_out)
#                loopred[i, j] = invGshift_1 * temp1 + invGshift_2 * temp2
#            a = cache[i]
#            bc = crosscache[i]
#            invdetGshift = divide(1., ((1. - a) * (1. - d) - multiply(bc, bc)))
#            invGshift_1 = invdetGshift * (1. - d)
#            invGshift_2 = multiply(invdetGshift, bc)
#            Yi = self.Y[i]
#            temp1 = P[i] - (multiply(a, Yi) + bc * Y_out)
#            temp2 = P_out - (multiply(bc, Yi) + dY_out)
#            loopred[i] = multiply(invGshift_1, temp1) + multiply(invGshift_2, temp2)
        a = cache
        bc = crosscache
        invdetGshift = np.divide(1., ((1. - a) * (1. - d) - np.multiply(bc, bc)))
        invGshift_1 = invdetGshift * (1. - d)
        invGshift_2 = np.multiply(invdetGshift, bc)
        Y = self.Y
        temp1 = P - (np.multiply(a, Y) + bc * Y_out)
        temp2 = P_out - (np.multiply(bc, Y) + dY_out)
        loopred = np.multiply(invGshift_1, temp1) + np.multiply(invGshift_2, temp2)
        return np.artray(loopred)
    
    
    def nested_in_sample_loo_BU(self, outer_row_coord, outer_col_coord):
        P = self.K1.T * self.A * self.K2
        P_out = P[outer_row_coord, outer_col_coord]
        Y_out = self.Y[outer_row_coord, outer_col_coord]
        
        newevals = np.multiply(self.evals2 * self.evals1.T, 1. / (self.evals2 * self.evals1.T + self.regparam))
        Vsqr = np.multiply(self.V, self.V)
        Usqr = np.multiply(self.U, self.U)
        d = (Vsqr[outer_row_coord] * newevals.T * Usqr[outer_col_coord].T)[0, 0]
        dY_out = d * Y_out
        
        Vox = np.multiply(self.V, self.V[outer_row_coord])
        Uoy = np.multiply(self.U, self.U[outer_col_coord])
        
        cache = Vsqr * newevals.T * Usqr.T
        crosscache = Vox * newevals.T * Uoy.T
        
        loopred = np.mat(np.zeros((self.V.shape[0], self.U.shape[0])))
        #print self.U.shape[0], self.V.shape[0], self.Y.shape, loopred.shape, P.shape
        for i in range(self.V.shape[0]):
            #cache = Vsqr[i] * newevals.T
            #crosscache = Vox[i] * newevals.T
            jinds = range(self.U.shape[0])
            if i == outer_row_coord:
                jinds.remove(outer_col_coord)
            for j in jinds:
                #a = (cache * Usqr[j].T)[0, 0]
                a = cache[i, j]
                #bc = (crosscache * Uoy[j].T)[0, 0]
                bc = crosscache[i, j]
                #G = mat([[a, bc], [bc, d]])
                #invG = 1. / (a * d - bc * bc) * mat([[d, -bc], [-bc, a]])
                #invGshift = 1. / ((1. - a) * (1. - d) - bc * bc) * mat([[(1. - d), bc], [bc, (1. - a)]])
                invdetGshift = 1. / ((1. - a) * (1. - d) - bc * bc)
                invGshift_1 = invdetGshift * (1. - d)
                invGshift_2 = invdetGshift * bc
                #YY = mat([self.Y[i, j], self.Y[outer_row_coord, outer_col_coord]]).T
                #PP = mat([P[i, j], P[outer_row_coord, outer_col_coord]]).T
                #loopred[i, j] = (invGshift * (PP - G * YY))[0, 0]
                Yij = self.Y[i, j]
                temp1 = P[i, j] - (a * Yij + bc * Y_out)
                temp2 = P_out - (bc * Yij + dY_out)
                loopred[i, j] = invGshift_1 * temp1 + invGshift_2 * temp2
                #loopred[i, j] = (1. / (1. - ccc)) * (P[i, j] - ccc * self.Y[i, j])
                #loopred[i, j] = P[i, j]
        return np.asarray(loopred)
    
    
    def prepareLooCaches(self):
        
        #Hirvee hakkerointi
        if not hasattr(self, "Vsqr"):
            self.Vsqr = np.multiply(self.V, self.V)
            self.Usqr = np.multiply(self.U, self.U)
        self.newlooevals = np.multiply(self.evals2 * self.evals1.T, 1. / (self.evals2 * self.evals1.T + self.regparam))
        self.P = self.K1.T * self.A * self.K2
        self.newlooevalsUsqr = self.newlooevals.T * self.Usqr.T
        self.Vsqrnewlooevals = self.Vsqr * self.newlooevals.T
        self.Vcache = self.Vsqr * self.newlooevalsUsqr
        self.Ucache = self.Vsqrnewlooevals * self.Usqr.T
        self.diagGcache = self.Vsqr * self.newlooevals.T * self.Usqr.T
    
    
    def nested_in_sample_looApproximation(self, outer_row_coord, outer_col_coord):
        Y_out = self.Y[outer_row_coord, outer_col_coord]
        
        #d = (self.Vsqr[outer_row_coord] * self.newlooevals.T * self.Usqr[outer_col_coord].T)[0, 0]
        ddd = self.diagGcache[outer_row_coord, outer_col_coord]
        dY_out = ddd * Y_out
        one_minus_d = 1. - ddd
        
        #P_col = self.K1.T * (self.A * self.K2[:, outer_col_coord])
        #P_row = (self.K1[outer_row_coord] * self.A) * self.K2
        P_out = self.P[outer_row_coord, outer_col_coord]
        
        #Vox = multiply(self.V, self.V[outer_row_coord])
        #Uoy = multiply(self.U, self.U[outer_col_coord])
        
        #cache = self.Vsqr * self.newlooevals.T * self.Usqr.T
        #crosscache = Vox * self.newlooevals.T * Uoy.T
        #Vcache = self.Vsqr * (self.newlooevalsUsqr[:, outer_col_coord])
        #Vcrosscache = Vox * (self.newlooevals.T * Uoy[outer_col_coord].T)
        Vcrosscache = self.V * np.multiply(self.V[outer_row_coord].T, self.newlooevalsUsqr[:, outer_col_coord])
        
        #loopred = mat(zeros((self.V.shape[0], self.U.shape[0])))
        #print self.U.shape[0], self.V.shape[0], self.Y.shape, loopred.shape, P.shape
        
        VcrosscacheSqr = np.multiply(Vcrosscache, Vcrosscache)
        
        a = self.Vcache[:, outer_col_coord]
        #bc = (crosscache * Uoy[j].T)[0, 0]
        bc = Vcrosscache
        #G = mat([[a, bc], [bc, d]])
        #invG = 1. / (a * d - bc * bc) * mat([[d, -bc], [-bc, a]])
        #invGshift = 1. / ((1. - a) * (1. - d) - bc * bc) * mat([[(1. - d), bc], [bc, (1. - a)]])
        invdetGshift = 1. / ((1. - a) * one_minus_d - VcrosscacheSqr)
        invGshift_1 = invdetGshift * one_minus_d
        invGshift_2 = np.multiply(invdetGshift, bc)
        #YY = mat([self.Y[i, j], self.Y[outer_row_coord, outer_col_coord]]).T
        #PP = mat([P[i, j], P[outer_row_coord, outer_col_coord]]).T
        #loopred[i, j] = (invGshift * (PP - G * YY))[0, 0]
        Y_j = self.Y[:, outer_col_coord]#self.Y[i, outer_col_coord]
        temp1 = self.P[:, outer_col_coord] - (np.multiply(a, Y_j) + bc * Y_out)
        temp2 = P_out - (np.multiply(bc, Y_j) + dY_out)
        loocolumn = np.multiply(invGshift_1, temp1) + np.multiply(invGshift_2, temp2)
        #loopred[i, j] = (1. / (1. - ccc)) * (P[i, j] - ccc * self.Y[i, j])
        #loopred[i, j] = P[i, j]
        
        #Ucache = self.Vsqrnewlooevals[outer_row_coord] * self.Usqr.T
        Ucrosscache = np.multiply(self.Vsqrnewlooevals[outer_row_coord], self.U[outer_col_coord]) * self.U.T
        
        #loopred = mat(zeros((self.V.shape[0], self.U.shape[0])))
        #print self.U.shape[0], self.V.shape[0], self.Y.shape, loopred.shape, P.shape
        '''
        for j in range(self.Y.shape[1]):
            if j == outer_col_coord: continue
            #a = (cache * self.Usqr[j].T)[0, 0]
            a = self.Ucache[outer_row_coord, j]
            #bc = (crosscache * Uoy[j].T)[0, 0]
            bc = Ucrosscache[0, j]
            #G = mat([[a, bc], [bc, d]])
            #invG = 1. / (a * d - bc * bc) * mat([[d, -bc], [-bc, a]])
            #invGshift = 1. / ((1. - a) * (1. - d) - bc * bc) * mat([[(1. - d), bc], [bc, (1. - a)]])
            invdetGshift = 1. / ((1. - a) * one_minus_d - bc * bc)
            invGshift_1 = invdetGshift * one_minus_d
            invGshift_2 = invdetGshift * bc
            #YY = mat([self.Y[i, j], self.Y[outer_row_coord, outer_col_coord]]).T
            #PP = mat([P[i, j], P[outer_row_coord, outer_col_coord]]).T
            #loopred[i, j] = (invGshift * (PP - G * YY))[0, 0]
            Yij = self.Ylist[outer_row_coord][j]#self.Y[outer_row_coord, j]
            temp1 = self.P[outer_row_coord, j] - (a * Yij + bc * Y_out)
            temp2 = P_out - (bc * Yij + dY_out)
            loopred[outer_row_coord, j] = invGshift_1 * temp1 + invGshift_2 * temp2
            #loopred[i, j] = (1. / (1. - ccc)) * (P[i, j] - ccc * self.Y[i, j])
            #loopred[i, j] = P[i, j]
        '''
        
        UcrosscacheSqr = np.multiply(Ucrosscache, Ucrosscache)
        
        #a = (cache * self.Usqr[j].T)[0, 0]
        a = self.Ucache[outer_row_coord]
        #bc = (crosscache * Uoy[j].T)[0, 0]
        bc = Ucrosscache
        #G = mat([[a, bc], [bc, d]])
        #invG = 1. / (a * d - bc * bc) * mat([[d, -bc], [-bc, a]])
        #invGshift = 1. / ((1. - a) * (1. - d) - bc * bc) * mat([[(1. - d), bc], [bc, (1. - a)]])
        invdetGshift = 1. / ((1. - a) * one_minus_d - UcrosscacheSqr)
        invGshift_1 = invdetGshift * one_minus_d
        invGshift_2 = np.multiply(invdetGshift, bc)
        #YY = mat([self.Y[i, j], self.Y[outer_row_coord, outer_col_coord]]).T
        #PP = mat([P[i, j], P[outer_row_coord, outer_col_coord]]).T
        #loopred[i, j] = (invGshift * (PP - G * YY))[0, 0]
        Yi = self.Y[outer_row_coord]#self.Y[outer_row_coord, j]
        temp1 = self.P[outer_row_coord] - (np.multiply(a, Yi) + np.multiply(bc, Y_out))
        temp2 = P_out - (np.multiply(bc, Yi) + dY_out)
        loorow = np.multiply(invGshift_1, temp1) + np.multiply(invGshift_2, temp2)
        #loopred[i, j] = (1. / (1. - ccc)) * (P[i, j] - ccc * self.Y[i, j])
        #loopred[i, j] = P[i, j]
        
        loocolumn[outer_row_coord] = 0
        loorow[:, outer_col_coord] = 0
        return loocolumn, loorow



