
import pyximport; pyximport.install()

from numpy import *
import numpy.linalg as la

from rlscore.learner.abstract_learner import AbstractLearner
from rlscore import data_sources
from rlscore import model
from rlscore.utilities import array_tools
from rlscore.utilities import decomposition

from rlscore.utilities import sparse_kronecker_multiplication_tools


class KronRLS(AbstractLearner):

    def loadResources(self):
        Y = self.resource_pool[data_sources.TRAIN_LABELS]
        self.K1 = mat(self.resource_pool['kmatrix1'])
        self.K2 = mat(self.resource_pool['kmatrix2'])
        Y = array_tools.as_labelmatrix(Y)
        assert Y.shape == (self.K1.shape[0], self.K2.shape[0]), 'Y.shape!=(K1.shape[0],K2.shape[0]). Y.shape=='+str(Y.shape)+', K1.shape=='+str(self.K1.shape)+', K2.shape=='+str(self.K2.shape)
        self.Y = Y
        self.trained = False
    
    
    def train(self):
        regparam = self.resource_pool['regparam']
        self.solve(regparam)
    
    def solve(self, regparam):
        self.regparam = regparam
        K1 = self.K1
        K2 = self.K2
        
        if not self.trained:
            self.trained = True
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
            self.VTYU = V.T * self.Y * U
        
        newevals = 1. / (self.L * self.S.T + regparam)
        
        self.A = multiply(self.VTYU, newevals)
        self.A = self.V * self.A * self.U.T
    
    
    def solve_linear_hack(self, regparam):
        self.regparam = regparam
        X1 = mat(self.resource_pool['xmatrix1'])
        X2 = mat(self.resource_pool['xmatrix2'])
        
        if not self.trained:
            self.trained = True
            L, V, rsvecs1 = decomposition.decomposeDataMatrix(X1.T)
            self.L = L.T
            self.V = V
            self.rsvecs1 = mat(rsvecs1)
            
            S, U, rsvecs2 = decomposition.decomposeDataMatrix(X2.T)
            self.S = S.T
            self.U = U
            self.rsvecs2 = mat(rsvecs2)
            
            self.VTYU = V.T * self.Y * U
        
        kronsvals = self.L * self.S.T
        
        newevals = divide(kronsvals, multiply(kronsvals, kronsvals) + regparam)
        self.W = multiply(self.VTYU, newevals)
        self.W = self.rsvecs1.T * self.W * self.rsvecs2
    
    
    def imputationLOO(self):
        P = self.K1.T * self.A * self.K2
        
        newevals = multiply(self.S * self.L.T, 1. / (self.S * self.L.T + self.regparam))
        Vsqr = multiply(self.V, self.V)
        Usqr = multiply(self.U, self.U)
        #loopred = mat(zeros((self.V.shape[0], self.U.shape[0])))
        #print self.U.shape[0], self.V.shape[0], self.Y.shape, loopred.shape, P.shape
        #for i in range(self.V.shape[0]):
            #cache = Vsqr[i] * newevals.T
            #for j in range(self.U.shape[0]):
            #    ccc = (cache * Usqr[j].T)[0, 0]
            #    loopred[i, j] = (1. / (1. - ccc)) * (P[i, j] - ccc * self.Y[i, j])
            #    #loopred[i, j] = P[i, j]
        ccc = Vsqr * newevals.T * Usqr.T
        loopred = multiply(1. / (1. - ccc), P - multiply(ccc, self.Y))
        return loopred
    
    
    def compute_ho(self, row_inds, col_inds):
        
        P_ho = self.K1.T[row_inds] * self.A * self.K2[:, col_inds]
        
        newevals = multiply(self.S * self.L.T, 1. / (self.S * self.L.T + self.regparam))
        
        rowcount = len(row_inds)
        colcount = len(col_inds)
        hosize = rowcount * colcount
        
        VV = mat(zeros((rowcount * rowcount, self.V.shape[1])))
        UU = mat(zeros((colcount * colcount, self.U.shape[1])))
        
        def bar():
            for i in range(len(row_inds)):
                ith_row = self.V[row_inds[i]]
                for h in range(len(row_inds)):
                    VV[i * rowcount + h] = multiply(ith_row, self.V[row_inds[h]])
            
            for j in range(len(col_inds)):
                jth_col = self.U[col_inds[j]]
                for k in range(len(col_inds)):
                    UU[j * colcount + k] = multiply(jth_col, self.U[col_inds[k]])
        
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
            B_in_right_order = mat(zeros((hosize, hosize)))
            sparse_kronecker_multiplication_tools.cpy_reorder(B_in_right_order, B_in_wrong_order, rowcount, colcount)
            #print B_in_right_order
            #print
            hopred = la.inv(mat(eye(hosize)) - B_in_right_order) * (P_ho.ravel().T - B_in_right_order * self.Y[ix_(row_inds, col_inds)].ravel().T)
            return hopred
        bar()
        hopred = baz()
        #print rowcount, colcount, hosize, hopred.shape
        return hopred.reshape(rowcount, colcount)
    
    
    def nested_imputationLOO(self, outer_row_coord, outer_col_coord):
        P = self.K1.T * self.A * self.K2
        P_out = P[outer_row_coord, outer_col_coord]
        Y_out = self.Y[outer_row_coord, outer_col_coord]
        
        newevals = multiply(self.S * self.L.T, 1. / (self.S * self.L.T + self.regparam))
        Vsqr = multiply(self.V, self.V)
        Usqr = multiply(self.U, self.U)
        d = (Vsqr[outer_row_coord] * newevals.T * Usqr[outer_col_coord].T)[0, 0]
        dY_out = d * Y_out
        
        Vox = multiply(self.V, self.V[outer_row_coord])
        Uoy = multiply(self.U, self.U[outer_col_coord])
        
        cache = Vsqr * newevals.T * Usqr.T
        crosscache = Vox * newevals.T * Uoy.T
        
        loopred = mat(zeros((self.V.shape[0], self.U.shape[0])))
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
        invdetGshift = divide(1., ((1. - a) * (1. - d) - multiply(bc, bc)))
        invGshift_1 = invdetGshift * (1. - d)
        invGshift_2 = multiply(invdetGshift, bc)
        Y = self.Y
        temp1 = P - (multiply(a, Y) + bc * Y_out)
        temp2 = P_out - (multiply(bc, Y) + dY_out)
        loopred = multiply(invGshift_1, temp1) + multiply(invGshift_2, temp2)
        return loopred
    
    
    def nested_imputationLOO_BU(self, outer_row_coord, outer_col_coord):
        P = self.K1.T * self.A * self.K2
        P_out = P[outer_row_coord, outer_col_coord]
        Y_out = self.Y[outer_row_coord, outer_col_coord]
        
        newevals = multiply(self.S * self.L.T, 1. / (self.S * self.L.T + self.regparam))
        Vsqr = multiply(self.V, self.V)
        Usqr = multiply(self.U, self.U)
        d = (Vsqr[outer_row_coord] * newevals.T * Usqr[outer_col_coord].T)[0, 0]
        dY_out = d * Y_out
        
        Vox = multiply(self.V, self.V[outer_row_coord])
        Uoy = multiply(self.U, self.U[outer_col_coord])
        
        cache = Vsqr * newevals.T * Usqr.T
        crosscache = Vox * newevals.T * Uoy.T
        
        loopred = mat(zeros((self.V.shape[0], self.U.shape[0])))
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
        return loopred
    
    
    def prepareLooCaches(self):
        
        #Hirvee hakkerointi
        if not hasattr(self, "Vsqr"):
            self.Vsqr = multiply(self.V, self.V)
            self.Usqr = multiply(self.U, self.U)
        self.newlooevals = multiply(self.S * self.L.T, 1. / (self.S * self.L.T + self.regparam))
        self.P = self.K1.T * self.A * self.K2
        self.newlooevalsUsqr = self.newlooevals.T * self.Usqr.T
        self.Vsqrnewlooevals = self.Vsqr * self.newlooevals.T
        self.Vcache = self.Vsqr * self.newlooevalsUsqr
        self.Ucache = self.Vsqrnewlooevals * self.Usqr.T
        self.diagGcache = self.Vsqr * self.newlooevals.T * self.Usqr.T
    
    
    def nested_imputationLooApproximation(self, outer_row_coord, outer_col_coord):
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
        Vcrosscache = self.V * multiply(self.V[outer_row_coord].T, self.newlooevalsUsqr[:, outer_col_coord])
        
        #loopred = mat(zeros((self.V.shape[0], self.U.shape[0])))
        #print self.U.shape[0], self.V.shape[0], self.Y.shape, loopred.shape, P.shape
        
        VcrosscacheSqr = multiply(Vcrosscache, Vcrosscache)
        
        a = self.Vcache[:, outer_col_coord]
        #bc = (crosscache * Uoy[j].T)[0, 0]
        bc = Vcrosscache
        #G = mat([[a, bc], [bc, d]])
        #invG = 1. / (a * d - bc * bc) * mat([[d, -bc], [-bc, a]])
        #invGshift = 1. / ((1. - a) * (1. - d) - bc * bc) * mat([[(1. - d), bc], [bc, (1. - a)]])
        invdetGshift = 1. / ((1. - a) * one_minus_d - VcrosscacheSqr)
        invGshift_1 = invdetGshift * one_minus_d
        invGshift_2 = multiply(invdetGshift, bc)
        #YY = mat([self.Y[i, j], self.Y[outer_row_coord, outer_col_coord]]).T
        #PP = mat([P[i, j], P[outer_row_coord, outer_col_coord]]).T
        #loopred[i, j] = (invGshift * (PP - G * YY))[0, 0]
        Y_j = self.Y[:, outer_col_coord]#self.Y[i, outer_col_coord]
        temp1 = self.P[:, outer_col_coord] - (multiply(a, Y_j) + bc * Y_out)
        temp2 = P_out - (multiply(bc, Y_j) + dY_out)
        loocolumn = multiply(invGshift_1, temp1) + multiply(invGshift_2, temp2)
        #loopred[i, j] = (1. / (1. - ccc)) * (P[i, j] - ccc * self.Y[i, j])
        #loopred[i, j] = P[i, j]
        
        #Ucache = self.Vsqrnewlooevals[outer_row_coord] * self.Usqr.T
        Ucrosscache = multiply(self.Vsqrnewlooevals[outer_row_coord], self.U[outer_col_coord]) * self.U.T
        
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
        
        UcrosscacheSqr = multiply(Ucrosscache, Ucrosscache)
        
        #a = (cache * self.Usqr[j].T)[0, 0]
        a = self.Ucache[outer_row_coord]
        #bc = (crosscache * Uoy[j].T)[0, 0]
        bc = Ucrosscache
        #G = mat([[a, bc], [bc, d]])
        #invG = 1. / (a * d - bc * bc) * mat([[d, -bc], [-bc, a]])
        #invGshift = 1. / ((1. - a) * (1. - d) - bc * bc) * mat([[(1. - d), bc], [bc, (1. - a)]])
        invdetGshift = 1. / ((1. - a) * one_minus_d - UcrosscacheSqr)
        invGshift_1 = invdetGshift * one_minus_d
        invGshift_2 = multiply(invdetGshift, bc)
        #YY = mat([self.Y[i, j], self.Y[outer_row_coord, outer_col_coord]]).T
        #PP = mat([P[i, j], P[outer_row_coord, outer_col_coord]]).T
        #loopred[i, j] = (invGshift * (PP - G * YY))[0, 0]
        Yi = self.Y[outer_row_coord]#self.Y[outer_row_coord, j]
        temp1 = self.P[outer_row_coord] - (multiply(a, Yi) + multiply(bc, Y_out))
        temp2 = P_out - (multiply(bc, Yi) + dY_out)
        loorow = multiply(invGshift_1, temp1) + multiply(invGshift_2, temp2)
        #loopred[i, j] = (1. / (1. - ccc)) * (P[i, j] - ccc * self.Y[i, j])
        #loopred[i, j] = P[i, j]
        
        loocolumn[outer_row_coord] = 0
        loorow[:, outer_col_coord] = 0
        return loocolumn, loorow
    
    
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


class PairwiseModelLinearHack(object):
    
    def __init__(self, W):
        """Initializes the linear model
        @param W: primal coefficient matrix
        @type W: numpy matrix"""
        self.W = W
    
    
    def predictWithDataMatrices(self, X1pred, X2pred):
        P = X1pred * self.W * X2pred.T
        return P

