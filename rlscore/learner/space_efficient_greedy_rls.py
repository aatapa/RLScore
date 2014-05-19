
import random as pyrandom

from numpy import *
import numpy.linalg as la
import scipy.sparse as sp

from rlscore.learner.abstract_learner import AbstractSupervisedLearner
from rlscore.learner.abstract_learner import AbstractIterativeLearner
from rlscore import data_sources
from rlscore import model

class SpaceEfficientGreedyRLS(AbstractSupervisedLearner, AbstractIterativeLearner):
    
    def loadResources(self):
        """
        Loads the resources from the previously set resource pool.
        
        @raise Exception: when some of the resources required by the learner is not available in the ResourcePool object.
        """
        AbstractIterativeLearner.loadResources(self)
        X = self.resource_pool[data_sources.TRAIN_FEATURES]
        if isinstance(X, sp.base.spmatrix):
            self.X = X.todense()
        else:
            self.X = X
        self.X = self.X.T
        self.Y = self.resource_pool[data_sources.TRAIN_LABELS]
        #Number of training examples
        self.size = self.Y.shape[0]
        if self.resource_pool.has_key('bias'):
            self.bias = float(self.resource_pool['bias'])
        else:
            self.bias = 0.
        if self.resource_pool.has_key(data_sources.PERFORMANCE_MEASURE):
            self.measure = None
        #    self.measure = self.resource_pool[data_sources.PERFORMANCE_MEASURE]
        else:
            self.measure = None
        self.results = {}
    
    
    def train(self):
        regparam = float(self.resource_pool[data_sources.TIKHONOV_REGULARIZATION_PARAMETER])
        self.regparam = regparam
        
        ##The current version works only with the squared error measure
        #self.measure = None
        #self.solve_weak(regparam)
        #return
        #if not self.Y.shape[1] == 1:
        self.solve_bu(regparam)
        #else:
        #    self.solve_tradeoff(regparam)
    
    
    def getModel(self):
        return model.LinearModel(self.A, self.b)
    
    
    def solve_bu(self, regparam):
        """Trains RLS with the given value of the regularization parameter
        
        @param regparam: value of the regularization parameter
        @type regparam: float
        """
        
        self.regparam = regparam
        X = self.X
        Y = self.Y
        
        if not hasattr(self, "bias"):
            self.bias = 0.
        
        tsize = self.size
        fsize = X.shape[0]
        assert X.shape[1] == tsize
        self.A = mat(zeros((fsize, Y.shape[1])))
        
        rp = regparam
        rpinv = 1. / rp
        
        if not self.resource_pool.has_key('subsetsize'):
            raise Exception("Parameter 'subsetsize' must be given.")
        desiredfcount = int(self.resource_pool['subsetsize'])
        if not fsize >= desiredfcount:
            raise Exception('The overall number of features ' + str(fsize) + ' is smaller than the desired number ' + str(desiredfcount) + ' of features to be selected.')
        
        #Biaz
        bias_slice = sqrt(self.bias)*mat(ones((1,X.shape[1]),dtype=float64))
        cv = bias_slice
        ca = rpinv * (1. / (1. + cv * rpinv * cv.T)) * (cv * rpinv)
        
        self.dualvec = rpinv * Y - cv.T * rpinv * (1. / (1. + cv * rpinv * cv.T)) * (cv * rpinv * Y)
        
        diagG = []
        for i in range(tsize):
            diagGi = rpinv - cv.T[i, 0] * ca[0, i]
            diagG.append(diagGi)
        diagG = mat(diagG).T
        
        #listX = []
        #for ci in range(fsize):
        #    listX.append(X[ci])
        
        U, S, VT = la.svd(cv, full_matrices = False)
        U, S, VT = mat(U), mat(S), mat(VT)
        Omega = 1. / (S * S + rp) - rpinv
        
        self.selected = []
        
        currentfcount = 0
        self.performances = []
        while currentfcount < desiredfcount:
            
            if not self.measure == None:
                bestlooperf = None
            else:
                bestlooperf = float('inf')
            
            self.looperf = []
            for ci in range(fsize):
                if ci in self.selected: continue
                #cv = listX[ci]
                cv = X[ci]
                GXT_ci = VT.T * multiply(Omega.T, (VT * cv.T)) + rpinv * cv.T #GXT[:, ci]
                ca = GXT_ci * (1. / (1. + cv * GXT_ci))
                updA = self.dualvec - ca * (cv * self.dualvec)
                invupddiagG = 1. / (diagG - multiply(ca, GXT_ci))
                
                if not self.measure == None:
                    loopred = Y - multiply(invupddiagG, updA)
                    looperf_i = self.measure.multiOutputPerformance(Y, loopred)
                    if bestlooperf == None:
                        bestlooperf = looperf_i
                        bestcind = ci
                    if self.measure.comparePerformances(looperf_i, bestlooperf) > 0:
                        bestcind = ci
                        bestlooperf = looperf_i
                else:
                    #This default squared performance is a bit faster to compute than the one loaded separately.
                    loodiff = multiply(invupddiagG, updA)
                    #looperf_i = (loodiff.T * loodiff)[0, 0]
                    looperf_i = mean(sum(multiply(loodiff, loodiff), axis = 0))
                    if looperf_i < bestlooperf:
                        bestcind = ci
                        bestlooperf = looperf_i
                self.looperf.append(looperf_i)
            self.looperf = mat(self.looperf)
            
            self.bestlooperf = bestlooperf
            self.performances.append(bestlooperf)
            #cv = listX[bestcind]
            cv = X[bestcind]
            #GXT_bci = GXT[:, bestcind]
            GXT_bci = VT.T * multiply(Omega.T, (VT * cv.T)) + rpinv * cv.T
            ca = GXT_bci * (1. / (1. + cv * GXT_bci))
            self.dualvec = self.dualvec - ca * (cv * self.dualvec)
            diagG = diagG - multiply(ca, GXT_bci)
            #GXT = GXT - ca * (cv * GXT)
            self.selected.append(bestcind)
            X_sel = X[self.selected]
            if isinstance(X_sel, sp.base.spmatrix):
                X_sel = X_sel.todense()
            U, S, VT = la.svd(vstack([X_sel, bias_slice]), full_matrices = False)
            U, S, VT = mat(U), mat(S), mat(VT)
            Omega = 1. / (multiply(S, S) + rp) - rpinv
            #print self.selected
            #print self.performances
            currentfcount += 1
            
            #Linear model with bias
            self.A[self.selected] = X[self.selected] * self.dualvec
            self.b = bias_slice * self.dualvec
            
            self.callback()
            #print who(locals())
        self.finished()
        self.A[self.selected] = X[self.selected] * self.dualvec
        self.b = bias_slice * self.dualvec
        self.results[data_sources.SELECTED_FEATURES] = self.selected
        self.results[data_sources.GREEDYRLS_LOO_PERFORMANCES] = self.performances
        self.results[data_sources.MODEL] = self.getModel()
    
    
    def solve_tradeoff(self, regparam):
        """Trains RLS with the given value of the regularization parameter
        
        @param regparam: value of the regularization parameter
        @type regparam: float
        """
        
        self.regparam = regparam
        X = self.X
        Y = self.Y
        
        if not hasattr(self, "bias"):
            self.bias = 0.
        
        tsize = self.size
        fsize = X.shape[0]
        assert X.shape[1] == tsize
        self.A = mat(zeros((fsize, Y.shape[1])))
        
        rp = regparam
        rpinv = 1. / rp
        
        if not self.resource_pool.has_key('subsetsize'):
            raise Exception("Parameter 'subsetsize' must be given.")
        desiredfcount = int(self.resource_pool['subsetsize'])
        if not fsize >= desiredfcount:
            raise Exception('The overall number of features ' + str(fsize) + ' is smaller than the desired number ' + str(desiredfcount) + ' of features to be selected.')
        
        #Biaz
        bias_slice = sqrt(self.bias)*mat(ones((1,X.shape[1]),dtype=float64))
        cv = bias_slice
        ca = rpinv * (1. / (1. + cv * rpinv * cv.T)) * (cv * rpinv)
        
        self.dualvec = rpinv * Y - cv.T * rpinv * (1. / (1. + cv * rpinv * cv.T)) * (cv * rpinv * Y)
        
        diagG = []
        for i in range(tsize):
            diagGi = rpinv - cv.T[i, 0] * ca[0, i]
            diagG.append(diagGi)
        diagG = mat(diagG).T
        
        #listX = []
        #for ci in range(fsize):
        #    listX.append(X[ci])
        
        U, S, VT = la.svd(cv, full_matrices = False)
        U, S, VT = mat(U), mat(S), mat(VT)
        Omega = 1. / (S * S + rp) - rpinv
        
        self.selected = []
        
        blocksize = 1000
        blocks = []
        blockcount = 0
        while True:
            startind = blockcount * blocksize
            if (blockcount + 1) * blocksize < fsize:
                print blockcount, fsize, (blockcount + 1) * blocksize
                endind = (blockcount + 1) * blocksize
                blocks.append(range(startind, endind))
                blockcount += 1
            else:
                blocks.append(range(startind, fsize))
                blockcount += 1
                break
        
        
        currentfcount = 0
        self.performances = []
        while currentfcount < desiredfcount:
            
            if not self.measure == None:
                self.bestlooperf = None
            else:
                self.bestlooperf = float('inf')
            
            
            looperf = mat(zeros((1, fsize)))
            
            for blockind in range(blockcount):
                
                block = blocks[blockind]
                
                tempmatrix = mat(zeros((tsize, len(block))))
                temp2 = mat(zeros((tsize, len(block))))
                
                X_block = X[block]
                GXT_block = VT.T * multiply(Omega.T, (VT * X_block.T)) + rpinv * X_block.T
                
                multiply(X_block.T, GXT_block, tempmatrix)
                XGXTdiag = sum(tempmatrix, axis = 0)
                
                XGXTdiag = 1. / (1. + XGXTdiag)
                multiply(GXT_block, XGXTdiag, tempmatrix)
                
                tempvec1 = multiply((X_block * self.dualvec).T, XGXTdiag)
                multiply(GXT_block, tempvec1, temp2)
                subtract(self.dualvec, temp2, temp2)
                
                multiply(tempmatrix, GXT_block, tempmatrix)
                subtract(diagG, tempmatrix, tempmatrix)
                divide(1, tempmatrix, tempmatrix)
                multiply(tempmatrix, temp2, tempmatrix)
                
                
                if not self.measure == None:
                    subtract(Y, tempmatrix, tempmatrix)
                    multiply(temp2, 0, temp2)
                    add(temp2, Y, temp2)
                    looperf_block = self.measure.multiTaskPerformance(temp2, tempmatrix)
                    looperf_block = mat(looperf_block)
                else:
                    multiply(tempmatrix, tempmatrix, tempmatrix)
                    looperf_block = sum(tempmatrix, axis = 0)
                looperf[:, block] = looperf_block
                
            if not self.measure == None:
                if self.measure.isErrorMeasure():
                    looperf[0, self.selected] = float('inf')
                    bestcind = argmin(looperf)
                    self.bestlooperf = amin(looperf)
                else:
                    looperf[0, self.selected] = - float('inf')
                    bestcind = argmax(looperf)
                    self.bestlooperf = amax(looperf)
            else:
                looperf[0, self.selected] = float('inf')
                bestcind = argmin(looperf)
                self.bestlooperf = amin(looperf)
                
            self.looperf = looperf
            
            self.performances.append(self.bestlooperf)
            #cv = listX[bestcind]
            cv = X[bestcind]
            #GXT_bci = GXT[:, bestcind]
            GXT_bci = VT.T * multiply(Omega.T, (VT * cv.T)) + rpinv * cv.T
            ca = GXT_bci * (1. / (1. + cv * GXT_bci))
            self.dualvec = self.dualvec - ca * (cv * self.dualvec)
            diagG = diagG - multiply(ca, GXT_bci)
            #GXT = GXT - ca * (cv * GXT)
            self.selected.append(bestcind)
            X_sel = X[self.selected]
            if isinstance(X_sel, sp.base.spmatrix):
                X_sel = X_sel.todense()
            U, S, VT = la.svd(vstack([X_sel, bias_slice]), full_matrices = False)
            U, S, VT = mat(U), mat(S), mat(VT)
            #print U.shape, S.shape, VT.shape
            Omega = 1. / (multiply(S, S) + rp) - rpinv
            #print self.selected
            #print self.performances
            currentfcount += 1
            
            #Linear model with bias
            self.A[self.selected] = X[self.selected] * self.dualvec
            self.b = bias_slice * self.dualvec
            
            self.callback()
            #print who(locals())
        self.finished()
        self.A[self.selected] = X[self.selected] * self.dualvec
        self.b = bias_slice * self.dualvec
        self.results[data_sources.SELECTED_FEATURES] = self.selected
        self.results[data_sources.GREEDYRLS_LOO_PERFORMANCES] = self.performances
        self.results[data_sources.MODEL] = self.getModel()
    
    
    def solve_weak(self, regparam):
        
        self.regparam = regparam
        X = self.X
        Y = self.Y
        
        if not hasattr(self, "bias"):
            self.bias = 0.
        
        tsize = self.size
        fsize = X.shape[0]
        assert X.shape[1] == tsize
        self.A = mat(zeros((fsize, Y.shape[1])))
        
        rp = regparam
        rpinv = 1. / rp
        
        if not self.resource_pool.has_key('subsetsize'):
            raise Exception("Parameter 'subsetsize' must be given.")
        desiredfcount = int(self.resource_pool['subsetsize'])
        if not fsize >= desiredfcount:
            raise Exception('The overall number of features ' + str(fsize) + ' is smaller than the desired number ' + str(desiredfcount) + ' of features to be selected.')
        
        #Biaz
        bias_slice = sqrt(self.bias)*mat(ones((1,X.shape[1]),dtype=float64))
        cv = bias_slice
        ca = rpinv * (1. / (1. + cv * rpinv * cv.T)) * (cv * rpinv)
        
        self.dualvec = rpinv * Y - cv.T * rpinv * (1. / (1. + cv * rpinv * cv.T)) * (cv * rpinv * Y)
        self.F = cv.T * (cv * self.dualvec)
        
        diagG = []
        for i in range(tsize):
            diagGi = rpinv - cv.T[i, 0] * ca[0, i]
            diagG.append(diagGi)
        diagG = mat(diagG).T
        
        #listX = []
        #for ci in range(fsize):
        #    listX.append(X[ci])
        
        U, S, VT = la.svd(cv, full_matrices = False)
        U, S, VT = mat(U), mat(S), mat(VT)
        Omega = 1. / (S * S + rp) - rpinv
        
        self.selected = []
        notselected = set(range(fsize))
        currentfcount = 0
        self.performances = []
        while currentfcount < desiredfcount:
            
            if not self.measure == None:
                bestlooperf = None
            else:
                bestlooperf = float('inf')
            
            X_s = X[self.selected]
            
            self.looperf = []
            sample_60 = pyrandom.sample(notselected, len(notselected))
            sample_60 = sorted(sample_60)
            print sample_60
            #sample_60 = pyrandom.sample(notselected, 1)
            for ci in sample_60:
                cv = X[ci]
                GXT_ci = VT.T * multiply(Omega.T, (VT * cv.T)) + rpinv * cv.T #GXT[:, ci]
                ca = GXT_ci * (1. / (1. + cv * GXT_ci))
                updA = self.dualvec - ca * (cv * self.dualvec)
                #updF = self.F - X_s.T * (X_s * (ca * (cv * self.dualvec))) + cv.T * (cv * updA)
                updF = bias_slice.T * (bias_slice * updA) + X_s.T * (X_s * updA) + cv.T * (cv * updA) #PREFITTING (SLOW)
                invupddiagG = 1. / (diagG - multiply(ca, GXT_ci))
                
                if not self.measure == None:
                    loopred = Y - multiply(invupddiagG, updA)
                    looperf_i = self.measure.multiOutputPerformance(Y, loopred)
                    if bestlooperf == None:
                        bestlooperf = looperf_i
                        bestcind = ci
                    if self.measure.comparePerformances(looperf_i, bestlooperf) > 0:
                        bestcind = ci
                        bestlooperf = looperf_i
                else:
                    #This default squared performance is a bit faster to compute than the one loaded separately.
                    updtrainingerr = updF - self.Y
                    updtrainingerr = mean(sum(multiply(updtrainingerr, updtrainingerr), axis = 0))
                    looperf_i = updtrainingerr
                    #loodiff = multiply(invupddiagG, updA)
                    #looperf_i = mean(sum(multiply(loodiff, loodiff), axis = 0))
                    if looperf_i < bestlooperf:
                        bestcind = ci
                        bestlooperf = looperf_i
                self.looperf.append(looperf_i)
            self.looperf = mat(self.looperf)
            
            self.bestlooperf = bestlooperf
            print bestlooperf
            self.performances.append(bestlooperf)
            #cv = listX[bestcind]
            cv = X[bestcind]
            #GXT_bci = GXT[:, bestcind]
            GXT_bci = VT.T * multiply(Omega.T, (VT * cv.T)) + rpinv * cv.T
            ca = GXT_bci * (1. / (1. + cv * GXT_bci))
            self.dualvec = self.dualvec - ca * (cv * self.dualvec)
            diagG = diagG - multiply(ca, GXT_bci)
            #self.F = self.F + cv.T * (cv * self.dualvec)
            self.F = X_s.T * (X_s * self.dualvec) + cv.T * (cv * self.dualvec)
            #GXT = GXT - ca * (cv * GXT)
            self.selected.append(bestcind)
            notselected.remove(bestcind)
            X_sel = X[self.selected]
            if isinstance(X_sel, sp.base.spmatrix):
                X_sel = X_sel.todense()
            U, S, VT = la.svd(vstack([X_sel, bias_slice]), full_matrices = False)
            U, S, VT = mat(U), mat(S), mat(VT)
            Omega = 1. / (multiply(S, S) + rp) - rpinv
            currentfcount += 1
            
            #Linear model with bias
            self.A[self.selected] = X[self.selected] * self.dualvec
            self.b = bias_slice * self.dualvec
            
            self.callback()
        self.finished()
        self.A[self.selected] = X[self.selected] * self.dualvec
        self.b = bias_slice * self.dualvec
        self.results[data_sources.SELECTED_FEATURES] = self.selected
        self.results[data_sources.GREEDYRLS_LOO_PERFORMANCES] = self.performances
        self.results[data_sources.MODEL] = self.getModel()


