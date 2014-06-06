
from numpy import *
from scipy import sparse as sp
import numpy as np
from abstract_learner import AbstractSupervisedLearner
from abstract_learner import AbstractIterativeLearner
from rlscore import model
from rlscore.utilities import array_tools

#import pyximport; pyximport.install()
import cython_greedy_rls

SELECTED_FEATURES = 'selected_features'
GREEDYRLS_LOO_PERFORMANCES = 'GreedyRLS_LOO_performances'
GREEDYRLS_TEST_PERFORMANCES = 'GreedyRLS_test_performances'

class GreedyRLS(AbstractSupervisedLearner, AbstractIterativeLearner):
    """Linear time greedy forward selection for RLS.
    
    Performs greedy forward selection, where at each step the feature selected
    is the one whose addition leads to lowest leave-one-out mean squared error.

    Parameters
    ----------
    train_features: {array-like, sparse matrix}, shape = [n_samples, n_features]
        Data matrix
    train_labels: {array-like}, shape = [n_samples] or [n_samples, n_labels] (if n_labels >1)
        Training set labels
    regparam: float (regparam > 0)
        regularization parameter
    subsetsize: int (0 < subsetsize <= n_labels)
        number of features to be selected
    bias: float, optional
        value of constant feature added to each data point (default 0)
 
    References
    ----------
    
    Greedy RLS is described in  [1]_.
           
    ..[1] Tapio Pahikkala, Antti Airola, and Tapio Salakoski.
    Speeding up Greedy Forward Selection for Regularized Least-Squares.
    Proceedings of The Ninth International Conference on Machine Learning and Applications,
    325-330, IEEE Computer Society, 2010.
    """
    
    def __init__(self, **kwargs):
        super(GreedyRLS, self).__init__(**kwargs)
        self.regparam = float(kwargs['regparam'])
        X = kwargs['train_features']
        if isinstance(X, sp.base.spmatrix):
            self.X = X.todense()
        else:
            self.X = X
        self.X = self.X.T
        self.Y = kwargs['train_labels']
        self.Y = array_tools.as_labelmatrix(self.Y)
        #Number of training examples
        self.size = self.Y.shape[0]
        #if not self.Y.shape[1] == 1:
        #    raise Exception('GreedyRLS currently supports only one output at a time. The output matrix is now of shape ' + str(self.Y.shape) + '.')
        if kwargs.has_key('bias'):
            self.bias = float(kwargs['bias'])
        else:
            self.bias = 0.
        if kwargs.has_key('measure'):
            self.measure = kwargs['measure']
        else:
            self.measure = None
        
        tsize = self.size
        fsize = X.shape[1]
        if not kwargs.has_key('subsetsize'):
            raise Exception("Parameter 'subsetsize' must be given.")
        self.desiredfcount = int(kwargs['subsetsize'])
        if not fsize >= self.desiredfcount:
            raise Exception('The overall number of features ' + str(fsize) + ' is smaller than the desired number ' + str(self.desiredfcount) + ' of features to be selected.')
        self.results = {}
    
    
    def train(self):
        """Trains the learning algorithm.
        
        After the learner is trained, one can call the method getModel
        to get the trained model
        """
        
        #The current version works only with the squared error measure
        self.measure = None
        
        if True:#self.Y.shape[1] > 1:
        #if False:#self.Y.shape[1] > 1:
            #self.solve_bu(regparam)
            self.solve_cython(self.regparam)
        else:
            #self.solve_new(regparam, float32)
            self.solve_new(self.regparam, float64)
            #self.solve_bu(regparam)
    
    
    def solve_cython(self, regparam):
        self.regparam = regparam
        X = self.X
        Y = self.Y
        
        if not hasattr(self, "bias"):
            self.bias = 0.
        bias_slice = sqrt(self.bias)*mat(ones((1,X.shape[1]),dtype=float64))
        
        '''su = sum(X, axis = 1)
        cc = 0
        indsmap = {}
        allinds = []
        for ci in range(X.shape[0]):
            if su[ci] == 0:
                pass
            else:
                allinds.append(ci)
                indsmap[ci] = cc
                cc += 1
        #print len(allinds)
        
        X = X[allinds]'''
        
        tsize = self.size
        fsize = X.shape[0]
        assert X.shape[1] == tsize
        #self.A = mat(zeros((fsize,1)))
        self.A = mat(zeros((fsize, Y.shape[1])))
        
        rp = regparam
        rpinv = 1. / rp
        
        
        
        #Biaz
        cv = sqrt(self.bias)*mat(ones((1, tsize)))
        ca = rpinv * (1. / (1. + cv * rpinv * cv.T)) * (cv * rpinv)
        
        
        self.dualvec = rpinv * Y - cv.T * rpinv * (1. / (1. + cv * rpinv * cv.T)) * (cv * rpinv * Y)
        
        XT = X.T
        GXT = rpinv * XT - cv.T * rpinv * (1. / (1. + cv * rpinv * cv.T)) * ((cv * rpinv) * XT)
        diagG = []
        for i in range(tsize):
            diagGi = rpinv - cv.T[i, 0] * ca[0, i]
            diagG.append(diagGi)
        diagG = array(diagG)
        
        listX = []
        for ci in range(fsize):
            listX.append(X[ci])
        
        self.selected = []
        
        currentfcount = 0
        self.performances = []
        selectedvec = zeros(fsize, dtype = int16)
        tempvec1, tempvec2, tempvec3 = zeros(tsize), zeros(Y.shape[1]), zeros((tsize, Y.shape[1]))
        while currentfcount < self.desiredfcount:
            
            if not self.measure == None:
                bestlooperf = None
            else:
                bestlooperf = 9999999999.
            
            #for ci in range(fsize):
            #print Y.dtype, X.dtype, GXT.dtype, diagG.dtype, self.dualvec.dtype
            self.looperf = ones(fsize) * float('Inf')
            #'''
            bestcind = cython_greedy_rls.find_optimal_feature(np.array(Y),
                                                              np.array(X),
                                                              np.array(GXT),
                                                              diagG,
                                                              np.array(self.dualvec),
                                                              self.looperf,
                                                              fsize,
                                                              tsize,
                                                              Y.shape[1],
                                                              selectedvec,
                                                              tempvec1,
                                                              tempvec2,
                                                              tempvec3)
            #foo
            '''
            diagG = mat(diagG).T
            for ci in allinds:
                ci_mapped = indsmap[ci]
                if ci in self.selected: continue
                cv = listX[ci_mapped]
                GXT_ci = GXT[:, ci_mapped]
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
                    print loodiff
                    foo
                    looperf_i = mean(multiply(loodiff, loodiff))
                    if looperf_i < bestlooperf:
                        bestcind = ci
                        bestlooperf = looperf_i
                self.looperf[ci] = looperf_i
            '''
            #'''
            self.bestlooperf = self.looperf[bestcind]#bestlooperf
            self.looperf = mat(self.looperf)
            self.performances.append(bestlooperf)
            ci_mapped = bestcind#indsmap[bestcind]
            cv = listX[ci_mapped]
            GXT_bci = GXT[:, ci_mapped]
            ca = GXT_bci * (1. / (1. + cv * GXT_bci))
            self.dualvec = self.dualvec - ca * (cv * self.dualvec)
            diagG = diagG - array(multiply(ca, GXT_bci)).reshape((self.size))
            GXT = GXT - ca * (cv * GXT)
            self.selected.append(bestcind)
            #print self.selected
            #print bestlooperf
            currentfcount += 1
            
            #Linear model with bias
            self.A[self.selected] = X[self.selected] * self.dualvec
            self.b = bias_slice * self.dualvec# * sqrt(self.bias)
            
            self.callback()
        self.finished()
        self.A[self.selected] = X[self.selected] * self.dualvec
        self.b = bias_slice * self.dualvec# * sqrt(self.bias)
        self.results[SELECTED_FEATURES] = self.selected
        self.results[GREEDYRLS_LOO_PERFORMANCES] = self.performances
    
    
    def solve_new(self, regparam, floattype):
        
        self.regparam = regparam
        X = self.X
        Y = mat(self.Y, dtype=floattype)
        
        if not hasattr(self, "bias"):
            self.bias = 0.
        bias_slice = sqrt(self.bias)*mat(ones((1,X.shape[1]), dtype=floattype))
        
        tsize = self.size
        fsize = X.shape[0]
        assert X.shape[1] == tsize
        self.A = mat(zeros((fsize,1),dtype=floattype))
        
        rp = regparam
        rpinv = 1. / rp
        
        
        #Biaz
        cv = sqrt(self.bias)*mat(ones((1, tsize), dtype=floattype))
        ca = mat(rpinv * (1. / (1. + cv * rpinv * cv.T)) * (cv * rpinv), dtype=floattype)
        
        
        self.dualvec = rpinv * Y - cv.T * rpinv * (1. / (1. + cv * rpinv * cv.T)) * (cv * rpinv * Y)
        
        GXT = cv.T * mat((rpinv * (1. / (1. + cv * rpinv * cv.T)) * (cv * rpinv)) * X.T, dtype=floattype)
        tempmatrix = mat(zeros(X.T.shape, dtype=floattype))
        multiply(X.T, rpinv, tempmatrix)
        #tempmatrix = rpinv * X.T
        subtract(tempmatrix, GXT, GXT)
        
        diagG = []
        for i in range(tsize):
            diagGi = rpinv - cv.T[i, 0] * ca[0, i]
            diagG.append(diagGi)
        diagG = mat(diagG, dtype=floattype).T
        
        self.selected = []
        self.performances = []
        currentfcount = 0
        
        temp2 = mat(zeros(tempmatrix.shape, dtype=floattype))
        
        while currentfcount < self.desiredfcount:
            
            multiply(X.T, GXT, tempmatrix)
            XGXTdiag = sum(tempmatrix, axis = 0)
            
            XGXTdiag = 1. / (1. + XGXTdiag)
            multiply(GXT, XGXTdiag, tempmatrix)
            
            tempvec1 = multiply((X * self.dualvec).T, XGXTdiag)
            multiply(GXT, tempvec1, temp2)
            subtract(self.dualvec, temp2, temp2)
            
            multiply(tempmatrix, GXT, tempmatrix)
            subtract(diagG, tempmatrix, tempmatrix)
            divide(1, tempmatrix, tempmatrix)
            multiply(tempmatrix, temp2, tempmatrix)
            
            
            if not self.measure == None:
                subtract(Y, tempmatrix, tempmatrix)
                multiply(temp2, 0, temp2)
                add(temp2, Y, temp2)
                looperf = self.measure.multiTaskPerformance(temp2, tempmatrix)
                looperf = mat(looperf, dtype=floattype)
                if self.measure.isErrorMeasure():
                    looperf[0, self.selected] = float('inf')
                    bestcind = argmin(looperf)
                    self.bestlooperf = amin(looperf)
                else:
                    looperf[0, self.selected] = - float('inf')
                    bestcind = argmax(looperf)
                    self.bestlooperf = amax(looperf)
            else:
                multiply(tempmatrix, tempmatrix, temp2)
                looperf = sum(temp2, axis = 0)
                looperf[0, self.selected] = float('inf')
                bestcind = argmin(looperf)
                self.bestlooperf = amin(looperf)
                self.loo_predictions = Y - tempmatrix[:, bestcind]
            
            self.looperf = looperf   #Needed in test_GreedyRLS module
            
            self.performances.append(self.bestlooperf)
            cv = X[bestcind]
            GXT_bci = GXT[:, bestcind]
            ca = GXT_bci * (1. / (1. + cv * GXT_bci))
            self.dualvec = self.dualvec - ca * (cv * self.dualvec)
            diagG = diagG - multiply(ca, GXT_bci)
            multiply(tempmatrix, 0, tempmatrix)
            add(tempmatrix, ca, tempmatrix)
            tempvec1 = cv * GXT
            multiply(tempmatrix, tempvec1, tempmatrix)
            subtract(GXT, tempmatrix, GXT)
            self.selected.append(bestcind)
            currentfcount += 1
            
            #Linear model with bias
            self.A[self.selected] = X[self.selected] * self.dualvec
            self.b = bias_slice * self.dualvec# * sqrt(self.bias)
            
            self.callback()
        self.finished()
        self.A[self.selected] = X[self.selected] * self.dualvec
        self.b = bias_slice * self.dualvec# * sqrt(self.bias)
        self.results[SELECTED_FEATURES] = self.selected
        self.results[GREEDYRLS_LOO_PERFORMANCES] = self.performances
        self.results['model'] = self.getModel()
    
    
    def getModel(self):
        """Returns the trained model, call this only after training.
        
        Returns
        -------
        model : LinearModel
            prediction function (model.W contains at most "subsetsize" number of non-zero coefficients)
        """
        return model.LinearModel(self.A, self.b)
    
    
    def solve_bu(self, regparam):
        self.regparam = regparam
        X = self.X
        Y = self.Y
        
        if not hasattr(self, "bias"):
            self.bias = 0.
        bias_slice = sqrt(self.bias)*mat(ones((1,X.shape[1]),dtype=float64))
        
        su = sum(X, axis = 1)
        cc = 0
        indsmap = {}
        allinds = []
        for ci in range(X.shape[0]):
            if su[ci] == 0:
                pass
            else:
                allinds.append(ci)
                indsmap[ci] = cc
                cc += 1
        #print len(allinds)
        
        X = X[allinds]
        
        tsize = self.size
        fsize = X.shape[0]
        assert X.shape[1] == tsize
        #self.A = mat(zeros((fsize,1)))
        self.A = mat(zeros((fsize, Y.shape[1])))
        
        rp = regparam
        rpinv = 1. / rp
        
        #Biaz
        cv = sqrt(self.bias)*mat(ones((1, tsize)))
        ca = rpinv * (1. / (1. + cv * rpinv * cv.T)) * (cv * rpinv)
        
        
        self.dualvec = rpinv * Y - cv.T * rpinv * (1. / (1. + cv * rpinv * cv.T)) * (cv * rpinv * Y)
        
        XT = X.T
        GXT = rpinv * XT - cv.T * rpinv * (1. / (1. + cv * rpinv * cv.T)) * ((cv * rpinv) * XT)
        diagG = []
        for i in range(tsize):
            diagGi = rpinv - cv.T[i, 0] * ca[0, i]
            diagG.append(diagGi)
        diagG = mat(diagG).T
        
        listX = []
        for ci in range(fsize):
            listX.append(X[ci])
        
        self.selected = []
        
        currentfcount = 0
        self.performances = []
        while currentfcount < self.desiredfcount:
            
            if not self.measure == None:
                bestlooperf = None
            else:
                bestlooperf = 9999999999.
            
            self.looperf = []
            #for ci in range(fsize):
            for ci in allinds:
                ci_mapped = indsmap[ci]
                if ci in self.selected: continue
                cv = listX[ci_mapped]
                GXT_ci = GXT[:, ci_mapped]
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
                    looperf_i = mean(multiply(loodiff, loodiff))
                    if looperf_i < bestlooperf:
                        bestcind = ci
                        bestlooperf = looperf_i
                self.looperf.append(looperf_i)
            self.looperf = mat(self.looperf)
            self.bestlooperf = bestlooperf
            self.performances.append(bestlooperf)
            ci_mapped = indsmap[bestcind]
            cv = listX[ci_mapped]
            GXT_bci = GXT[:, ci_mapped]
            ca = GXT_bci * (1. / (1. + cv * GXT_bci))
            self.dualvec = self.dualvec - ca * (cv * self.dualvec)
            diagG = diagG - multiply(ca, GXT_bci)
            GXT = GXT - ca * (cv * GXT)
            self.selected.append(bestcind)
            #print self.selected
            #print bestlooperf
            currentfcount += 1
            
            #Linear model with bias
            self.A[self.selected] = X[self.selected] * self.dualvec
            self.b = bias_slice * self.dualvec# * sqrt(self.bias)
            
            self.callback()
        self.finished()
        self.A[self.selected] = X[self.selected] * self.dualvec
        self.b = bias_slice * self.dualvec# * sqrt(self.bias)
        self.results[SELECTED_FEATURES] = self.selected
        self.results[GREEDYRLS_LOO_PERFORMANCES] = self.performances
#            self.callback()
#        self.finished()
#        bias_slice = sqrt(self.bias)*mat(ones((1,X.shape[1]),dtype=float64))
#        X_biased = vstack([X,bias_slice])
#        selected_plus_bias = self.selected+[fsize]
#        #self.A = mat(eye(fsize+1))[:,selected_plus_bias]*(X_biased[selected_plus_bias]*self.dualvec)
#        self.results[SELECTED_FEATURES] = self.selected
#        self.results[GREEDYRLS_LOO_PERFORMANCES] = self.performances
