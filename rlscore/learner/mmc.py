#
# The MIT License (MIT)
#
# This file is part of RLScore 
#
# Copyright (c) 2012 - 2016 Tapio Pahikkala, Antti Airola
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import random
import numpy as np
from rlscore.utilities import array_tools
from rlscore.utilities import adapter
from rlscore.predictor import PredictorInterface


class MMC(PredictorInterface):
    """RLS-based maximum-margin clustering.
    
    Performs stochastic search, that aims to find a labeling of the data such
    that would minimize the RLS-objective function.

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape = [n_samples, n_features]
        Data matrix
        
    regparam : float, optional
        regularization parameter, regparam > 0 (default=1.0)
        
    number_of_clusters : int, optional
        number of clusters (default = 2)
        
    kernel : {'LinearKernel', 'GaussianKernel', 'PolynomialKernel', 'PrecomputedKernel', ...}
        kernel function name, imported dynamically from rlscore.kernel
        
    basis_vectors : {array-like, sparse matrix}, shape = [n_bvectors, n_features], optional
        basis vectors (typically a randomly chosen subset of the training data)
        
    Y : {array-like}, shape = [n_samples] or [n_samples, n_clusters], optional
        Initial clustering (binary or one-versus-all encoding)
        
    fixed_indixes : list of indices, optional
        Instances whose clusters are prefixed (i.e. not allowed to change)
    
    callback : callback function, optional
        called after each pass through data
        
        
    Other Parameters
    ----------------
    bias : float, optional
        LinearKernel: the model is w*x + bias*w0, (default=1.0)
        
    gamma : float, optional
        GaussianKernel: k(xi,xj) = e^(-gamma*<xi-xj,xi-xj>) (default=1.0)
        PolynomialKernel: k(xi,xj) = (gamma * <xi, xj> + coef0)**degree (default=1.0)
               
    coef0 : float, optional
        PolynomialKernel: k(xi,xj) = (gamma * <xi, xj> + coef0)**degree (default=0.)
        
    degree : int, optional
        PolynomialKernel: k(xi,xj) = (gamma * <xi, xj> + coef0)**degree (default=2)
        
    Attributes
    -----------
    predictor : {LinearPredictor, KernelPredictor}
        trained predictor
    
    Notes
    -----
    
    The MMC algorithm is described in [1].    
    
    References
    ----------
    
    [1] Fabian Gieseke, Tapio Pahikkala,  and Oliver Kramer.
    Fast Evolutionary Maximum Margin Clustering.
    Proceedings of the 26th International Conference on Machine Learning,
    361-368, ACM, 2009.
    """
    
    def __init__(self, X, regparam=1.0, number_of_clusters=2, kernel='LinearKernel', basis_vectors=None, Y = None, fixed_indices=None, callback=None,  **kwargs):
        kwargs['X'] = X 
        kwargs['kernel'] = kernel
        if basis_vectors is not None:
            kwargs['basis_vectors'] = basis_vectors
        self.svdad = adapter.createSVDAdapter(**kwargs)
        self.svals = np.mat(self.svdad.svals)
        self.svecs = self.svdad.rsvecs
        self.regparam = regparam
        self.constraint = 0
        #if not kwargs.has_key('number_of_clusters'):
        #    raise Exception("Parameter 'number_of_clusters' must be given.")
        self.labelcount = number_of_clusters
        if self.labelcount == 2:
            self.oneclass = True
        else:
            self.oneclass = False
        self.callbackfun = callback
        if Y is not None:
            Y_orig = array_tools.as_array(Y)
            #if Y_orig.shape[1] == 1:
            if len(Y_orig.shape) == 1:
                self.Y = np.zeros((Y_orig.shape[0], 2))
                self.Y[:, 0] = Y_orig
                self.Y[:, 1] = - Y_orig
                self.oneclass = True
            else:
                self.Y = Y_orig.copy()
                self.oneclass = False
            for i in range(self.Y.shape[0]):
                largestind = 0
                largestval = self.Y[i, 0]
                for j in range(self.Y.shape[1]):
                    if self.Y[i, j] > largestval:
                        largestind = j
                        largestval = self.Y[i, j]
                    self.Y[i, j] = -1.
                self.Y[i, largestind] = 1.
        else:
            size = self.svecs.shape[0]
            ysize = self.labelcount
            if self.labelcount is None: self.labelcount = 2
            self.Y = RandomLabelSource(size, ysize).readLabels()
        self.size = self.Y.shape[0]
        self.labelcount = self.Y.shape[1]
        self.classvec = - np.ones((self.size), dtype = np.int32)
        self.classcounts = np.zeros((self.labelcount), dtype = np.int32)
        for i in range(self.size):
            clazzind = 0
            largestlabel = self.Y[i, 0]
            for j in range(self.labelcount):
                if self.Y[i, j] > largestlabel:
                    largestlabel = self.Y[i, j]
                    clazzind = j
            self.classvec[i] = clazzind
            self.classcounts[clazzind] = self.classcounts[clazzind] + 1
        
        self.svecs_list = []
        for i in range(self.size):
            self.svecs_list.append(self.svecs[i].T)
        self.fixedindices = []
        if fixed_indices is not None:
            self.fixedindices = fixed_indices
        else:
            self.fixedindices = []
        self.results = {}
        self.solve(self.regparam)
    
    
    def solve(self, regparam):
        """Trains the learning algorithm, using the given regularization parameter.
               
        Parameters
        ----------
        regparam : float (regparam > 0)
            regularization parameter
        """
        self.regparam = regparam
        
        #Some counters for bookkeeping
        self.stepcounter = 0
        self.flipcounter = 0
        self.nochangecounter = 0
              
        #Cached results
        self.evals = np.multiply(self.svals, self.svals)
        self.newevals = 1. / (self.evals + self.regparam)
        newevalslamtilde = np.multiply(self.evals, self.newevals)
        self.D = np.sqrt(newevalslamtilde)
        #self.D = -newevalslamtilde
        
        self.VTY = self.svecs.T * self.Y
        DVTY = np.multiply(self.D.T, self.svecs.T * self.Y)
        
        #Using lists in order to avoid unnecessary matrix slicings
        self.DVTY_list = []
        self.YTVDDVTY_list = []
        self.classFitnessList = []
        for i in range(self.labelcount):
            DVTY_i = DVTY[:,i]
            self.DVTY_list.append(DVTY_i)
            YTVDDVTY_i = DVTY_i.T * DVTY_i
            self.YTVDDVTY_list.append(YTVDDVTY_i)
            fitness_i = self.size - DVTY_i.T * DVTY_i
            self.classFitnessList.append(fitness_i)
        
        self.Dsvecs_list = []
        self.svecsDDsvecs_list = []
        for i in range(self.size):
            Dsvec = np.multiply(self.D.T, self.svecs[i].T)
            self.Dsvecs_list.append(Dsvec)
            self.svecsDDsvecs_list.append(Dsvec.T*Dsvec)
        
        self.updateA()
        
        
        converged = False
        print(self.classcounts.T)
        if self.callbackfun is not None:
            self.callbackfun.callback(self)
        while True:
            
            converged = self.roundRobin()
            print(self.classcounts.T)
            if self.callbackfun is not None:
                self.callbackfun.callback(self)
            if converged: break
        
        if self.oneclass:
            self.Y = self.Y[:, 0]
            self.A = self.A[:, 0]
        self.results['predicted_clusters_for_training_data'] = self.Y
        self.predictor = self.svdad.createModel(self)
    
    
    def computeGlobalFitness(self):
        fitness = 0.
        for classind in range(self.labelcount):
            fitness += self.classFitnessList[classind]
        return fitness
    
    
    def computeFlipFitnessForSingleClass(self, flipindex, classind):
        
        currentclassind = self.classvec[flipindex]
        coef = -2
        if currentclassind != classind:
            coef = 2
        
        
        DVTY_old = self.DVTY_list[classind]
        fitness_old = self.classFitnessList[classind]
        fitness_new = self.size - (self.YTVDDVTY_list[classind] + 2 * coef * DVTY_old.T * self.Dsvecs_list[flipindex] + 4 * self.svecsDDsvecs_list[flipindex])
        fitnessdiff = fitness_new - fitness_old
        return fitnessdiff

    
    def flipClass(self, flipindex, newclassind, DVTY_new_currentclass=None, DVTY_new_newclass=None):
        currentclassind = self.classvec[flipindex]
        self.Y[flipindex, currentclassind] = -1.
        self.Y[flipindex, newclassind] = 1.
        self.classcounts[currentclassind] -= 1
        self.classcounts[newclassind] += 1
        self.classvec[flipindex] = newclassind
        
        if DVTY_new_currentclass is None:
            DVTY_new_currentclass = self.DVTY_list[currentclassind] - 2 * self.Dsvecs_list[flipindex]
        if DVTY_new_newclass is None:
            DVTY_new_newclass = self.DVTY_list[newclassind] + 2 * self.Dsvecs_list[flipindex]
        
        self.DVTY_list[currentclassind] = DVTY_new_currentclass
        self.DVTY_list[newclassind] = DVTY_new_newclass
    
    
    def step(self, flipindex):
        """Perform one iteration step by checking whether flipping the label indexed by flipindex improves the current state. If it does, then the label is flipped. Otherwise, the the current label vector is left unchanged.
        flipindex:    the index of the label to be flipped.
        returns:      True, if the label is flipped and False otherwise."""
        y = self.Y[flipindex, 0]
        currentclassind = self.classvec[flipindex]
        
        DVTY_old_currentclass = self.DVTY_list[currentclassind]
        fitness_old_currentclass = self.classFitnessList[currentclassind]
        DVTY_new_currentclass = DVTY_old_currentclass - 2 * self.Dsvecs_list[flipindex]
        fitness_new_currentclass = self.size - DVTY_new_currentclass.T * DVTY_new_currentclass
        fitnessdiff_currentclass = fitness_new_currentclass - fitness_old_currentclass
        
        changed = False
        
        bestclassind = None
        VTY_new_bestclass = None
        bestfitnessdiff = float('Inf')
        fitness_new_bestclass = None
        
        for newclassind in range(self.labelcount):
            
            if newclassind == currentclassind: continue
            
            DVTY_old_newclass = self.DVTY_list[newclassind]
            DVTY_new_newclass = DVTY_old_newclass + 2 * self.Dsvecs_list[flipindex]
            fitness_old_newclass = self.classFitnessList[newclassind]
            fitness_new_newclass = self.size - DVTY_new_newclass.T * DVTY_new_newclass
            fitnessdiff_newclass = fitness_new_newclass - fitness_old_newclass
            
            fitnessdiff = fitnessdiff_currentclass + fitnessdiff_newclass
            
            if fitnessdiff < 0 and fitnessdiff < bestfitnessdiff:
                bestfitnessdiff = fitnessdiff[0, 0]
                bestclassind = newclassind
                DVTY_new_bestclass = DVTY_new_newclass
                fitness_new_bestclass = fitness_new_newclass
        
        
        if bestclassind is not None:
            if self.classcounts[currentclassind] > self.constraint:
                
                self.Y[flipindex, currentclassind] = -1.
                self.Y[flipindex, bestclassind] = 1.
                self.classcounts[currentclassind] -= 1
                self.classcounts[bestclassind] += 1
                self.classvec[flipindex] = bestclassind
                
                self.DVTY_list[currentclassind] = DVTY_new_currentclass
                self.DVTY_list[bestclassind] = DVTY_new_bestclass
                self.classFitnessList[currentclassind] = fitness_new_currentclass
                self.classFitnessList[bestclassind] = fitness_new_bestclass
                
                changed = True
        
        return changed
    
    
    def stepLOO(self, flipindex):
        """Perform one iteration step by checking whether flipping the label indexed by flipindex improves the current state. If it does, then the label is flipped. Otherwise, the the current label vector is left unchanged.
        flipindex :    the index of the label to be flipped.
        returns :      True, if the label is flipped and False otherwise."""
        y = self.Y[flipindex, 0]
        currentclassind = self.classvec[flipindex]
        
        DVTY_old_currentclass = self.DVTY_list[currentclassind]
        DVTY_new_currentclass = DVTY_old_currentclass - 2 * self.Dsvecs_list[flipindex]

            
        bevals = np.multiply(self.evals, self.newevals)
        RV = self.Dsvecs_list[flipindex]
        right = DVTY_old_currentclass - RV * (-1.)
        RQY = RV.T * np.multiply(bevals.T, right)
        RQRT = RV.T * np.multiply(bevals.T, RV)
        result = 1. / (1. - RQRT) * RQY
        fitnessdiff_currentclass = (-1. - result) ** 2 - (1. - result) ** 2
        
        changed = False
        
        bestclassind = None
        DVTY_new_bestclass = None
        bestfitnessdiff = 10.
        fitness_new_bestclass = None
        
        for newclassind in range(self.labelcount):
            
            if newclassind == currentclassind: continue
            
            DVTY_old_newclass = self.DVTY_list[newclassind]
            DVTY_new_newclass = DVTY_old_newclass + 2 * self.Dsvecs_list[flipindex]
            
            right = DVTY_old_newclass - RV * (-1.)
            RQY = RV.T * np.multiply(bevals.T, right)
            RQRT = RV.T * np.multiply(bevals.T, RV)
            result = 1. / (1. - RQRT) * RQY
            
            fitnessdiff_newclass = (1. - result) ** 2 - (-1. - result) ** 2
            
            fitnessdiff = fitnessdiff_currentclass + fitnessdiff_newclass
            
            if fitnessdiff < 0 and fitnessdiff < bestfitnessdiff:
                bestfitnessdiff = fitnessdiff
                bestclassind = newclassind
                DVTY_new_bestclass = DVTY_new_newclass
                #fitness_new_bestclass = fitness_new_newclass
        
        if bestclassind is not None:
            if self.classcounts[currentclassind] > self.constraint:
                
                self.Y[flipindex, currentclassind] = -1.
                self.Y[flipindex, bestclassind] = 1.
                self.classcounts[currentclassind] -= 1
                self.classcounts[bestclassind] += 1
                self.classvec[flipindex] = bestclassind
                
                self.DVTY_list[currentclassind] = DVTY_new_currentclass
                self.DVTY_list[bestclassind] = DVTY_new_bestclass

                
                changed = True
        
        return changed
    
    
    def updateA(self):
        self.A = self.svecs * np.multiply(self.newevals.T, self.VTY)

    
    
    def roundRobin(self, LOO = False):
        
        localstepcounter = 0
        
        converged = False
        
        flipindex = self.size - 1
        allinds = set(range(self.size)) - set(self.fixedindices)
        
        for flipindex in allinds:
            if LOO:
                flipped = self.stepLOO(flipindex)
            else:
                flipped = self.step(flipindex)
            self.stepcounter += 1
            localstepcounter += 1
            if flipped:
                self.flipcounter += 1
                self.nochangecounter = 0
            else:
                self.nochangecounter += 1
            if self.nochangecounter >= len(allinds):
                converged = True
                break
        
        self.updateA()
        return converged
        
    
    def giveAndTake(self, howmany):
        
        allinds = set(range(self.size))
        allinds = allinds - set(self.fixedindices)
        allinds = list(allinds)
        random.shuffle(allinds)
        
        for clazz in range(self.labelcount):
            givelist = []
            for flipindex in allinds:
                if self.classvec[flipindex] != clazz:
                    continue
                else:
                    bestfit = None
                    bestclass = None
                    ffit1 = self.computeFlipFitnessForSingleClass(flipindex, clazz)
                    for newclassind in range(self.labelcount):
                        if newclassind == clazz: continue
                        ffit2 = self.computeFlipFitnessForSingleClass(flipindex, newclassind)
                        flipfit = ffit1 + ffit2
                        if bestfit is None or bestfit > flipfit:
                            bestfit = flipfit
                            bestclass = newclassind
                    givelist.append((bestfit,bestclass,flipindex))
            givelist.sort()
            
            for i in range(min(howmany,len(givelist))):
                fit, newclassind, flipindex = givelist[i]
                self.flipClass(flipindex, newclassind)
        
            takelist = []
            for flipindex in allinds:
                if self.classvec[flipindex] != clazz:
                    ffit1 = self.computeFlipFitnessForSingleClass(flipindex, self.classvec[flipindex])
                    ffit2 = self.computeFlipFitnessForSingleClass(flipindex, clazz)
                    flipfit = ffit1 + ffit2
                    takelist.append((flipfit,flipindex))
            takelist.sort()
            
            takeind = 0
            takecount = 0
            while True:
                fit, flipindex = takelist[takeind]
                oldclazz = self.classvec[flipindex]
                if self.classcounts[oldclazz] > self.constraint: 
                    self.flipClass(flipindex, clazz)
                    takecount += 1
                    if takecount >= min(howmany, len(takelist)): break
                takeind += 1
        
        self.updateA()
    
    
    
    
    def giveAndTakeALT(self, howmany):
        
        allinds = set(range(self.size))
        allinds = allinds - set(self.fixedindices)
        allinds = list(allinds)
        random.shuffle(allinds)
        
        for clazz in range(self.labelcount):
            givelist = []
            for flipindex in allinds:
                if self.classvec[flipindex] != clazz:
                    continue
                else:
                    ffit1 = self.computeFlipFitnessForSingleClass(flipindex, clazz)
                    bestfit = None
                    bestclass = None
                    for newclassind in range(self.labelcount):
                        if newclassind == clazz: continue
                        ffit2 = self.computeFlipFitnessForSingleClass(flipindex, newclassind)
                        flipfit = ffit1 + ffit2
                        if bestfit is None or bestfit > flipfit:
                            bestfit = flipfit
                            bestclass = newclassind
                    givelist.append((ffit1, bestclass, flipindex))
            givelist.sort()
            
            for i in range(min(howmany,len(givelist))):
                fit, newclassind, flipindex = givelist[i]
                self.flipClass(flipindex, newclassind)
        
            takelist = []
            for flipindex in allinds:
                if self.classvec[flipindex] != clazz:
                    ffit2 = self.computeFlipFitnessForSingleClass(flipindex, clazz)
                    takelist.append((ffit2,flipindex))
            takelist.sort()
            
            takeind = 0
            takecount = 0
            while True:
                fit, flipindex = takelist[takeind]
                oldclazz = self.classvec[flipindex]
                if self.classcounts[oldclazz] > self.constraint: 
                    self.flipClass(flipindex, clazz)
                    takecount += 1
                    if takecount >= min(howmany, len(takelist)): break
                takeind += 1
        self.updateA()




class RandomLabelSource(object):
    
    def __init__(self, size, labelcount):
        self.rand = random.Random()
        self.rand.seed(100)
        self.Y = - np.ones((size, labelcount), dtype = np.float64)
        self.classvec = - np.ones((size), dtype = np.int32)
        allinds = set(range(size))
        self.classcounts = np.zeros((labelcount), dtype = np.int32)
        for i in range(labelcount-1):
            inds = self.rand.sample(allinds, size / labelcount) #sampling without replacement
            allinds = allinds - set(inds)
            for ind in inds:
                self.Y[ind, i] = 1.
                self.classvec[ind] = i
                self.classcounts[i] += 1
        for ind in allinds:
            self.Y[ind, labelcount - 1] = 1.
            self.classvec[ind] = labelcount - 1
            self.classcounts[labelcount - 1] += 1
    
    def readLabels(self):
        return self.Y

