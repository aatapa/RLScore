#
# The MIT License (MIT)
#
# This file is part of RLScore 
#
# Copyright (c) 2013 - 2016 Tapio Pahikkala, Antti Airola
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

import numpy as np
import random as pyrandom
pyrandom.seed(200)

from . import _steepest_descent_mmc

from rlscore.utilities import adapter
from rlscore.utilities import array_tools
from rlscore.predictor import PredictorInterface

class SteepestDescentMMC(PredictorInterface):
    
    """RLS-based maximum-margin clustering. Performs steepest descent search with a shaking heuristic to avoid getting stuck in
    local minima.

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
    
    The steepest descent variant of the algorithm is described in [1].    
    
    References
    ----------
    
    [1] Tapio Pahikkala, Antti Airola, Fabian Gieseke, and Oliver Kramer.
    Unsupervised multi-class regularized least-squares classification.
    The 12th IEEE International Conference on Data Mining (ICDM 2012), pages 585--594.
    IEEE Computer Society, December 2012
    """
    
    def __init__(self, X, regparam=1.0, number_of_clusters=2, kernel='LinearKernel', basis_vectors=None, Y = None, fixed_indices=None, callback=None,  **kwargs):
        kwargs['X'] = X
        kwargs['kernel'] = kernel
        if basis_vectors is not None:
            kwargs['basis_vectors'] = basis_vectors
        self.svdad = adapter.createSVDAdapter(**kwargs)
        self.svals = np.mat(self.svdad.svals)
        self.svecs = np.mat(self.svdad.rsvecs)
        self.callbackfun = callback
        self.regparam = regparam
        self.constraint = 0
        self.labelcount = int(number_of_clusters)
         
        if self.labelcount == 2:
            self.oneclass = True
        else:
            self.oneclass = False
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
        self.lockvec = np.zeros((self.size), dtype = np.int32)
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
        self.regparam = regparam
        
        #Cached results
        self.evals = np.multiply(self.svals, self.svals)
        self.newevals = 1. / (self.evals + self.regparam)
        newevalslamtilde = np.multiply(self.evals, self.newevals)
        self.D = np.sqrt(newevalslamtilde)
        
        self.VTY = self.svecs.T * self.Y
        
        self.sqrtR = np.multiply(np.sqrt(newevalslamtilde), self.svecs)
        
        self.R = self.sqrtR * self.sqrtR.T
        self.mdiagRx2 = - 2 * np.diag(self.R)
        
        #'''
        #Space efficient variation
        #self.R = None
        #self.mdiagRx2 = - 2 * np.array(np.sum(np.multiply(self.sqrtR, self.sqrtR), axis = 1)).reshape((self.size))
        #'''
        
        self.RY = self.sqrtR * (self.sqrtR.T * self.Y)
        self.Y_Schur_RY = np.multiply(self.Y, self.RY)
        
        self.YTRY_list = []
        self.classFitnessList = []
        
        for i in range(self.labelcount):
            YTRY_i = self.Y[:,i].T * self.RY[:,i]
            self.YTRY_list.append(YTRY_i)
            fitness_i = self.size - YTRY_i
            self.classFitnessList.append(fitness_i[0, 0])
        self.classFitnessRowVec = np.array(self.classFitnessList)
        
        self.updateA()
        
        
        converged = False
        if not self.callbackfun is None:
            self.callbackfun.callback(self)
        
        cons = self.size / self.labelcount
        for i in range(20):
            converged = self.findSteepestDirRotateClasses(cons / (2. ** i))
            if not self.callbackfun is None:
                self.callbackfun.callback(self)
            if converged: break
        
        if self.oneclass:
            self.Y = self.Y[:, 0]
            self.A = self.A[:, 0]
        self.results = {}
        self.results['predicted_clusters_for_training_data'] = self.Y
        self.predictor = self.svdad.createModel(self)
    
    
    def computeGlobalFitness(self):
        fitness = 0.
        for classind in range(self.labelcount):
            fitness += self.classFitnessList[classind]
        return fitness
    
    
    def updateA(self):
        self.A = self.svecs * np.multiply(self.newevals.T, self.VTY)
    
    def claim_n_points(self, howmany, newclazz):
        _steepest_descent_mmc.claim_n_points(self.Y,
                                                self.R,
                                                self.RY,
                                                self.Y_Schur_RY,
                                                self.classFitnessRowVec,
                                                self.mdiagRx2,
                                                self.classcounts,
                                                self.classvec,
                                                self.size,
                                                self.labelcount,
                                                howmany,
                                                self.sqrtR,
                                                self.sqrtR.shape[1],
                                                self.lockvec,
                                                newclazz)
    
    def findSteepestDirRotateClasses(self, howmany, LOO = False):
        _steepest_descent_mmc.findSteepestDirRotateClasses(self.Y,
                                                self.R,
                                                self.RY,
                                                self.Y_Schur_RY,
                                                self.classFitnessRowVec,
                                                self.mdiagRx2,
                                                self.classcounts,
                                                self.classvec,
                                                self.size,
                                                self.labelcount,
                                                howmany,
                                                self.sqrtR,
                                                self.sqrtR.shape[1],
                                                self.lockvec)
        return
        
        #The slow python code. Use the above cython instead.
        for newclazz in range(self.labelcount):
            
            #!!!!!!!!!!!!!!!
            takenum = (self.size / self.labelcount) - self.classcounts[newclazz] + int(howmany)
            
            for h in range(takenum):
                dirsneg = self.classFitnessRowVec + (2 * self.mdiagRx2[:, None] + 4 * np.multiply(self.Y, self.RY))
                dirsnegdiff = dirsneg - self.classFitnessRowVec
                dirscc = dirsnegdiff[np.arange(self.size), self.classvec].T
                dirs = dirsnegdiff + dirscc
                dirs[np.arange(self.size), self.classvec] = float('Inf')
                dirs = dirs[:, newclazz]
                steepestdir = np.argmin(dirs)
                steepness = np.amin(dirs)
                oldclazz = self.classvec[steepestdir]
                self.Y[steepestdir, oldclazz] = -1.
                self.Y[steepestdir, newclazz] = 1.
                self.classvec[steepestdir] = newclazz
                self.classcounts[oldclazz] = self.classcounts[oldclazz] - 1
                self.classcounts[newclazz] = self.classcounts[newclazz] + 1
                self.RY[:, oldclazz] = self.RY[:, oldclazz] - 2 * self.R[:, steepestdir]
                self.RY[:, newclazz] = self.RY[:, newclazz] + 2 * self.R[:, steepestdir]
                
                for i in range(self.labelcount):
                    YTRY_i = self.Y[:,i].T * self.RY[:,i]
                    fitness_i = self.size - YTRY_i
                    self.classFitnessRowVec[i] = fitness_i[0, 0]
                
                self.updateA()
            #self.callback()
        return False
    
    
    def findSteepestDirRotateClasses_FOCUSSETSTUFF(self, howmany, LOO = False):
        
        focusrange = np.mat(np.arange(self.size)).T
        
        for j in range(self.labelcount):
            #!!!!!!!!!!!!!!!
            takenum = (self.size / self.labelcount) - self.classcounts[j] + int(howmany)
            #takenum = min([maxtake, (self.size / self.labelcount) - self.classcounts[j] + int(howmany)])
            #self.focusset = self.findNewFocusSet(j)
            #self.focusset = pyrandom.sample(range(self.size),50)
            #takenum = 2*((len(self.focusset) / self.labelcount) - (len(self.focusset) / self.size) * self.classcounts[j] + int(howmany))
            #self.focusset = pyrandom.sample(range(self.size),takenum[0,0])
            #self.focusset = self.findNewFocusSet(j, maxtake)
            #self.focusset = set(range(self.size))
            for h in range(takenum):
                self.focusset = set(pyrandom.sample(range(self.size),10))
                dirsnegdiff = 2 * self.mdiagRx2 + 4 * np.multiply(self.Y, self.RY)
                dirscc = dirsnegdiff[focusrange, self.classvec]
                dirs = dirsnegdiff + dirscc
                dirs[focusrange, self.classvec] = float('Inf')
                dirs[list(set(range(self.size))-self.focusset)] = float('Inf')
                dirs = dirs[:, j]
                steepestdir, newclazz = np.unravel_index(np.argmin(dirs), dirs.shape)
                newclazz = j
                oldclazz = self.classvec[steepestdir, 0]
                
                self.Y[steepestdir, oldclazz] = -1.
                self.Y[steepestdir, newclazz] = 1.
                self.classvec[steepestdir] = newclazz
                self.classcounts[oldclazz] = self.classcounts[oldclazz] - 1
                self.classcounts[newclazz] = self.classcounts[newclazz] + 1
                self.RY[:, oldclazz] = self.RY[:, oldclazz] - 2 * self.R[steepestdir].T
                self.RY[:, newclazz] = self.RY[:, newclazz] + 2 * self.R[steepestdir].T
            '''
            while True:
                if takecount >= takenum: break
                for h in range(maxtake):
                    diagR = mat(diag(self.R)).T
                    dirsnegdiff = - 4 * diagR + 4 * np.multiply(self.Y, self.RY)
                    dirscc = dirsnegdiff[focusrange, self.classvec]
                    dirs = dirsnegdiff + dirscc
                    dirs[focusrange, self.classvec] = float('Inf')
                    dirs[list(set(range(self.size))-self.focusset)] = float('Inf')
                    dirs = dirs[:, j]
                    steepestdir, newclazz = unravel_index(argmin(dirs), dirs.shape)
                    newclazz = j
                    oldclazz = self.classvec[steepestdir, 0]
                    
                    self.Y[steepestdir, oldclazz] = -1.
                    self.Y[steepestdir, newclazz] = 1.
                    self.classvec[steepestdir] = newclazz
                    self.classcounts[oldclazz] = self.classcounts[oldclazz] - 1
                    self.classcounts[newclazz] = self.classcounts[newclazz] + 1
                    self.RY[:, oldclazz] = self.RY[:, oldclazz] - 2 * self.R[steepestdir].T
                    self.RY[:, newclazz] = self.RY[:, newclazz] + 2 * self.R[steepestdir].T
                    takecount += 1
                    if takecount >= takenum: break
                if takecount >= takenum: break
                self.focusset = self.findNewFocusSet(j, maxtake)
                '''
            self.callback()
        return False
    
    
    def findNewFocusSet(self, clazz = 0, focsize = 50):
        
        diagR = np.mat(np.diag(self.R)).T
        dirsnegdiff = - 4 * diagR + 4 * np.multiply(self.Y, self.RY)
        dirscc = dirsnegdiff[np.mat(np.arange(self.size)).T, self.classvec]
        dirs = dirsnegdiff + dirscc
        dirs[np.mat(np.arange(self.size)).T, self.classvec] = float('Inf')
        
        dirlist = []
        for i in range(self.size):
            row = dirs[i]
            #dirlist.append((amin(row), i))
            dirlist.append((row[0, clazz], i))
            dirlist = sorted(dirlist)[0:focsize]
        focusset = []
        for i in range(focsize):
            focusset.append(dirlist[i][1])
        return set(focusset)
        





class RandomLabelSource(object):
    
    def __init__(self, size, labelcount):
        self.rand = pyrandom.Random()
        self.rand.seed(100)
        self.Y = - np.ones((size, labelcount), dtype = np.float64)
        self.classvec = - np.ones((size), dtype = np.int32)
        allinds = set(range(size))
        self.classcounts = np.zeros((labelcount), dtype = np.int32)
        for i in range(labelcount-1):
            inds = self.rand.sample(allinds, int(size / labelcount)) #sampling without replacement
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

