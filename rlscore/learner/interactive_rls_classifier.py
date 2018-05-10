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

from . import _interactive_rls_classifier

import random as pyrandom
pyrandom.seed(200)
import numpy as np
from rlscore.utilities import adapter

class InteractiveRlsClassifier(object):
    
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
        self.labelcount = number_of_clusters
        self.size = X.shape[0] 
        #if self.labelcount == 2:
        #    self.oneclass = True
        #else:
        #    self.oneclass = False
        
        if Y is None:
            self.classvec = np.zeros(self.size, np.int)
        else:
            self.classvec = Y
        #self.size = self.classvec.shape[0]
        self.Y = -np.ones((self.size, self.labelcount))
        self.classcounts = np.zeros((self.labelcount), dtype = np.int32)
        for i in range(self.size):
            clazzind = self.classvec[i]
            self.Y[i, clazzind] = 1
            self.classcounts[clazzind] = self.classcounts[clazzind] + 1
        
        self.fixedindices = []
        if fixed_indices is not None:
            self.fixedindices = fixed_indices
        self.train()
    
    
    def train(self):
        
        #Cached results
        self.evals = np.multiply(self.svals, self.svals)
        self.newevals = 1. / (self.evals + self.regparam)
        newevalslamtilde = np.multiply(self.evals, self.newevals)
        self.D = np.sqrt(newevalslamtilde)
        #self.D = -newevalslamtilde
        
        self.VTY = self.svecs.T * self.Y
        self.DVTY = np.multiply(self.D.T, self.svecs.T * self.Y)
        
        self.sqrtR = np.multiply(np.sqrt(newevalslamtilde), self.svecs)
        
        #self.R = self.svecs * multiply(newevalslamtilde.T, self.svecs.T)
        
        '''
        #Global variation
        self.R = self.sqrtR * self.sqrtR.T
        self.minus_diagRx2 = - 2 * np.diag(self.R)
        '''
        #Space efficient variation
        self.R = None
        self.minus_diagRx2 = - 2 * np.array(np.sum(np.multiply(self.sqrtR, self.sqrtR), axis = 1)).reshape((self.size))
        #'''
        
        self.RY = self.sqrtR * (self.sqrtR.T * self.Y)
        self.Y_Schur_RY = np.multiply(self.Y, self.RY)
        
        self.classFitnessList = []
        for i in range(self.labelcount):
            #DVTY_i = DVTY[:,i]
            #self.DVTY_list.append(DVTY_i)
            YTRY_i = self.Y[:,i].T * self.RY[:,i]
            #self.YTRY_list.append(YTRY_i)
            fitness_i = self.size - YTRY_i
            self.classFitnessList.append(fitness_i[0, 0])
        self.classFitnessRowVec = np.array(self.classFitnessList)
        
        if not self.callbackfun is None:
            self.callbackfun.callback(self)
        
        #Initial working set contains all points
        self.new_working_set(list(range(self.size)))
        
    
    
    def computeGlobalFitness(self):
        fitness = 0.
        for classind in range(self.labelcount):
            fitness += self.classFitnessList[classind]
        return fitness
    
    
    def updateA(self):
        self.A = self.svecs * np.multiply(self.newevals.T, self.VTY)
    
    
    def new_working_set(self, working_set):
        self.working_set = working_set
        
        #self.RY = self.sqrtR * (self.sqrtR.T * self.Y)
        #self.Y_Schur_RY = np.multiply(self.Y, self.RY)
        #self.Y_Schur_RY_ws = self.Y_Schur_RY[working_set]
        self.RY = None
        self.Y_Schur_RY = None
        self.Y_Schur_RY_ws = None
        
        self.Y_ws = self.Y[working_set]
        #self.R_ws = self.R[np.ix_(working_set, working_set)]
        #self.R_ws = self.sqrtR[working_set] * self.sqrtR[working_set].T
        self.R_ws = None
        self.sqrtRx2_ws = 2 * self.sqrtR[working_set]
        #self.RY_ws = self.RY[working_set]
        self.RY_ws = None
        #self.minus_diagRx2_ws = self.minus_diagRx2[working_set]
        self.minus_diagRx2_ws = None
        self.classvec_ws = self.classvec[working_set]
        self.size_ws = len(working_set)
        
        self.classcounts_ws = np.zeros((self.labelcount), dtype = np.int32)
        for i in range(self.labelcount):
            self.classcounts_ws[i] = self.size_ws - np.count_nonzero(self.classvec_ws - i)
        #self.compute_steepness_vector()
        

    def compute_steepness_vector(self):        
        gradient_vec0 = np.zeros((self.size_ws))
        tempvec = np.zeros((self.Y.shape[1]))
        _interactive_rls_classifier.compute_gradient(self.Y_ws,
                     gradient_vec0,
                     self.classcounts_ws,
                     self.classvec_ws,
                     self.size_ws,
                     self.size,
                     self.DVTY,
                     self.sqrtRx2_ws,
                     self.sqrtR.shape[1],
                     0,
                     tempvec,
                     self.Y.shape[1])
        gradient_vec1 = np.zeros((self.size_ws))
        tempvec = np.zeros((self.Y.shape[1]))
        _interactive_rls_classifier.compute_gradient(self.Y_ws,
                     gradient_vec1,
                     self.classcounts_ws,
                     self.classvec_ws,
                     self.size_ws,
                     self.size,
                     self.DVTY,
                     self.sqrtRx2_ws,
                     self.sqrtR.shape[1],
                     1,
                     tempvec,
                     self.Y.shape[1])
        
        steepness_vector = np.zeros((self.size_ws))
        steepness_vector[0:self.classcounts_ws[1]] = np.sort(gradient_vec0)[0:self.classcounts_ws[1]][::-1]
        steepness_vector[self.classcounts_ws[1]:] = np.sort(gradient_vec1)[0:self.classcounts_ws[0]]
        #print steepness_vector
        return steepness_vector
    
    
    def claim_all_points_in_working_set(self, newclazz):
        #'''
        _interactive_rls_classifier.claim_all_points(self.Y_ws,
                     self.classcounts_ws,
                     self.classvec_ws,
                     self.size_ws,
                     self.DVTY,
                     self.sqrtRx2_ws,
                     self.sqrtR.shape[1],
                     newclazz)
        #Update global books DOES NOT WORK FOR RY or Y_Schur_RY!!!
        self.Y[self.working_set] = self.Y_ws
        #print np.sum(np.abs(np.multiply(self.D.T, self.svecs.T * self.Y) - self.DVTY))
        #self.RY[working_set] = self.RY_ws
        #self.Y_Schur_RY[working_set] = self.Y_Schur_RY_ws
        self.classvec[self.working_set] = self.classvec_ws
        for i in range(self.labelcount):
            self.classcounts[i] = self.size - np.count_nonzero(self.classvec - i)
        
        #self.compute_steepness_vector()
    
    
    def cyclic_descent_in_working_set(self, maxbalancechange = None):
        fitvec = np.zeros(self.labelcount)
        if maxbalancechange is None:
            maxbalancechange = self.size_ws
        changecount = _interactive_rls_classifier.cyclic_desccent(self.Y_ws,
                     self.classcounts_ws,
                     self.classvec_ws,
                     fitvec,
                     self.size_ws,
                     self.size,
                     self.DVTY,
                     self.sqrtRx2_ws,
                     self.sqrtR.shape[1],
                     self.labelcount,
                     maxbalancechange,
                     np.zeros((self.labelcount), dtype = np.int32))
        #Update global books DOES NOT WORK FOR RY or Y_Schur_RY!!!
        self.Y[self.working_set] = self.Y_ws
        #print np.sum(np.abs(np.multiply(self.D.T, self.svecs.T * self.Y) - self.DVTY))
        #self.RY[working_set] = self.RY_ws
        #self.Y_Schur_RY[working_set] = self.Y_Schur_RY_ws
        self.classvec[self.working_set] = self.classvec_ws
        for i in range(self.labelcount):
            self.classcounts[i] = self.size - np.count_nonzero(self.classvec - i)
        return changecount
    
    
#     def claim_a_point(self, newclazz):
#         working_set = self.working_set
#         if self.classcounts_ws[newclazz] == self.size_ws:
#             raise Exception('The whole working set already belongs to class '+str(newclazz))
#         use_full_caches = 0
#         tempvec = np.zeros((self.Y.shape[1]))
#         steepestdir, oldclazz = _interactive_rls_classifier.claim_a_point(self.Y_ws,
#                                  self.R_ws,
#                                  self.RY_ws,
#                                  self.Y_Schur_RY_ws,
#                                  self.minus_diagRx2_ws,
#                                  self.classcounts_ws,
#                                  self.classvec_ws,
#                                  self.size_ws,
#                                  use_full_caches,
#                                  self.DVTY,
#                                  self.sqrtRx2_ws,
#                                  self.sqrtR.shape[1],
#                                  newclazz,
#                                  tempvec,
#                                  self.Y.shape[1])
#         #Update global books DOES NOT WORK FOR RY or Y_Schur_RY!!!
#         self.Y[working_set] = self.Y_ws
#         self.classvec[working_set] = self.classvec_ws
#         self.classcounts[oldclazz] -= 1
#         self.classcounts[newclazz] += 1
#         
#         global_steepestdir = working_set[steepestdir]
#         return global_steepestdir
    
    
    def claim_n_points(self, nnn, newclazz):
        working_set = self.working_set
        if self.classcounts_ws[newclazz] == self.size_ws:
            raise Exception('The whole working set already belongs to class '+str(newclazz))
        gradient_vec = np.zeros((self.size_ws))
        tempvec = np.zeros((self.Y.shape[1]))
        _interactive_rls_classifier.compute_gradient(self.Y_ws,
                     gradient_vec,
                     self.classcounts_ws,
                     self.classvec_ws,
                     self.size_ws,
                     self.size,
                     self.DVTY,
                     self.sqrtRx2_ws,
                     self.sqrtR.shape[1],
                     newclazz,
                     tempvec,
                     self.Y.shape[1])
        
        claiminds = np.argsort(gradient_vec)
        
        clazzcountchanges = np.zeros((self.labelcount), dtype = np.int32)
        _interactive_rls_classifier.claim_n_points(self.Y_ws,
                     gradient_vec,
                     self.classcounts_ws,
                     self.classvec_ws,
                     claiminds,
                     nnn,
                     self.DVTY,
                     self.sqrtRx2_ws,
                     self.sqrtR.shape[1],
                     newclazz,
                     clazzcountchanges)
        
        self.Y[working_set] = self.Y_ws
        self.classvec[working_set] = self.classvec_ws
        #self.classcounts[oldclazz] -= nnn
        self.classcounts[newclazz] += nnn
        for i in range(self.labelcount):
            self.classcounts[i] += clazzcountchanges[i]
        #self.compute_steepness_vector()
        #global_steepestdir = working_set[steepestdir]
        #return global_steepestdir
    
    
#     def findSteepestDirRotateClasses(self, howmany, LOO = False):
#         
#         working_set = pyrandom.sample(range(self.size), self.size)
#         Y = self.Y[working_set]
#         R = self.R[np.ix_(working_set, working_set)]
#         RY = self.RY[working_set]
#         Y_Schur_RY = self.Y_Schur_RY[working_set]
#         minus_diagRx2 = self.minus_diagRx2[working_set]
#         classvec = self.classvec[working_set]
#         size = len(working_set)
#         
#         cython_mmc.findSteepestDirRotateClasses(Y,
#                                                 R,
#                                                 RY,
#                                                 Y_Schur_RY,
#                                                 self.classFitnessRowVec,
#                                                 minus_diagRx2,
#                                                 self.classcounts,
#                                                 classvec,
#                                                 size,
#                                                 self.labelcount,
#                                                 howmany,
#                                                 self.sqrtR,
#                                                 self.sqrtR.shape[1])
#         self.Y[working_set] = Y
#         self.RY[working_set] = RY
#         self.Y_Schur_RY[working_set] = Y_Schur_RY
#         self.classvec[working_set] = classvec
#         return
#     
#     
#     def findSteepestDirRotateClasses_(self, howmany, LOO = False):
#         cython_mmc.findSteepestDirRotateClasses(self.Y,
#                                                 self.R,
#                                                 self.RY,
#                                                 self.Y_Schur_RY,
#                                                 self.classFitnessRowVec,
#                                                 self.minus_diagRx2,
#                                                 self.classcounts,
#                                                 self.classvec,
#                                                 self.size,
#                                                 self.labelcount,
#                                                 howmany,
#                                                 self.sqrtR,
#                                                 self.sqrtR.shape[1])
#         return
#         
#         #The slow python code. Use the above cython instead.
#         for newclazz in range(self.labelcount):
#             
#             #!!!!!!!!!!!!!!!
#             takenum = (self.size / self.labelcount) - self.classcounts[newclazz] + int(howmany)
#             
#             for h in range(takenum):
#                 dirsneg = self.classFitnessRowVec + (2 * self.minus_diagRx2[:, None] + 4 * np.multiply(self.Y, self.RY))
#                 dirsnegdiff = dirsneg - self.classFitnessRowVec
#                 dirscc = dirsnegdiff[np.arange(self.size), self.classvec].T
#                 dirs = dirsnegdiff + dirscc
#                 dirs[np.arange(self.size), self.classvec] = float('Inf')
#                 dirs = dirs[:, newclazz]
#                 steepestdir = np.argmin(dirs)
#                 steepness = np.amin(dirs)
#                 oldclazz = self.classvec[steepestdir]
#                 self.Y[steepestdir, oldclazz] = -1.
#                 self.Y[steepestdir, newclazz] = 1.
#                 self.classvec[steepestdir] = newclazz
#                 self.classcounts[oldclazz] = self.classcounts[oldclazz] - 1
#                 self.classcounts[newclazz] = self.classcounts[newclazz] + 1
#                 self.RY[:, oldclazz] = self.RY[:, oldclazz] - 2 * self.R[:, steepestdir]
#                 self.RY[:, newclazz] = self.RY[:, newclazz] + 2 * self.R[:, steepestdir]
#                 
#                 for i in range(self.labelcount):
#                     YTRY_i = self.Y[:,i].T * self.RY[:,i]
#                     fitness_i = self.size - YTRY_i
#                     self.classFitnessRowVec[i] = fitness_i[0, 0]
#                 
#                 self.updateA()
#             #self.callback()
#         return False
        





class RandomLabelSource(object):
    
    def __init__(self, size, labelcount):
        self.rand = pyrandom.Random()
        self.rand.seed(100)
        self.Y = - np.ones((size, labelcount), dtype = np.float64)
        self.classvec = - np.ones((size, 1), dtype = np.int32)
        allinds = set(range(size))
        self.classcounts = np.zeros((labelcount, 1), dtype = np.int32)
        for i in range(labelcount-1):
            inds = self.rand.sample(allinds, size / labelcount) #sampling without replacement
            allinds = allinds - set(inds)
            for ind in inds:
                self.Y[ind, i] = 1.
                self.classvec[ind, 0] = i
                self.classcounts[i, 0] += 1
        for ind in allinds:
            self.Y[ind, labelcount - 1] = 1.
            self.classvec[ind, 0] = labelcount - 1
            self.classcounts[labelcount - 1, 0] += 1
    
    def readLabels(self):
        return self.Y

