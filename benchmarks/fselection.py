#
# The MIT License (MIT)
#
# This file is part of RLScore 
#
# Copyright (c) 2016 Tapio Pahikkala, Antti Airola
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

#Benchmark code for comparing the fast greedy RLS algorithm implemented in RLScore
#to a baseline method that uses the fast leave-one-out algorithm in sklearn to speed
#up selection. The compared methods are equivalent, but the RLScore algorithm has
#lower computational complexity.
#
#Greedy RLS is described in  [1,2]. The extension of the method to multi-target learning
#was proposed in [3].           

#    [1] Tapio Pahikkala, Antti Airola, and Tapio Salakoski.
#    Speeding up Greedy Forward Selection for Regularized Least-Squares.
#    Proceedings of The Ninth International Conference on Machine Learning and Applications,
#    325-330, IEEE Computer Society, 2010.
#    
#    [2] Tapio Pahikkala, Sebastian Okser, Antti Airola, Tapio Salakoski, and Tero Aittokallio.
#    Wrapper-based selection of genetic features in genome-wide association studies through fast matrix operations.
#    Algorithms for Molecular Biology, 7(1):11, 2012.
#    
#    [3] Pekka Naula, Antti Airola, Tapio Salakoski, and Tapio Pahikkala.
#    Multi-label learning under feature extraction budgets.
#    Pattern Recognition Letters, 40:56--65, April 2014. 

import numpy as np
import time

from rlscore.learner import GreedyRLS
from sklearn.linear_model import RidgeCV

def make_classification(size, n_features, n_classes):
    X = np.random.randn(size, n_features)
    Y = np.random.randn(size, n_classes)
    return X, Y

class Callback(object):

    def __init__(self):
        self.start = time.clock()
        self.iteration = 0

    def callback(self, learner):
        self.iteration += 1
        dur = time.clock() - self.start
        print("Iteration %d, CPU time %f, LOO error %f, selected %d" %(self.iteration, dur, learner.performances[-1], learner.selected[-1]))

    def finished(self, learner):
        pass

def loo_sklearn(X,y, regparam):
    learner = RidgeCV(alphas = [regparam], store_cv_values = True, fit_intercept=False)
    learner.fit(X,y)
    e = np.mean(learner.cv_values_[:,:,0])
    return e

def sklearn_greedyrls(X,y, regparam, scount):
    start = time.clock()
    findices = range(X.shape[1])
    selected = []
    for i in range(scount):
        errors = []
        tested = []
        for j in findices:
            fset = selected[:]
            fset.append(j)
            X_new = X[:, fset]
            e = loo_sklearn(X_new, y, regparam)
            errors.append(e)
            tested.append(j)
        best_ind = np.argmin(errors)
        best = findices.pop(best_ind)
        selected.append(best)
        dur = time.clock() - start
        print("Iteration %d, CPU time %f, LOO error %f, selected %d" %(i+1, dur, min(errors), best))
    return selected

def core_greedyrls(X, y, regparam, scount):
    cb = Callback()
    learner = GreedyRLS(X, y, scount, regparam=regparam, callbackfun=cb, bias=0.)
    selected = learner.selected
    return selected

if __name__=="__main__":
    #Selects 25 features with greedy forward selection
    #both with RLScore and with the baseline. The data
    #has 10000 training instances, 1000 features and 10
    #output targets. Both methods print CPU time,
    #leave-one-out error and index of selected feature
    #on each iteration (note that the LOO error becomes
    #optimistically biased due to overfitting via feature selection).
    np.random.seed(10)
    X, Y = make_classification(10000, 1000, 10)
    print("RLScore running")
    core_greedyrls(X, Y, 1.0, 25)
    print("sklearn running")
    sklearn_greedyrls(X, Y,1.0 , 25)
