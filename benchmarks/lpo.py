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

#Benchmark code for comparing the fast leave-pair-out algorithm implemented in
#RLScore to a naive implementation based on sklearn. The algorithm is an adaptation
#of the method published originally in [1]. The use of leave-pair-out cross-validation
#for AUC estimation has been analyzed in [2].
#
#    [1] Tapio Pahikkala, Antti Airola, Jorma Boberg, and Tapio Salakoski.
#    Exact and efficient leave-pair-out cross-validation for ranking RLS.
#    In Proceedings of the 2nd International and Interdisciplinary Conference
#    on Adaptive Knowledge Representation and Reasoning (AKRR'08), pages 1-8,
#    Espoo, Finland, 2008.
#        
#    [2] Antti Airola, Tapio Pahikkala, Willem Waegeman, Bernard De Baets, and Tapio Salakoski.
#    An experimental comparison of cross-validation techniques for estimating the area under the ROC curve.
#    Computational Statistics & Data Analysis, 55(4):1828--1844, April 2011.

import numpy as np
import time

from rlscore.learner import RLS
from rlscore.measure import sqerror
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import LeavePOut


def random_data(size, n_features):
    np.random.seed(77)
    X = np.random.randn(size, n_features)
    Y = np.random.randn(size)
    return X, Y

def lpo_core(X,y, regparam):
    start, end = [], []
    for i in range(X.shape[0]-1):
        for j in range(i+1, X.shape[0]):
            start.append(i)
            end.append(j)
    rls = RLS(X,y, regparam=regparam, kernel="GaussianKernel", gamma=0.01)
    pred0, pred1 = rls.leave_pair_out(start, end)
    return pred0, pred1

def lpo_sklearn(X,y, regparam):
    lpo = LeavePOut(p=2)
    preda = []
    predb = []
    for train, test in lpo.split(X):
        rls = KernelRidge(kernel="rbf", gamma=0.01)
        rls.fit(X[train], y[train])
        p = rls.predict(X[test])
        preda.append(p[0])
        predb.append(p[1])
    return preda, predb

if __name__=="__main__":
    #computes leave-pair-out for different sample sizes
    #comparing CPU time and verifying that the LPO
    #predictions are same for both methods
    sizes = [50, 100, 150, 200, 400, 600, 800, 1000]
    for size in sizes:
        X, y = random_data(size, 500)
        start = time.clock()
        pred0, pred1 = lpo_core(X, y, 1.0)
        dur = time.clock() - start
        print("RLScore instances: %d, CPU time: %f" %(size, dur))
        start = time.clock()
        preda, predb = lpo_sklearn(X, y, 1.0)
        dur = time.clock() - start
        print("scikit-learn instances: %d, CPU time: %f" %(size, dur))
        print("are lpo predictions same: %r" %(np.allclose(pred0, preda) and np.allclose(pred1, predb)))
        print("******")
