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

#Benchmark code for comparing the fast leave-group-out algorithm implemented in
#RLScore to a naive implementation based on sklearn. The fast holdout algorithm
#is based on results presented in [1,2].    
#        
#        [1] Tapio Pahikkala, Jorma Boberg, and Tapio Salakoski.
#        Fast n-Fold Cross-Validation for Regularized Least-Squares.
#        Proceedings of the Ninth Scandinavian Conference on Artificial Intelligence,
#        83-90, Otamedia Oy, 2006.
#        
#        [2] Tapio Pahikkala, Hanna Suominen, and Jorma Boberg.
#        Efficient cross-validation for kernelized least-squares regression with sparse basis expansions.
#        Machine Learning, 87(3):381--407, June 2012.

import numpy as np
import time

from rlscore.learner import RLS
from rlscore.measure import sqerror
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import LeaveOneGroupOut


def random_data(size, n_features):
    np.random.seed(77)
    X = np.random.randn(size, n_features)
    Y = np.random.randn(size)
    groups = [i/10 for i in range(size)]
    return X, Y, groups

def lgo_core(X,y, groups, regparam):
    logo = LeaveOneGroupOut()
    rls = RLS(X,y, regparam=regparam, kernel="GaussianKernel", gamma=0.01)
    errors = []
    for train, test in logo.split(X, y, groups=groups):
        p = rls.holdout(test)
        e = sqerror(y[test], p)
        errors.append(e)
    return np.mean(errors)

def lgo_sklearn(X,y, groups, regparam):
    logo = LeaveOneGroupOut()
    errors = []
    for train, test in logo.split(X, y, groups=groups):
        rls = KernelRidge(kernel="rbf", gamma=0.01)
        rls.fit(X[train], y[train])
        p = rls.predict(X[test])
        e = sqerror(y[test], p)       
        errors.append(e)
    return np.mean(errors)

if __name__=="__main__":
    #computes leave-group-out for different sample sizes
    #comparing CPU time and verifying that the LGO mean
    #squared erros are the same for both methods
    sizes = [2**i*100 for i in range(8)]
    for size in sizes:
        X, y, groups = random_data(size, 500)
        start = time.clock()
        e = lgo_core(X, y, groups, 1.0)
        dur = time.clock() - start
        print("RLScore instances: %d, CPU time: %f, LGO MSE: %f" %(size, dur, e))
        start = time.clock()
        e = lgo_sklearn(X, y, groups, 1.0)
        dur = time.clock() - start
        print("scikit-learn instances: %d, CPU time: %f, LGO MSE: %f" %(size, dur, e))
        print("******")
