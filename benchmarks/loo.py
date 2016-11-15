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

#Benchmark code for comparing the fast leave-one-out algorithm implemented in RLScore
#to the implementation in sklearn. Both packages implement the algorithm described
#[1] (as well as in several earlier works).
#    
#   [1] Ryan Rifkin, Ross Lippert. Notes on Regularized Least Squares.
#   Technical Report, MIT, 2007.


import numpy as np
import time

from rlscore.learner import RLS
from rlscore.measure import sqerror
from sklearn.linear_model import RidgeCV

def random_data(size, n_features):
    np.random.seed(77)
    X = np.random.randn(size, n_features)
    Y = np.random.randn(size)
    return X, Y

def loo_sklearn(X,y, regparam):
    learner = RidgeCV(alphas = [regparam], fit_intercept=False, store_cv_values = True)
    learner.fit(X,y)
    e = np.mean(learner.cv_values_)
    return e

def loo_core(X,y,regparam):
    learner = RLS(X,y,regparam, bias=0.)
    p = learner.leave_one_out()
    e = sqerror(y, p)
    return e

if __name__=="__main__":
    #computes leave-one-out for different sample sizes
    #comparing CPU time and verifying that the LOO mean
    #squared erros are the same for both methods
    sizes = [2**i*100 for i in range(8)]
    for size in sizes:
        X, y = random_data(size, size)
        start = time.clock()
        e = loo_core(X, y, 1.0)
        dur = time.clock() - start
        print("RLScore instances: %d, CPU time: %f, LOO MSE: %f" %(size, dur, e))
        start = time.clock()
        e = loo_sklearn(X, y, 1.0)
        dur = time.clock() - start
        print("scikit-learn instances: %d, CPU time: %f, LOO MSE: %f" %(size, dur, e))
        print("******")
