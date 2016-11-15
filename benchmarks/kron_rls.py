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

#Benchmark code for comparing the fast Kronecker RLS algorithm implemented in
#RLScore to a naive implementation based on sklearn. The fast training algorithm
#was described in [1,2].
#    
#    [1] Tapio Pahikkala, Willem Waegeman, Antti Airola, Tapio Salakoski, and Bernard De Baets.
#    Conditional ranking on relational data.
#    Machine Learning and Knowledge Discovery in Databases (ECML PKDD), 2010
#    
#    [2] Tapio Pahikkala, Antti Airola, Michiel Stock, Bernard De Baets, and Willem Waegeman.
#    Efficient regularized least-squares algorithms for conditional ranking on relational data.
#    Machine Learning, 93(2-3):321--356, 2013.

import numpy as np
import time

from rlscore.kernel import GaussianKernel
from rlscore.learner import KronRLS
from sklearn.kernel_ridge import KernelRidge

def random_data(size, n_features):
    np.random.seed(77)
    X1 = np.random.randn(size, n_features)
    X2 = np.random.randn(size, n_features)
    Y = np.random.randn(size**2)
    return X1, X2, Y

if __name__=="__main__":
    #trains Kronecker RLS for different sample sizes
    #comparing CPU time and verifying that the learned
    #dual coefficients are same for both methods
    regparam = 1.0
    for size in [10, 20, 40, 60, 80, 100, 500, 1000, 2000, 4000, 6000]:
        X1, X2, y = random_data(size, 100)
        kernel1 = GaussianKernel(X1, gamma=0.01)
        K1 = kernel1.getKM(X1)
        kernel2 = GaussianKernel(X2, gamma=0.01)
        K2 = kernel2.getKM(X2)
        start = time.clock()
        rls = KronRLS(K1=K1, K2=K2, Y=y, regparam=regparam)
        dur = time.clock() - start
        print("RLScore pairs: %d, CPU time: %f" %(size**2, dur))
        #forming full Kronecker product kernel matrix becomes fast
        #unfeasible
        if size <=100:
            K = np.kron(K2, K1)
            start = time.clock()
            ridge = KernelRidge(alpha=regparam, kernel="precomputed")
            ridge.fit(K, y)
            dur = time.clock() - start
            print("sklearn pairs: %d, CPU time: %f" %(size**2, dur))
            sklearn_coef = ridge.dual_coef_
            core_coef = rls.predictor.A.reshape(K1.shape[0], K2.shape[0]).T.ravel()
            print("Are the coefficients same: %r" %np.allclose(sklearn_coef, core_coef))
        else:
            print("sklearn: too much data")
        print "*****"
