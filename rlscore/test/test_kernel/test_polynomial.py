"""This module contains the test functions for the kernels"""
import unittest

import numpy as np

from rlscore.kernel import PolynomialKernel
from rlscore.test.test_kernel.abstract_kernel_test import AbstractKernelTest

class Test(AbstractKernelTest):
        
    def setParams(self):
        self.kernel = PolynomialKernel
        self.paramsets = [{"bias":0., "gamma":1., "coef0":0, "degree":1},
                          {"bias":2., "gamma":2., "coef0":1, "degree":3},
                          {"bias":1., "gamma":3., "coef0":0., "degree":4},
                          {"bias":0., "gamma":10, "coef0":3, "degree":2}]
        
    def k_func(self, x1, x2, params):
        #linear kernel is simply the dot product, optionally
        #with a bias parameter
        bias = params["bias"]
        gamma = params["gamma"]
        degree = params["degree"]
        coef0 = params["coef0"]
        return (gamma * np.dot(x1,x2) +coef0) ** degree + bias

