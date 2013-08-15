"""This module contains the test functions for the kernels"""
import unittest

import numpy as np

from rlscore.kernel import GaussianKernel
from rlscore.test.test_kernel.abstract_kernel_test import AbstractKernelTest


class Test(AbstractKernelTest):
        
    def setParams(self):
        self.kernel = GaussianKernel
        self.paramsets = [{"bias":0., "gamma":1}, {"bias":3., "gamma":1},
                          {"bias":1, "gamma":100}, {"bias":0, "gamma":0.0001}]
        
    def k_func(self, x1, x2, params):
        #linear kernel is simply the dot product, optionally
        #with a bias parameter
        bias = params["bias"]
        gamma = params["gamma"]
        f = x1-x2
        return np.exp(-gamma * np.dot(f,f))+bias
                            