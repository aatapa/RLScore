"""This module contains the test functions for the kernels"""
import unittest

import numpy as np

from rlscore.kernel import LinearKernel
from rlscore.test.test_kernel.abstract_kernel_test import AbstractKernelTest


class Test(AbstractKernelTest):
        
    def setParams(self):
        self.kernel = LinearKernel
        self.paramsets = [{"bias":0.}, {"bias":3.}]
        
    def k_func(self, x1, x2, params):
        #linear kernel is simply the dot product, optionally
        #with a bias parameter
        bias = params["bias"]
        return np.dot(x1, x2)+bias
        