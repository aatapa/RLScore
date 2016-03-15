import numpy as np
import unittest

from rlscore.measure.sqerror_measure import sqerror

def simple_sqerror(Y, P):
    e = 0.
    for i in range(len(Y)):
        e += (Y[i] - P[i])**2
    return e/len(Y)

class Test(unittest.TestCase):
    
    def test_sqerror(self):
        n = 100
        Y = np.random.rand(n)
        P = np.random.rand(n)
        e1 = simple_sqerror(Y, P)
        e2 = sqerror(Y, P)
        self.assertAlmostEqual(e1, e2)
        #multiple columns
        Y = np.random.rand(n, 3)
        P = np.random.rand(n,3)
        e1 = []
        for i in range(Y.shape[1]):
            e1.append(simple_sqerror(Y[:,i], P[:,i]))
        e1 = np.mean(e1)
        e2 = sqerror(Y, P)
        self.assertAlmostEqual(e1, e2)
        
        
