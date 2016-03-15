import numpy as np
import unittest

from rlscore.measure.auc_measure import auc
from rlscore.measure.measure_utilities import UndefinedPerformance

def slowAUC(y, p):
    #quadratic time AUC computation
    assert y.shape == p.shape
    pos, neg = 0., 0.
    posindices = []
    negindices = []
    for i in range(y.shape[0]):
        if y[i] > 0:
            pos += 1.
            posindices.append(i)
        else:
            neg += 1
            negindices.append(i)
    if pos == 0 or neg == 0:
        return np.nan
    else:
        auc = 0.
        for i in posindices:
            for j in negindices:
                if p[i] > p[j]:
                    auc += 1.
                elif p[i] == p[j]:
                    auc += 0.5
        auc /= pos * neg
        return auc

class Test(unittest.TestCase):
        
    def testFastAUC(self):
        #Tests that the n*log(n) AUC gives same results as simple reference
        #implementation
        n = 100
        Y = np.random.rand(n)
        #basic case
        Y = np.where(Y>0.5, 1, -1)
        P = np.random.rand(n)-0.5
        perf = auc(Y,P)
        perf2 = slowAUC(Y,P)
        self.assertAlmostEqual(perf, perf2)
        #all zeros
        P2 = np.zeros(Y.shape)
        perf = auc(Y, P2)
        self.assertEqual(perf, 0.5)
        #multiple columns
        Y = np.random.rand(n, 3)
        Y = np.where(Y>0.5, 1, -1)
        P = np.random.rand(n,3)-0.5
        perf1 = []
        for i in range(Y.shape[1]):
            perf1.append(slowAUC(Y[:,i], P[:,i]))
        perf1 = np.mean(perf1)
        perf2 = auc(Y, P)
        #Only values 1 and -1 should be allowed
        self.assertAlmostEqual(perf1, perf2)
        Y = [1,0,0]
        self.assertRaises(UndefinedPerformance, auc, Y, Y)
        #Y and P must be of same length
        Y = [1, -1]
        P = [1, 2, 3]
        self.assertRaises(UndefinedPerformance, auc, Y, P)
        #AUC is undefined, if all Y-values are the same
        Y = [1, 1, 1]
        self.assertRaises(UndefinedPerformance, auc, Y, P)
        

       

