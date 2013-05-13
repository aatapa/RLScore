import numpy as np

from rlscore.measure.auc_measure import *
from rlscore.test.test_measure.abstract_measure_test import AbstractMultiTaskMeasureTest


class Test(AbstractMultiTaskMeasureTest):
    
    def setUp(self):
        AbstractMultiTaskMeasureTest.setUp(self)
        self.func = auc
        self.func_singletask = auc_singletask
        self.func_multitask = auc_multitask
        
    def testFastAUC(self):
        #Tests that the n*log(n) AUC gives same results as simple reference
        #implementation
        y = self.Y[:,0]
        p = self.P[:,0]
        perf = self.func(y,p)
        perf2 = slowAUC(y,p)
        self.assertAlmostEqual(perf, perf2)

       
def slowAUC(y, p):
    #quadratic time AUC computation
    assert y.shape == p.shape
    pos, neg = 0., 0.
    posindices = []
    negindices = []
    for i in range(y.shape[0]):
        if y[i,0] > 0:
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
                if p[i,0] > p[j,0]:
                    auc += 1.
                elif p[i,0] == p[j,0]:
                    auc += 0.5
        auc /= pos * neg
        return auc
