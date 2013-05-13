import unittest

import numpy as np


class AbstractMeasureTest(unittest.TestCase):
    
    def setUp(self):
        np.random.seed(10)
        m = 50
        n = 10
        P = 2*np.random.randn(m,n)
        self.Y = -1.*np.ones((m,n))
        indices = np.random.randint(0,10,m)
        for i,j in enumerate(indices):
            self.Y[i,j] = 1
        self.P = P+self.Y
        self.Y = np.mat(self.Y)
        self.P = np.mat(self.P)
        #some ties added
        self.P[3,0] = self.P[7,0]
        self.P[3,0] = self.P[10,0]
        #and a zerp prediction
        self.P[14,0] = 0.
    
    def testArray_vs_Mat(self):
        #A simple one vs. all classification test with
        #100 predictions and 10 classes
        Y = self.Y
        P = self.P
        Y_a = Y.getA()
        P_a = P.getA()
        p1 = self.func(Y,P)
        p2 = self.func(Y_a, P_a)
        np.testing.assert_array_equal(p1, p2)

class AbstractMultiTaskMeasureTest(AbstractMeasureTest):
    
    def setUp(self):
        AbstractMeasureTest.setUp(self)
    
    def testMultiTask(self):
        perfs = self.func_multitask(self.Y, self.P)
        for i in range(self.Y.shape[1]):
            y = self.Y[:,i]
            p = self.P[:,i]
            perf = self.func_singletask(y,p)
            self.assertAlmostEqual(perfs[i], perf)