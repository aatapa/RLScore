import numpy as np
import unittest

from rlscore.measure.cindex_measure import cindex
from rlscore.measure.measure_utilities import UndefinedPerformance

def slow_cindex(Y, P):
    correct = Y
    predictions = P
    assert len(correct) == len(predictions)
    disagreement = 0.
    decisions = 0.
    for i in range(len(correct)):
        for j in range(len(correct)):
                if correct[i] > correct[j]:
                    decisions += 1.
                    if predictions[i] < predictions[j]:
                        disagreement += 1.
                    elif predictions[i] == predictions[j]:
                        disagreement += 0.5
    #Disagreement error is not defined for cases where there
    #are no disagreeing pairs
    disagreement /= decisions
    return 1. - disagreement

class Test(unittest.TestCase):
    

    def testCindex(self):
        y = np.random.random(100)
        p = np.random.random(100)
        perf = cindex(y,p)
        perf2 = slow_cindex(y,p)
        self.assertAlmostEqual(perf, perf2)
        y = np.random.random(10000)
        p = np.ones(10000)
        self.assertEqual(cindex(y,p), 0.5)
        #9 pairs
        y = np.array([1,2,3,3,4])
        p = np.array([-4,1,5,5,7])
        #0 inversions
        self.assertEqual(cindex(y,p), 1.0)
        #1 inversion
        p = np.array([-4,1,8,5,7])
        self.assertAlmostEqual(cindex(y,p), 8./9.)
        #1.5 inversions
        p = np.array([-4,1,8,7,7])
        self.assertAlmostEqual(cindex(y,p), 7.5/9.)
        #all wrong
        p = np.array([10,9,8,7,6])
        self.assertEqual(cindex(y,p), 0.)
        #all tied
        p = np.array([10,10,10,10,10])
        self.assertEqual(cindex(y,p), 0.5)
        self.assertRaises(UndefinedPerformance, cindex, p, p)


