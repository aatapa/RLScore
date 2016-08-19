import numpy as np
import unittest

from rlscore.measure.multi_accuracy_measure import ova_accuracy
from rlscore.utilities import multiclass

class Test(unittest.TestCase):
    
    def test(self):
        Y = np.random.randint(0, 5, 100)
        P = np.random.random((100, 5))
        Y_ova = multiclass.to_one_vs_all(Y)
        perf1 = ova_accuracy(Y_ova, P)
        P = np.argmax(P, axis=1)
        perf2 = np.mean(Y == P)
        self.assertAlmostEqual(perf1, perf2)
