import unittest
import numpy as np

from rlscore.measure.accuracy_measure import accuracy
from rlscore.measure.measure_utilities import UndefinedPerformance

def simple_accuracy(Y, P):
    correct = 0.
    for i in range(len(Y)):
        if Y[i] == 1 and P[i] > 0:
            correct += 1
        elif Y[i] == -1 and P[i] < 0:
            correct += 1
        elif P[i] == 0:
            correct += 0.5
    return correct / len(Y)

    

class Test(unittest.TestCase):
        
    def test_accuracy(self):
        n = 100
        Y = np.random.rand(n)
        #basic case
        Y = np.where(Y>0.5, 1, -1)
        P = np.random.rand(n)-0.5
        acc1 = simple_accuracy(Y, P)
        acc2 = accuracy(Y, P)
        self.assertAlmostEqual(acc1, acc2)
        #all zeros
        P2 = np.zeros(Y.shape)
        acc = accuracy(Y, P2)
        self.assertEqual(acc, 0.5)
        #multiple columns
        Y = np.random.rand(n, 3)
        Y = np.where(Y>0.5, 1, -1)
        P = np.random.rand(n,3)-0.5
        acc1 = []
        for i in range(Y.shape[1]):
            acc1.append(simple_accuracy(Y[:,i], P[:,i]))
        acc1 = np.mean(acc1)
        acc2 = accuracy(Y, P)
        #Only values 1 and -1 should be allowed
        self.assertAlmostEqual(acc1, acc2)
        Y = [1,0,0]
        self.assertRaises(UndefinedPerformance, accuracy, Y, Y)
        #Y and P must be of same length
        Y = [1, -1]
        P = [1, 2, 3]
        self.assertRaises(UndefinedPerformance, accuracy, Y, P)