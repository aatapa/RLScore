from numpy import *
random.seed(100)
import numpy.linalg as la
import unittest
from numpy.testing import assert_allclose, assert_array_equal
from rlscore.learner import GreedyRLS
from rlscore.learner import RLS
import numpy as np

    
    
def speedtest():
    tsize, fsize = 3000, 3000
    desiredfcount = 5
    Xtrain = mat(random.rand(fsize, tsize), dtype=float64)
    bias = 2.
    rp = 1.
    ylen = 2
    Y = mat(random.rand(tsize, ylen), dtype=float64)
    
    rpool = {}
    class TestCallback(object):
        def callback(self, learner):
            print('round')
        def finished(self, learner):
            pass
    tcb = TestCallback()
    rpool['callback'] = tcb
    rpool['X'] = Xtrain.T
    rpool['Y'] = Y
    
    rpool['subsetsize'] = str(desiredfcount)
    rpool['regparam'] = rp
    rpool['bias'] = bias
    grls = GreedyRLS(**rpool)
    
    print(grls.selected)
    print(grls.A[grls.selected])
    print(grls.b)



class Test(unittest.TestCase):

    def setUp(self):
        np.random.seed(100)
        m= 30
        self.Xtrain1 = np.random.rand(m, 20)
        self.Xtrain2 = np.random.rand(m, 40)
        self.Ytrain1 = np.random.randn(m)
        self.Ytrain2 = np.random.randn(m, 5)
    
    def test_compare(self):
        for X in [self.Xtrain1, self.Xtrain2]:
            for Y in [self.Ytrain1, self.Ytrain2]:
                #No bias
                greedy_rls = GreedyRLS(X, Y, subsetsize = 10, regparam=12, bias = 0.)
                selected = greedy_rls.selected
                s_complement = list(set(range(X.shape[1])).difference(selected))
                X_cut = X[:,selected]
                rls = RLS(X_cut, Y, regparam=12., bias = 0.)
                W = greedy_rls.predictor.W[selected]
                W2 = rls.predictor.W
                assert_allclose(W, W2)
                assert_array_equal(greedy_rls.predictor.W[s_complement], 0)
                assert_array_equal(greedy_rls.predictor.b, 0)
                #Bias
                greedy_rls = GreedyRLS(X, Y, subsetsize = 10, regparam=12, bias = 2.)
                selected = greedy_rls.selected
                X_cut = X[:,selected]
                rls = RLS(X_cut, Y, regparam=12., bias = 2.)
                W = greedy_rls.predictor.W[selected]
                W2 = rls.predictor.W
                assert_allclose(W, W2)
                assert_allclose(greedy_rls.predictor.b, rls.predictor.b)

    
    def testRLS(self):
        print("\n\n\n\nTesting the correctness of the GreedyRLS module.\n\n")
        tsize, fsize = 10, 30
        desiredfcount = 5
        Xtrain = mat(random.rand(fsize, tsize), dtype=float64)
        bias = 2.
        bias_slice = sqrt(bias)*mat(ones((1,Xtrain.shape[1]), dtype=float64))
        Xtrain_biased = vstack([Xtrain,bias_slice])
        ylen = 2
        Y = mat(random.rand(tsize, ylen), dtype=float64)
        selected = []
        rp = 1.
        currentfcount=0
        while currentfcount < desiredfcount:
            selected_plus_bias = selected + [fsize]
            bestlooperf = 9999999999.
            for ci in range(fsize):
                if ci in selected_plus_bias: continue
                updK = Xtrain_biased[selected_plus_bias+[ci]].T*Xtrain_biased[selected_plus_bias+[ci]]
                looperf = 0.
                for hi in range(tsize):
                    hoinds = list(range(0, hi)) + list(range(hi + 1, tsize))
                    updcutK = updK[ix_(hoinds, hoinds)]
                    updcrossK = updK[ix_([hi], hoinds)]
                    loopred = updcrossK * la.inv(updcutK + rp * mat(eye(tsize-1))) * Y[hoinds]
                    looperf += mean(multiply((loopred - Y[hi]), (loopred - Y[hi])))
                if looperf < bestlooperf:
                    bestcind = ci
                    bestlooperf = looperf
                print('Tester ', ci, looperf)
            selected.append(bestcind)
            print('Tester ', selected)
            currentfcount += 1
        selected_plus_bias = selected + [fsize]
        K = Xtrain_biased[selected_plus_bias].T*Xtrain_biased[selected_plus_bias]
        G = la.inv(K+rp * mat(eye(tsize)))
        A = Xtrain_biased[selected_plus_bias]*G*Y
        print('Tester ', A)
        rpool = {}
        class TestCallback(object):
            def callback(self, learner):
                print('GreedyRLS', learner.looperf.T)
                pass
            def finished(self, learner):
                pass
        tcb = TestCallback()
        rpool['callback'] = tcb
        rpool['X'] = Xtrain.T
        rpool['Y'] = Y
        rpool['subsetsize'] = desiredfcount
        rpool['regparam'] = rp
        rpool['bias'] = bias
        grls = GreedyRLS(**rpool)
        assert_array_equal(selected, grls.selected)
        assert_allclose(A[:-1], grls.A[selected])
        assert_allclose(np.sqrt(bias)*A[-1], grls.b)

        

if __name__=="__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(Test)
    unittest.TextTestRunner(verbosity=2).run(suite)

