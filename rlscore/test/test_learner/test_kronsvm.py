import unittest
import random as pyrandom


import numpy as np
from numpy import random as numpyrandom
from rlscore.learner.kron_svm import KronSVM

from rlscore.utilities import sampled_kronecker_products

def dual_svm_objective(a, K1, K2, Y, rowind, colind, lamb):
    #dual form of the objective function for support vector machine
    #a: current dual solution
    #K1: samples x samples kernel matrix for domain 1
    #K2: samples x samples kernel matrix for domain 2
    #rowind: row indices for training pairs
    #colind: column indices for training pairs
    #lamb: regularization parameter
    P =  sampled_kronecker_products.sampled_vec_trick(a, K2, K1, colind, rowind, colind, rowind)
    z = (1. - Y*P)
    z = np.where(z>0, z, 0)
    Ka = sampled_kronecker_products.sampled_vec_trick(a, K2, K1, colind, rowind, colind, rowind)
    return 0.5*(np.dot(z,z)+lamb*np.dot(a, Ka))

def primal_svm_objective(v, X1, X2, Y, rowind, colind, lamb):
    P = sampled_kronecker_products.sampled_vec_trick(v, X2, X1, colind, rowind)
    z = (1. - Y*P)
    #print z
    z = np.where(z>0, z, 0)
    #return np.dot(z,z)
    return 0.5*(np.dot(z,z)+lamb*np.dot(v,v))

def load_data(primal=True, fold_index=0):
    X1 = np.random.rand(20, 300)
    X2 = np.random.rand(10, 200)
    dfold = [1,2,4,5]
    tfold = [0,6,7]
    Y = np.random.randn(X1.shape[0], X2.shape[0])
    Y = np.where(Y>=0, 1., -1.)
    dtraininds = list(set(range(Y.shape[0])).difference(dfold))
    ttraininds = list(set(range(Y.shape[1])).difference(tfold))    
    X1_train = X1[dtraininds, :]
    X2_train = X2[ttraininds, :]
    X1_test = X1[dfold,:]
    X2_test = X2[tfold,:]
    KT = np.mat(X2)
    KT = KT * KT.T
    KD = np.mat(X1)
    KD = KD * KD.T
    K1_train = KD[np.ix_(dtraininds, dtraininds)]
    K2_train = KT[np.ix_(ttraininds, ttraininds)]
    Y_train = Y[np.ix_(dtraininds, ttraininds)]
    K1_test = KD[np.ix_(dfold,dtraininds)]
    K2_test = KT[np.ix_(tfold,ttraininds)]
    Y_test = Y[np.ix_(dfold, tfold)]
    ssize = int(Y_train.shape[0]*Y_train.shape[1]*0.25)
    rows = numpyrandom.random_integers(0, K1_train.shape[0]-1, ssize)
    cols = numpyrandom.random_integers(0, K2_train.shape[0]-1, ssize)
    ind = np.ravel_multi_index([rows, cols], (K1_train.shape[0], K2_train.shape[0]))
    Y_train = Y_train.ravel()[ind]
    Y_test = Y_test.ravel(order='F')
    if primal:
        return X1_train, X2_train, Y_train, rows, cols, X1_test, X2_test, Y_test
    else:
        return K1_train, K2_train, Y_train, rows, cols, K1_test, K2_test, Y_test

class Test(unittest.TestCase):
    
    def setUp(self):
        np.random.seed(55)
    
    

    def test_kronsvm(self):
        
        regparam = 0.01
        pyrandom.seed(100)
        numpyrandom.seed(100)      
        X1_train, X2_train, Y_train, rows, cols, X1_test, X2_test, Y_test = load_data(primal=True)

        
        class PrimalCallback(object):
            def __init__(self):
                self.iter = 0
            def callback(self, learner):
                X1 = learner.resource_pool['X1']
                X2 = learner.resource_pool['X2']
                rowind = learner.label_row_inds
                colind = learner.label_col_inds
                w = learner.W.ravel(order='F')
                loss = primal_svm_objective(w, X1, X2, Y_train, rowind, colind, regparam)
                print("iteration", self.iter)
                print("Primal SVM loss", loss)
                self.iter += 1
            def finished(self, learner):
                pass        
        params = {}
        params["X1"] = X1_train
        params["X2"] = X2_train
        params["Y"] = Y_train
        params["label_row_inds"] = rows
        params["label_col_inds"] = cols
        params["maxiter"] = 100
        params["inneriter"] = 100
        params["regparam"] = regparam
        params['callback'] = PrimalCallback()  
        learner = KronSVM(**params)
        P_linear = learner.predictor.predict(X1_test, X2_test)
        pyrandom.seed(100)
        numpyrandom.seed(100)         
        K1_train, K2_train, Y_train, rows, cols, K1_test, K2_test, Y_test = load_data(primal=False)       
        
        class DualCallback(object):
            def __init__(self):
                self.iter = 0
                self.atol = None
    
            def callback(self, learner):
                K1 = learner.resource_pool['K1']
                K2 = learner.resource_pool['K2']
                rowind = learner.label_row_inds
                colind = learner.label_col_inds
                #loss = dual_svm_objective(learner.A, K1, K2, Y_train, rowind, colind, regparam)
                #loss = learner.bestloss
                print("iteration", self.iter)
                #print("Dual SVM loss", loss)
                #model = learner.predictor
                self.iter += 1
            def finished(self, learner):
                pass
        params = {}
        params["K1"] = K1_train
        params["K2"] = K2_train
        params["Y"] = Y_train
        params["label_row_inds"] = rows
        params["label_col_inds"] = cols
        params["maxiter"] = 100
        params["inneriter"] = 100
        params["regparam"] = regparam
        params['callback'] = DualCallback()
        learner = KronSVM(**params)
        P_dual = learner.predictor.predict(K1_test, K2_test)
        print(np.max(1. - np.abs(P_linear / P_dual)))
        assert np.max(1. - np.abs(P_linear / P_dual)) < 0.001
        
        '''
        params = {}
        params["K1"] = [K1_train, K1_train]
        params["K2"] = [K2_train, K2_train]
        #params["weights"] = [1, 1]
        params["weights"] = [1. / 3, 2. / 3]
        #params["weights"] = [1.]# / np.sqrt(3), np.sqrt(2. / 3)]
        params["Y"] = Y_train
        params["label_row_inds"] = rows
        params["label_col_inds"] = cols
        params["maxiter"] = 100
        params["inneriter"] = 100
        params["regparam"] = regparam
        params['callback'] = DualCallback()
        learner = KronSVM(**params)
        P_dual = learner.predictor.predict([K1_test, K1_test], [K2_test, K2_test])
        print(np.max(1. - np.abs(P_linear / P_dual)))
        assert np.max(1. - np.abs(P_linear / P_dual)) < 0.001
        '''
        




if __name__=="__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(Test)
    unittest.TextTestRunner(verbosity=2).run(suite)
