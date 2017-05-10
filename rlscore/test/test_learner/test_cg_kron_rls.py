import unittest
import random as pyrandom
pyrandom.seed(100)

import numpy as np
from rlscore.kernel import LinearKernel
from rlscore.learner.cg_kron_rls import CGKronRLS
from rlscore.predictor import LinearPairwisePredictor
from rlscore.predictor import KernelPairwisePredictor
from rlscore.learner.rls import RLS


class Test(unittest.TestCase):
    
    def setUp(self):
        np.random.seed(55)
    
    
    def generate_data(self, poscount, negcount, dim, mean1, mean2):
        #Generates a standard binary classification data set,
        #with poscount+negcount instances. Data is normally
        #distributed, with mean1 for positive class,
        #mean2 for negative class and unit variance
        X_pos = np.random.randn(poscount, dim)+mean1
        X_neg = np.random.randn(negcount, dim)+mean2
        X = np.vstack((X_pos, X_neg))
        Y = np.vstack((np.ones((poscount, 1)), -1.*np.ones((negcount,1))))
        perm = np.random.permutation(range(poscount+negcount))
        X = X[perm]
        Y = Y[perm]
        return X, Y
    
    
    def generate_xortask(self):
        np.random.seed(55)
        trainpos1 = 5
        trainneg1 = 5
        trainpos2 = 6
        trainneg2 = 7
        X_train1, Y_train1 = self.generate_data(trainpos1, trainneg1, 5, 0, 1)
        X_train2, Y_train2 = self.generate_data(trainpos2, trainneg2, 5, 4, 6)
        
        testpos1 = 6
        testneg1 = 7
        testpos2 = 5
        testneg2 = 8
        X_test1, Y_test1 = self.generate_data(testpos1, testneg1, 5, 0, 1)
        X_test2, Y_test2 = self.generate_data(testpos2, testneg2, 5, 4, 6)
        
        #kernel1 = GaussianKernel.createKernel(gamma=0.01, X=X_train1)
        kernel1 = LinearKernel(X_train1, bias=0.0)
        K_train1 = kernel1.getKM(X_train1)
        K_test1 = kernel1.getKM(X_test1)
        
        #kernel2 = GaussianKernel.createKernel(gamma=0.01, X=X_train2)
        kernel2 = LinearKernel(X_train2, bias=0.0)
        K_train2 = kernel2.getKM(X_train2)
        K_test2 = kernel2.getKM(X_test2)
        
        #The function to be learned is a xor function on the class labels
        #of the two classification problems
        Y_train = -1.*np.ones((trainpos1+trainneg1, trainpos2+trainneg2))
        for i in range(trainpos1+trainneg1):
            for j in range(trainpos2+trainneg2):
                if Y_train1[i,0] != Y_train2[j,0]:
                    Y_train[i, j] = 1.
        
        Y_test = -1.*np.ones((testpos1+testneg1, testpos2+testneg2))    
        for i in range(testpos1+testneg1):
            for j in range(testpos2+testneg2):
                if Y_test1[i,0] != Y_test2[j,0]:
                    Y_test[i, j] = 1.
        
        return K_train1, K_train2, Y_train, K_test1, K_test2, Y_test, X_train1, X_train2, X_test1, X_test2
    
    
    def test_cg_kron_rls(self):
        
        regparam = 0.0001
        
        K_train1, K_train2, Y_train, K_test1, K_test2, Y_test, X_train1, X_train2, X_test1, X_test2 = self.generate_xortask()
        Y_train = Y_train.ravel(order = 'F')
        Y_test = Y_test.ravel(order = 'F')
        train_rows, train_columns = K_train1.shape[0], K_train2.shape[0]
        test_rows, test_columns = K_test1.shape[0], K_test2.shape[0]
        rowstimescols = train_rows * train_columns
        allindices = np.arange(rowstimescols)
        all_label_row_inds, all_label_col_inds = np.unravel_index(allindices, (train_rows, train_columns), order = 'F') 
        incinds = np.random.choice(allindices, 50, replace=False)
        label_row_inds, label_col_inds = all_label_row_inds[incinds], all_label_col_inds[incinds] 
        Y_train_known_outputs = Y_train.reshape(rowstimescols, order = 'F')[incinds]
        
        #Train an ordinary RLS regressor for reference
        params = {}
        params["X"] = np.kron(K_train2, K_train1)[np.ix_(incinds, incinds)]
        params["kernel"] = "PrecomputedKernel"
        params["Y"] = Y_train_known_outputs
        params["regparam"] = regparam
        ordrls_learner = RLS(**params)
        ordrls_model = ordrls_learner.predictor
        K_Kron_test = np.kron(K_test2, K_test1)[:, incinds]
        ordrls_testpred = ordrls_model.predict(K_Kron_test)
        ordrls_testpred = ordrls_testpred.reshape((test_rows, test_columns), order = 'F')
        
        #Train linear Kronecker RLS
        class TestCallback():
            def __init__(self):
                self.round = 0
            def callback(self, learner):
                self.round = self.round + 1
                tp = LinearPairwisePredictor(learner.W).predict(X_test1, X_test2)
                print(str(self.round) + ' ' + str(np.mean(np.abs(tp - ordrls_testpred.ravel(order = 'F')))))
            def finished(self, learner):
                print('finished')
        params = {}
        params["regparam"] = regparam
        params["X1"] = X_train1
        params["X2"] = X_train2
        params["Y"] = Y_train_known_outputs
        params["label_row_inds"] = label_row_inds
        params["label_col_inds"] = label_col_inds
        tcb = TestCallback()
        params['callback'] = tcb
        linear_kron_learner = CGKronRLS(**params)
        linear_kron_testpred = linear_kron_learner.predict(X_test1, X_test2).reshape((test_rows, test_columns), order = 'F')
        linear_kron_testpred_alt = linear_kron_learner.predict(X_test1, X_test2, [0, 0, 1], [0, 1, 0])
        
        #Train kernel Kronecker RLS
        params = {}
        params["regparam"] = regparam
        params["K1"] = K_train1
        params["K2"] = K_train2
        params["Y"] = Y_train_known_outputs
        params["label_row_inds"] = label_row_inds
        params["label_col_inds"] = label_col_inds
        class KernelCallback():
            def __init__(self):
                self.round = 0
            def callback(self, learner):
                self.round = self.round + 1
                tp = KernelPairwisePredictor(learner.A, learner.input1_inds, learner.input2_inds).predict(K_test1, K_test2)
                print(str(self.round) + ' ' + str(np.mean(np.abs(tp - ordrls_testpred.ravel(order = 'F')))))
            def finished(self, learner):
                print('finished')
        tcb = KernelCallback()
        params['callback'] = tcb
        kernel_kron_learner = CGKronRLS(**params)
        kernel_kron_testpred = kernel_kron_learner.predict(K_test1, K_test2).reshape((test_rows, test_columns), order = 'F')
        kernel_kron_testpred_alt = kernel_kron_learner.predict(K_test1, K_test2, [0, 0, 1], [0, 1, 0])
        
        print('Predictions: Linear CgKronRLS, Kernel CgKronRLS, ordinary RLS')
        print('[0, 0]: ' + str(linear_kron_testpred[0, 0]) + ' ' + str(kernel_kron_testpred[0, 0]) + ' ' + str(ordrls_testpred[0, 0]))#, linear_kron_testpred_alt[0], kernel_kron_testpred_alt[0]
        print('[0, 1]: ' + str(linear_kron_testpred[0, 1]) + ' ' + str(kernel_kron_testpred[0, 1]) + ' ' + str(ordrls_testpred[0, 1]))#, linear_kron_testpred_alt[1], kernel_kron_testpred_alt[1]
        print('[1, 0]: ' + str(linear_kron_testpred[1, 0]) + ' ' + str(kernel_kron_testpred[1, 0]) + ' ' + str(ordrls_testpred[1, 0]))#, linear_kron_testpred_alt[2], kernel_kron_testpred_alt[2]
        print('Meanabsdiff: linear KronRLS - ordinary RLS, kernel KronRLS - ordinary RLS')
        print(str(np.mean(np.abs(linear_kron_testpred - ordrls_testpred))) + ' ' + str(np.mean(np.abs(kernel_kron_testpred - ordrls_testpred))))
        np.testing.assert_almost_equal(linear_kron_testpred, ordrls_testpred, decimal=5)
        np.testing.assert_almost_equal(kernel_kron_testpred, ordrls_testpred, decimal=4)
        
        #Train multiple kernel Kronecker RLS
        params = {}
        params["regparam"] = regparam
        params["K1"] = [K_train1, K_train1]
        params["K2"] = [K_train2, K_train2]
        params["weights"] = [1. / 3, 2. / 3]
        params["Y"] = Y_train_known_outputs
        params["label_row_inds"] = label_row_inds
        params["label_col_inds"] = label_col_inds
        class KernelCallback():
            def __init__(self):
                self.round = 0
            def callback(self, learner):
                self.round = self.round + 1
                tp = KernelPairwisePredictor(learner.A, learner.input1_inds, learner.input2_inds).predict(K_test1, K_test2)
                print(str(self.round) + ' ' + str(np.mean(np.abs(tp - ordrls_testpred.ravel(order = 'F')))))
            def finished(self, learner):
                print('finished')
        tcb = KernelCallback()
        params['callback'] = tcb
        kernel_kron_learner = CGKronRLS(**params)
        kernel_kron_testpred = kernel_kron_learner.predict([K_test1, K_test1], [K_test2, K_test2]).reshape((test_rows, test_columns), order = 'F')
        #kernel_kron_testpred_alt = kernel_kron_learner.predict(K_test1, K_test2, [0, 0, 1], [0, 1, 0])
        
        print('Predictions: Linear CgKronRLS, Kernel CgKronRLS, ordinary RLS')
        print('[0, 0]: ' + str(linear_kron_testpred[0, 0]) + ' ' + str(kernel_kron_testpred[0, 0]) + ' ' + str(ordrls_testpred[0, 0]))#, linear_kron_testpred_alt[0], kernel_kron_testpred_alt[0]
        print('[0, 1]: ' + str(linear_kron_testpred[0, 1]) + ' ' + str(kernel_kron_testpred[0, 1]) + ' ' + str(ordrls_testpred[0, 1]))#, linear_kron_testpred_alt[1], kernel_kron_testpred_alt[1]
        print('[1, 0]: ' + str(linear_kron_testpred[1, 0]) + ' ' + str(kernel_kron_testpred[1, 0]) + ' ' + str(ordrls_testpred[1, 0]))#, linear_kron_testpred_alt[2], kernel_kron_testpred_alt[2]
        print('Meanabsdiff: linear KronRLS - ordinary RLS, kernel KronRLS - ordinary RLS')
        print(str(np.mean(np.abs(linear_kron_testpred - ordrls_testpred))) + ' ' + str(np.mean(np.abs(kernel_kron_testpred - ordrls_testpred))))
        np.testing.assert_almost_equal(linear_kron_testpred, ordrls_testpred, decimal=5)
        np.testing.assert_almost_equal(kernel_kron_testpred, ordrls_testpred, decimal=4)


if __name__=="__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(Test)
    unittest.TextTestRunner(verbosity=2).run(suite)

