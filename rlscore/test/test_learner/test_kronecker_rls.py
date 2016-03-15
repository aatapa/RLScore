import unittest

import numpy as np
from rlscore.kernel import LinearKernel
from rlscore.learner.kron_rls import KronRLS
from rlscore.learner.rls import RLS
from rlscore.learner.query_rankrls import QueryRankRLS


class Test(unittest.TestCase):
    
    def setUp(self):
        np.random.seed(55)
        #random.seed(100)
    
    
    def generate_data(self, poscount, negcount, dim, mean1, mean2):
        #Generates a standard binary classification data set,
        #with poscount+negcount instances. Data is normally
        #distributed, with mean1 for positive class,
        #mean2 for negative class and unit variance
        X_pos = np.random.randn(poscount, dim) + mean1
        X_neg = np.random.randn(negcount, dim) + mean2
        X = np.vstack((X_pos, X_neg))
        Y = np.vstack((np.ones((poscount, 1)), -1. * np.ones((negcount, 1))))
        perm = np.random.permutation(range(poscount + negcount))
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
        
        testpos1 = 26
        testneg1 = 27
        testpos2 = 25
        testneg2 = 25
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
    
    
    def test_kron_rls(self):
        
        regparam = 0.001
        
        K_train1, K_train2, Y_train, K_test1, K_test2, Y_test, X_train1, X_train2, X_test1, X_test2 = self.generate_xortask()
        Y_train = Y_train.ravel(order = 'F')
        Y_test = Y_test.ravel(order = 'F')
        train_rows, train_columns = K_train1.shape[0], K_train2.shape[0]
        test_rows, test_columns = K_test1.shape[0], K_test2.shape[0]
        trainlabelcount = train_rows * train_columns
        
        #Train linear Kronecker RLS with data-matrices
        params = {}
        params["regparam"] = regparam
        params["X1"] = X_train1
        params["X2"] = X_train2
        params["Y"] = Y_train
        linear_kron_learner = KronRLS(**params)
        linear_kron_testpred = linear_kron_learner.predict(X_test1, X_test2).reshape((test_rows, test_columns), order = 'F')
        
        #Train kernel Kronecker RLS with pre-computed kernel matrices
        params = {}
        params["regparam"] = regparam
        params["K1"] = K_train1
        params["K2"] = K_train2
        params["Y"] = Y_train
        kernel_kron_learner = KronRLS(**params)
        kernel_kron_testpred = kernel_kron_learner.predict(K_test1, K_test2).reshape((test_rows, test_columns), order = 'F')
        
        #Train an ordinary RLS regressor for reference
        K_Kron_train_x = np.kron(K_train2, K_train1)
        params = {}
        params["X"] = K_Kron_train_x
        params["kernel"] = "PrecomputedKernel"
        params["Y"] = Y_train.reshape(trainlabelcount, 1, order = 'F')
        ordrls_learner = RLS(**params)
        ordrls_learner.solve(regparam)
        K_Kron_test_x = np.kron(K_test2, K_test1)
        ordrls_testpred = ordrls_learner.predict(K_Kron_test_x)
        ordrls_testpred = ordrls_testpred.reshape((test_rows, test_columns), order = 'F')
        
        print('')
        print('Prediction: linear KronRLS, kernel KronRLS, ordinary RLS')
        print('[0, 0] ' + str(linear_kron_testpred[0, 0]) + ' ' + str(kernel_kron_testpred[0, 0]) + ' ' + str(ordrls_testpred[0, 0]))
        print('[0, 1] ' + str(linear_kron_testpred[0, 1]) + ' ' + str(kernel_kron_testpred[0, 1]) + ' ' + str(ordrls_testpred[0, 1]))
        print('[1, 0] ' + str(linear_kron_testpred[1, 0]) + ' ' + str(kernel_kron_testpred[1, 0]) + ' ' + str(ordrls_testpred[1, 0]))
        print('Meanabsdiff: linear KronRLS - ordinary RLS, kernel KronRLS - ordinary RLS')
        print(str(np.mean(np.abs(linear_kron_testpred - ordrls_testpred))) + ' ' + str(np.mean(np.abs(kernel_kron_testpred - ordrls_testpred))))
        np.testing.assert_almost_equal(linear_kron_testpred, ordrls_testpred)
        np.testing.assert_almost_equal(kernel_kron_testpred, ordrls_testpred)
        
        print('')
        ordrls_loopred = ordrls_learner.leave_one_out().reshape((train_rows, train_columns), order = 'F')
        linear_kron_loopred = linear_kron_learner.in_sample_loo().reshape((train_rows, train_columns), order = 'F')
        kernel_kron_loopred = kernel_kron_learner.in_sample_loo().reshape((train_rows, train_columns), order = 'F')
        print('In-sample LOO: linear KronRLS, kernel KronRLS, ordinary RLS')
        print('[0, 0] ' + str(linear_kron_loopred[0, 0]) + ' ' + str(kernel_kron_loopred[0, 0]) + ' ' + str(ordrls_loopred[0, 0]))
        print('[0, 1] ' + str(linear_kron_loopred[0, 1]) + ' ' + str(kernel_kron_loopred[0, 1]) + ' ' + str(ordrls_loopred[0, 1]))
        print('[1, 0] ' + str(linear_kron_loopred[1, 0]) + ' ' + str(kernel_kron_loopred[1, 0]) + ' ' + str(ordrls_loopred[1, 0]))
        print('Meanabsdiff: linear KronRLS - ordinary RLS, kernel KronRLS - ordinary RLS')
        print(str(np.mean(np.abs(linear_kron_loopred - ordrls_loopred))) + ' ' + str(np.mean(np.abs(kernel_kron_loopred - ordrls_loopred))))
        np.testing.assert_almost_equal(linear_kron_loopred, ordrls_loopred)
        np.testing.assert_almost_equal(kernel_kron_loopred, ordrls_loopred)
    
    
    def test_conditional_ranking(self):
        
        regparam = 0.001
        
        K_train1, K_train2, Y_train, K_test1, K_test2, Y_test, X_train1, X_train2, X_test1, X_test2 = self.generate_xortask()
        train_rows, train_columns = Y_train.shape
        trainlabelcount = train_rows * train_columns
        
        K_Kron_train_x = np.kron(K_train2, K_train1)
        
        
        #Train linear Conditional Ranking Kronecker RLS
        params = {}
        params["X1"] = X_train1
        params["X2"] = X_train2
        params["Y"] = Y_train
        params["regparam"] = regparam
        linear_kron_condrank_learner = KronRLS(**params)
        linear_kron_condrank_learner.solve_linear_conditional_ranking(regparam)
        
        #Train an ordinary RankRLS for reference
        params = {}
        params["X"] = K_Kron_train_x
        params["kernel"] = "PrecomputedKernel"
        params["Y"] = Y_train.reshape((trainlabelcount, 1), order = 'F')
        qids = []
        for j in range(Y_train.shape[1]):
            for i in range(Y_train.shape[0]):
                qids.append(i)
        params["qids"] = qids
        rankrls_learner = QueryRankRLS(**params)
        rankrls_learner.solve(regparam)
        K_test_x = np.kron(K_test2, K_test1)
        ordrankrls_testpred = rankrls_learner.predict(K_test_x)
        condrank_testpred = linear_kron_condrank_learner.predict(X_test1, X_test2)
        #print('')
        print('\n\nMeanabsdiff: conditional ranking vs rankrls ' + str(np.mean(np.abs(condrank_testpred - ordrankrls_testpred))) + '\n')
        np.testing.assert_almost_equal(condrank_testpred, ordrankrls_testpred)

