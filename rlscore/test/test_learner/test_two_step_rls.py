import unittest

import numpy as np
#from rlscore.measure.cindex_measure import cindex
from rlscore.measure import auc
from rlscore.measure import sqmprank
from rlscore.measure import sqerror
from rlscore.kernel import GaussianKernel
from rlscore.kernel import LinearKernel
from rlscore.learner.kron_rls import KronRLS
from rlscore.learner.two_step_rls import TwoStepRLS
from rlscore.learner.rls import RLS
from rlscore.learner.label_rankrls import LabelRankRLS


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
    
    
    def generate_xortask(self,
            trainpos1 = 5,
            trainneg1 = 5,
            trainpos2 = 6,
            trainneg2 = 7,
            testpos1 = 26,
            testneg1 = 27,
            testpos2 = 25,
            testneg2 = 25
            ):
        
        np.random.seed(55)
        X_train1, Y_train1 = self.generate_data(trainpos1, trainneg1, 5, 0, 1)
        X_train2, Y_train2 = self.generate_data(trainpos2, trainneg2, 5, 4, 6)
        
        X_test1, Y_test1 = self.generate_data(testpos1, testneg1, 5, 0, 1)
        X_test2, Y_test2 = self.generate_data(testpos2, testneg2, 5, 4, 6)
        
        #kernel1 = GaussianKernel.createKernel(gamma=0.01, X=X_train1)
        kernel1 = LinearKernel(X_train1)
        K_train1 = kernel1.getKM(X_train1)
        K_test1 = kernel1.getKM(X_test1)
        
        #kernel2 = GaussianKernel.createKernel(gamma=0.01, train_features=X_train2)
        kernel2 = LinearKernel(X_train2)
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
    
    
    def test_two_step_rls(self):
        
        regparam1 = 0.001
        regparam2 = 10
        
        K_train1, K_train2, Y_train, K_test1, K_test2, Y_test, X_train1, X_train2, X_test1, X_test2 \
            = self.generate_xortask()
        rows, columns = Y_train.shape
        #print K_train1.shape, K_train2.shape, K_test1.shape, K_test2.shape, rows, columns
        trainlabelcount = rows * columns
        indmatrix = np.mat(range(trainlabelcount)).T.reshape(rows, columns)
        
        
#         #Train linear Kronecker RLS with data-matrices
#         params = {}
#         params["regparam"] = regparam
#         params["xmatrix1"] = X_train1
#         params["xmatrix2"] = X_train2
#         params["Y"] = Y_train
#         linear_kron_learner = KronRLS(**params)
#         linear_kron_learner.train()
#         linear_kron_model = linear_kron_learner.getModel()
#         linear_kron_testpred = linear_kron_model.predictWithDataMatrices(X_test1, X_test2)
        
        
        #Train linear two-step RLS with data-matrices
        params = {}
        params["regparam1"] = regparam1
        params["regparam2"] = regparam2
        params["xmatrix1"] = X_train1
        params["xmatrix2"] = X_train2
        params["Y"] = Y_train
        linear_two_step_learner = TwoStepRLS(**params)
        linear_two_step_learner.train()
        linear_two_step_model = linear_two_step_learner.getModel()
        linear_two_step_testpred = linear_two_step_model.predictWithDataMatrices(X_test1, X_test2)
        
        #Train kernel two-step RLS with pre-computed kernel matrices
        params = {}
        params["regparam1"] = regparam1
        params["regparam2"] = regparam2
        params["kmatrix1"] = K_train1
        params["kmatrix2"] = K_train2
        params["Y"] = Y_train
        kernel_two_step_learner = TwoStepRLS(**params)
        kernel_two_step_learner.train()
        kernel_two_step_model = kernel_two_step_learner.getModel()
        kernel_two_step_testpred = kernel_two_step_model.predictWithKernelMatrices(K_test1, K_test2)
        
        #Train ordinary RLS in two steps for a reference
        params = {}
        params["regparam"] = regparam2
        params["X"] = K_train2
        params['kernel'] = 'precomputed'
        params["Y"] = Y_train.T
        ordinary_rls_first_step = RLS(**params)
        ordinary_rls_first_step.train()
        firststeploo = ordinary_rls_first_step.computeLOO().T
        params = {}
        params["regparam"] = regparam1
        params["X"] = K_train1
        params["kernel"] = "precomputed"
        params["Y"] = firststeploo
        ordinary_rls_second_step = RLS(**params)
        ordinary_rls_second_step.train()
        secondsteploo = ordinary_rls_second_step.computeLOO()
        #print 'Basic RLS', secondsteploo[0, 0]
        
        print
        #print type(linear_kron_testpred), type(kernel_kron_testpred), type(ordrls_testpred)
        #print linear_kron_testpred[0, 0], kernel_kron_testpred[0, 0], ordrls_testpred[0, 0]
        #print linear_kron_testpred[0, 1], kernel_kron_testpred[0, 1], ordrls_testpred[0, 1]
        #print linear_kron_testpred[1, 0], kernel_kron_testpred[1, 0], ordrls_testpred[1, 0]
        #print linear_kron_testpred[0, 0], kernel_two_step_testpred[0, 0]
        #print linear_kron_testpred[0, 1], kernel_two_step_testpred[0, 1]
        #print linear_kron_testpred[1, 0], kernel_two_step_testpred[1, 0]
        
        linear_twostepoutofsampleloo = linear_two_step_learner.computeLOO()
        kernel_twostepoutofsampleloo = kernel_two_step_learner.computeLOO()
        
        #Train linear two-step RLS without out-of-sample rows or columns for [0,0]
        params = {}
        params["regparam1"] = regparam1
        params["regparam2"] = regparam2
        params["xmatrix1"] = X_train1[range(1, X_train1.shape[0])]
        params["xmatrix2"] = X_train2[range(1, X_train2.shape[0])]
        params["Y"] = Y_train[np.ix_(range(1, Y_train.shape[0]), range(1, Y_train.shape[1]))]
        linear_kron_learner = TwoStepRLS(**params)
        linear_kron_learner.train()
        linear_kron_model = linear_kron_learner.getModel()
        linear_kron_testpred_00 = linear_kron_model.predictWithDataMatrices(X_train1[0], X_train2[0])
        
        #Train linear two-step RLS without out-of-sample rows or columns for [2,4]
        params = {}
        params["regparam1"] = regparam1
        params["regparam2"] = regparam2
        params["xmatrix1"] = X_train1[[0, 1] + range(3, K_train1.shape[0])]
        params["xmatrix2"] = X_train2[[0, 1, 2, 3] + range(5, K_train2.shape[0])]
        params["Y"] = Y_train[np.ix_([0, 1] + range(3, K_train1.shape[0]), [0, 1, 2, 3] + range(5, K_train2.shape[0]))]
        linear_kron_learner = TwoStepRLS(**params)
        linear_kron_learner.train()
        linear_kron_model = linear_kron_learner.getModel()
        linear_kron_testpred_24 = linear_kron_model.predictWithDataMatrices(X_train1[2], X_train2[4])
        
        #Train kernel two-step RLS without out-of-sample rows or columns for [0,0]
        params = {}
        params["regparam1"] = regparam1
        params["regparam2"] = regparam2
        params["kmatrix1"] = K_train1[np.ix_(range(1, K_train1.shape[0]), range(1, K_train1.shape[1]))]
        params["kmatrix2"] = K_train2[np.ix_(range(1, K_train2.shape[0]), range(1, K_train2.shape[1]))]
        params["Y"] = Y_train[np.ix_(range(1, Y_train.shape[0]), range(1, Y_train.shape[1]))]
        kernel_kron_learner = TwoStepRLS(**params)
        kernel_kron_learner.train()
        kernel_kron_model = kernel_kron_learner.getModel()
        kernel_kron_testpred_00 = kernel_kron_model.predictWithKernelMatrices(K_train1[range(1, K_train1.shape[0]), 0], K_train2[0, range(1, K_train2.shape[0])])
        
        #Train kernel two-step RLS without out-of-sample rows or columns for [2,4]
        params = {}
        params["regparam1"] = regparam1
        params["regparam2"] = regparam2
        params["kmatrix1"] = K_train1[np.ix_([0, 1] + range(3, K_train1.shape[0]), [0, 1] + range(3, K_train1.shape[0]))]
        params["kmatrix2"] = K_train2[np.ix_([0, 1, 2, 3] + range(5, K_train2.shape[0]), [0, 1, 2, 3] + range(5, K_train2.shape[0]))]
        params["Y"] = Y_train[np.ix_([0, 1] + range(3, Y_train.shape[0]), [0, 1, 2, 3] + range(5, Y_train.shape[1]))]
        kernel_kron_learner = TwoStepRLS(**params)
        kernel_kron_learner.train()
        kernel_kron_model = kernel_kron_learner.getModel()
        #print K_train1[range(1, K_train1.shape[0]), 0].shape, K_train2[0, range(1, K_train2.shape[0])].shape
        kernel_kron_testpred_24 = kernel_kron_model.predictWithKernelMatrices(K_train1[[0, 1] + range(3, K_train1.shape[0]), 2], K_train2[4, [0, 1, 2, 3] + range(5, K_train2.shape[0])])
        
        print Y_train.shape, secondsteploo.shape, kernel_twostepoutofsampleloo.shape
        print secondsteploo[0, 0], linear_kron_testpred_00, kernel_kron_testpred_00, linear_twostepoutofsampleloo[0, 0], kernel_twostepoutofsampleloo[0, 0]
        print secondsteploo[2, 4], linear_kron_testpred_24, kernel_kron_testpred_24, linear_twostepoutofsampleloo[2, 4], kernel_twostepoutofsampleloo[2, 4]
        print
        #print 'Two-step RLS LOO', twostepoutofsampleloo[2, 4]
        #print np.mean(np.abs(linear_kron_testpred - ordrls_testpred)), np.mean(np.abs(kernel_kron_testpred - ordrls_testpred))
        
        
        
        
        
        #Create symmetric data
        K_train1, K_train2, Y_train, K_test1, K_test2, Y_test, X_train1, X_train2, X_test1, X_test2 \
            = self.generate_xortask(
            trainpos1 = 6,
            trainneg1 = 7,
            trainpos2 = 6,
            trainneg2 = 7,
            testpos1 = 26,
            testneg1 = 27,
            testpos2 = 25,
            testneg2 = 25
            )
        K_train1 = K_train2
        K_test1 = K_test2
        Y_train = 0.5 * (Y_train + Y_train.T)
        
        rows, columns = Y_train.shape
        #print K_train1.shape, K_train2.shape, K_test1.shape, K_test2.shape, rows, columns
        trainlabelcount = rows * columns
        indmatrix = np.mat(range(trainlabelcount)).T.reshape(rows, columns)
        
        #Train symmetric kernel two-step RLS with pre-computed kernel matrices
        params = {}
        params["regparam1"] = regparam2
        params["regparam2"] = regparam2
        params["kmatrix1"] = K_train1
        params["kmatrix2"] = K_train2
        params["Y"] = Y_train
        kernel_two_step_learner = TwoStepRLS(**params)
        kernel_two_step_learner.train()
        kernel_two_step_model = kernel_two_step_learner.getModel()
        kernel_two_step_testpred = kernel_two_step_model.predictWithKernelMatrices(K_test1, K_test2)
        
        #Train two-step RLS without out-of-sample rows or columns
        rowind, colind = 2, 4
        trainrowinds = range(K_train1.shape[0])
        trainrowinds.remove(rowind)
        trainrowinds.remove(colind)
        traincolinds = range(K_train2.shape[0])
        traincolinds.remove(rowind)
        traincolinds.remove(colind)
        
        params = {}
        params["regparam1"] = regparam2
        params["regparam2"] = regparam2
        params["kmatrix1"] = K_train1[np.ix_(trainrowinds, trainrowinds)]
        params["kmatrix2"] = K_train2[np.ix_(traincolinds, traincolinds)]
        params["Y"] = Y_train[np.ix_(trainrowinds, traincolinds)]
        kernel_kron_learner = TwoStepRLS(**params)
        kernel_kron_learner.train()
        kernel_kron_model = kernel_kron_learner.getModel()
        #kernel_kron_testpred = kernel_kron_model.predictWithKernelMatrices(K_train1[np.ix_([rowind, colind], trainrowinds)], K_train2[np.ix_([rowind, colind], traincolinds)])
        kernel_kron_testpred = kernel_kron_model.predictWithKernelMatrices(K_train1[np.ix_([rowind], trainrowinds)], K_train2[np.ix_([colind], traincolinds)])
        print kernel_kron_testpred
        
        fcsho = kernel_two_step_learner.compute_symmetric_double_LOO()
        print fcsho[2, 4]
        



