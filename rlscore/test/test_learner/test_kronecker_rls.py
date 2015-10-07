import unittest

import numpy as np
#from rlscore.measure.cindex_measure import cindex
from rlscore.measure import auc
from rlscore.measure import sqmprank
from rlscore.measure import sqerror
from rlscore.kernel import GaussianKernel
from rlscore.kernel import LinearKernel
from rlscore.learner.kron_rls import KronRLS
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
        kernel1 = LinearKernel(X_train1)
        K_train1 = kernel1.getKM(X_train1)
        K_test1 = kernel1.getKM(X_test1)
        
        #kernel2 = GaussianKernel.createKernel(gamma=0.01, X=X_train2)
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
    
    
    def test_kron_rls(self):
        
        regparam = 0.001
        
        K_train1, K_train2, Y_train, K_test1, K_test2, Y_test, X_train1, X_train2, X_test1, X_test2 = self.generate_xortask()
        rows, columns = Y_train.shape
        #print K_train1.shape, K_train2.shape, K_test1.shape, K_test2.shape, rows, columns
        trainlabelcount = rows * columns
        indmatrix = np.mat(range(trainlabelcount)).T.reshape(rows, columns)
        
        #Train linear Kronecker RLS with data-matrices
        params = {}
        params["regparam"] = regparam
        params["xmatrix1"] = X_train1
        params["xmatrix2"] = X_train2
        params["Y"] = Y_train
        linear_kron_learner = KronRLS(**params)
        linear_kron_learner.train()
        linear_kron_model = linear_kron_learner.getModel()
        linear_kron_testpred = linear_kron_model.predictWithDataMatrices(X_test1, X_test2)
        
        #Train kernel Kronecker RLS with pre-computed kernel matrices
        params = {}
        params["regparam"] = regparam
        params["kmatrix1"] = K_train1
        params["kmatrix2"] = K_train2
        params["Y"] = Y_train
        kernel_kron_learner = KronRLS(**params)
        kernel_kron_learner.train()
        kernel_kron_model = kernel_kron_learner.getModel()
        kernel_kron_testpred = kernel_kron_model.predictWithKernelMatrices(K_test1, K_test2)
        
        #Train an ordinary RLS regressor for reference
        K_Kron_train_x = np.kron(K_train1, K_train2)
        params = {}
        params["kernel_matrix"] = K_Kron_train_x
        params["Y"] = Y_train.reshape(trainlabelcount, 1)
        ordrls_learner = RLS(**params)
        ordrls_learner.solve(regparam)
        ordrls_model = ordrls_learner.getModel()
        K_Kron_test_x = np.kron(K_test1, K_test2)
        ordrls_testpred = ordrls_model.predict(K_Kron_test_x)
        ordrls_testpred = ordrls_testpred.reshape(Y_test.shape[0], Y_test.shape[1])
        
        print
        print type(linear_kron_testpred), type(kernel_kron_testpred), type(ordrls_testpred)
        print linear_kron_testpred[0, 0], kernel_kron_testpred[0, 0], ordrls_testpred[0, 0]
        print linear_kron_testpred[0, 1], kernel_kron_testpred[0, 1], ordrls_testpred[0, 1]
        print linear_kron_testpred[1, 0], kernel_kron_testpred[1, 0], ordrls_testpred[1, 0]
        
        print np.mean(np.abs(linear_kron_testpred - ordrls_testpred)), np.mean(np.abs(kernel_kron_testpred - ordrls_testpred))
    
    
    def test_conditional_ranking(self):
        
        regparam = 0.001
        
        K_train1, K_train2, Y_train, K_test1, K_test2, Y_test, X_train1, X_train2, X_test1, X_test2 = self.generate_xortask()
        rows, columns = Y_train.shape
        trainlabelcount = rows * columns
        indmatrix = np.mat(range(trainlabelcount)).T.reshape(rows, columns)
        
        K_Kron_train_x = np.kron(K_train1, K_train2)
        K_test_x = np.kron(K_test1, K_test2)
        
        
        #Train linear Conditional Ranking Kronecker RLS
        params = {}
        params["xmatrix1"] = X_train1
        params["xmatrix2"] = X_train2
        params["Y"] = Y_train
        params["regparam"] = regparam
        linear_kron_condrank_learner = KronRLS(**params)
        linear_kron_condrank_learner.solve_linear_conditional_ranking(regparam)
        condrank_model = linear_kron_condrank_learner.getModel()
        
        #Train an ordinary RankRLS for reference
        params = {}
        params["kernel_matrix"] = K_Kron_train_x
        params["Y"] = Y_train.reshape(trainlabelcount, 1)
        params["qids"] = [range(i*Y_train.shape[1], (i+1)*Y_train.shape[1]) for i in range(Y_train.shape[0])]
        rankrls_learner = LabelRankRLS(**params)
        rankrls_learner.solve(regparam)
        rankrls_model = rankrls_learner.getModel()
        K_test_x = np.kron(K_test1, K_test2)
        ordrankrls_testpred = rankrls_model.predict(K_test_x).reshape(Y_test.shape[0], Y_test.shape[1])
        condrank_testpred = condrank_model.predictWithDataMatrices(X_test1, X_test2)
        print
        #print condrank_testpred.ravel().shape, Y_test.ravel().shape, ordrankrls_testpred.ravel().shape, Y_test.ravel().shape
        
        print 'TEST cond vs rankrls', np.mean(np.abs(condrank_testpred - ordrankrls_testpred))
    
    
    
    '''
    def testRLS_old_and_broken(self):
        
        regparam = 0.001
        
        K_train1, K_train2, Y_train, K_test1, K_test2, Y_test, X_train1, X_train2, X_test1, X_test2 = self.generate_xortask()
        rows, columns = Y_train.shape
        trainlabelcount = rows * columns
        indmatrix = np.mat(range(trainlabelcount)).T.reshape(rows, columns)
        
        
        #Train kernel Kronecker RLS with pre-computed kernel matrices
        params = {}
        params["regparam"] = regparam
        params["kmatrix1"] = K_train1
        params["kmatrix2"] = K_train2
        params["Y"] = Y_train
        kron_learner = KronRLS(**params)
        kron_learner.train()
        #kron_learner.solve_kernel(regparam)
        kron_model = kron_learner.getModel()
        
        
        #Train linear Kronecker RLS with data-matrices
        params = {}
        params["xmatrix1"] = X_train1
        params["xmatrix2"] = X_train2
        params["Y"] = Y_train
        linear_kron_learner = KronRLS(**params)
        linear_kron_learner.solve_linear(regparam)
        
        #print rows, columns
        
        ho_rows = [3, 4, 6]
        ho_cols = [2, 4, 7]
        
        #Train an ordinary RLS regressor for reference
        K_Kron_train_x = np.kron(K_train1, K_train2)
        params = {}
        params["kernel_matrix"] = K_Kron_train_x
        params["Y"] = Y_train.reshape(trainlabelcount, 1)
        ordrls_learner = RLS(**params)
        ordrls_learner.solve(regparam)
        ordrls_model = ordrls_learner.getModel()
        
        ordrls_loopred = ordrls_learner.computeLOO().reshape(Y_train.shape[0], Y_train.shape[1])
        kron_loopred = linear_kron_learner.imputationLOO()
        print 'LOO', np.sum(np.abs(ordrls_loopred-kron_loopred))
        
        #return
        
        kron_hopred = linear_kron_learner.compute_ho(ho_rows, ho_cols)
        ordrls_hopred = ordrls_learner.computeHO(indmatrix[np.ix_(ho_rows, ho_cols)].ravel().tolist()[0])
        
        print 'HO', np.sum(np.abs(kron_hopred.ravel().T-ordrls_hopred))
        #print kron_hopred.ravel().T, ordrls_hopred
        
        #Test nested LOO in setting 1
        outer_row_coord, outer_col_coord = 2, 5
        inner_row_coord, inner_col_coord = 6, 7
        kron_lpopred = linear_kron_learner.nested_imputationLOO(outer_row_coord, outer_col_coord)
        kron_lpopred[outer_row_coord, outer_col_coord] = 0.
        ordrls_lpopred = np.mat(np.zeros((trainlabelcount, 1)))
        for i in range(trainlabelcount):
            outer_entry_ind = indmatrix[outer_row_coord, outer_col_coord]
            if i == outer_entry_ind: continue
            ordrls_lpopred[i, 0] = ordrls_learner.computeHO([i, outer_entry_ind])[0, 0]
        #ordrls_lpopred =  ordrls_learner.computeHO([indmatrix[inner_row_coord, inner_col_coord], indmatrix[outer_row_coord, outer_col_coord]])
        print 'LPO aka nested LOO in setting 1: ', np.sum(np.abs(ordrls_lpopred - kron_lpopred.ravel().T))#np.sum(np.abs(ordrls_lpopred[0, 0] - kron_lpopred[inner_row_coord, inner_col_coord]))
        #return
        K_test_x = np.kron(K_test1, K_test2)
        print ordrls_model.predict(K_test_x.T).shape, Y_test.shape
        ordrls_testpred = ordrls_model.predict(K_test_x).reshape(Y_test.shape[0], Y_test.shape[1])
        #ordrls_testpred = ordrls_model.predict(K_test_x.T).reshape(Y_test.shape[0], Y_test.shape[1])
        kron_testpred = kron_model.predictWithKernelMatrices(K_test1, K_test2)
        print 'TEST', np.mean(np.abs(kron_testpred - ordrls_testpred))
        
        params = {}
        params["kmatrix1"] = K_train1
        params["kmatrix2"] = K_train2
        params["Y"] = Y_train
        condrank_learner = ConditionalRanking(**params)
        condrank_learner.solve(regparam)
        condrank_model = condrank_learner.getModel()
        
        #Train linear Conditional Ranking Kronecker RLS
        params = {}
        params["xmatrix1"] = X_train1
        params["xmatrix2"] = X_train2
        params["Y"] = Y_train
        linear_kron_condrank_learner = KronRLS(**params)
        linear_kron_condrank_learner.solve_linear_conditional_ranking(regparam)
        condrank_model = linear_kron_condrank_learner.getModel()
        
        params = {}
        params["kmatrix"] = K_Kron_train_x
        params["Y"] = Y_train.reshape(trainlabelcount, 1)#Y_train.reshape(Y_train.shape[0]*Y_train.shape[1],1)
        #params["qids"] = [range(i*Y_train.shape[1], (i+1)*Y_train.shape[1]) for i in range(Y_train.shape[0])]
        params["qids"] = [range(i*Y_train.shape[1], (i+1)*Y_train.shape[1]) for i in range(Y_train.shape[0])]
        rankrls_learner = LabelRankRLS(**params)
        
        rankrls_learner.solve(regparam)
        rankrls_model = rankrls_learner.getModel()
        K_test_x = np.kron(K_test1, K_test2)
        ordrankrls_testpred = rankrls_model.predict(K_test_x).reshape(Y_test.shape[0], Y_test.shape[1])
        #condrank_testpred = condrank_model.predictWithKernelMatrices(K_test1, K_test2)
        condrank_testpred = condrank_model.predictWithDataMatrices(X_test1, X_test2)
        print
        #print 'TEST rank', np.mean(np.abs(condrank_testpred - ordrankrls_testpred))
        #print 'TEST rank', sqmprank(condrank_testpred, Y_test), sqmprank(ordrankrls_testpred, Y_test)
        #print Y_test
        #print ordrankrls_testpred
        #print condrank_testpred
        #print kron_testpred
        print condrank_testpred.ravel().shape, Y_test.ravel().shape, ordrankrls_testpred.ravel().shape, Y_test.ravel().shape
        #print 'TEST rank', auc(condrank_testpred.ravel().T, Y_test.ravel()), disagreement(ordrankrls_testpred.ravel().T, Y_test.ravel())
        
        C_left = np.mat(np.eye(Y_test.shape[0])) - (1. / float(Y_test.shape[0])) * np.mat(np.ones((Y_test.shape[0], 1))) * np.mat(np.ones((Y_test.shape[0], 1))).T
        C_right = np.mat(np.eye(Y_test.shape[1])) - (1. / float(Y_test.shape[1])) * np.mat(np.ones((Y_test.shape[1], 1))) * np.mat(np.ones((Y_test.shape[1], 1))).T
        #print C
        print 'TEST kron vs rankrls', np.mean(np.abs(kron_testpred - ordrankrls_testpred))
        print 'TEST kron vs cond', np.mean(np.abs(kron_testpred - condrank_testpred))
        print 'TEST cond vs rankrls', np.mean(np.abs(condrank_testpred - ordrankrls_testpred))
        print
        print 'TEST kron vs rankrls', np.mean(np.abs(C_left * (kron_testpred - ordrankrls_testpred)))
        print 'TEST kron vs cond', np.mean(np.abs(C_left * (kron_testpred - condrank_testpred)))
        print 'TEST cond vs rankrls', np.mean(np.abs(C_left * (condrank_testpred - ordrankrls_testpred)))
        print
        print 'TEST kron vs rankrls', np.mean(np.abs((kron_testpred - ordrankrls_testpred) * C_right))
        print 'TEST kron vs cond', np.mean(np.abs((kron_testpred - condrank_testpred) * C_right))
        print 'TEST cond vs rankrls', np.mean(np.abs((condrank_testpred - ordrankrls_testpred) * C_right))
        
        #for i in range(Y_test.shape[0]):
        #    for j in range(Y_test.shape[1]):
        #        print Y_test[i, j],
        #    print
        #    print
        #print kron_learner.A
        #print condrank_learner.A
        
        #rlshopred = rankrls_modeordrls_testpredFromPool({'prediction_features':K_test_x.T}).reshape(Y_test.shape[0],Y_test.shape[1])
        #print np.mean(np.abs(kron_hopred-rlshopred))
        #print auc(Y_test.T, kron_hopred.T), sqerror(Y_test.T, kron_hopred.T)
        #print auc(Y_test.T, rlshopred.T), sqerror(Y_test.T, rlshopred.T)
        '''
