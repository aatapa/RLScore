import unittest

import numpy as np
import numpy.linalg as la

from rlscore.kernel import LinearKernel
from rlscore.learner.two_step_rls import TwoStepRLS
from rlscore.learner.rls import RLS


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
        
#         dim1 = 5
#         dim2 = 5
        dim1 = 15
        dim2 = 17
        
        np.random.seed(55)
        X_train1, Y_train1 = self.generate_data(trainpos1, trainneg1, dim1, 0, 1)
        X_train2, Y_train2 = self.generate_data(trainpos2, trainneg2, dim2, 4, 6)
        
        X_test1, Y_test1 = self.generate_data(testpos1, testneg1, dim1, 0, 1)
        X_test2, Y_test2 = self.generate_data(testpos2, testneg2, dim2, 4, 6)
        
        #kernel1 = GaussianKernel.createKernel(gamma=0.01, X=X_train1)
        kernel1 = LinearKernel(X_train1, bias=0.0)
        K_train1 = kernel1.getKM(X_train1)
        K_test1 = kernel1.getKM(X_test1)
        
        #kernel2 = GaussianKernel.createKernel(gamma=0.01, train_features=X_train2)
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
    
    
    def test_two_step_rls(self):
        
        regparam1 = 0.001
        regparam2 = 10
        #regparam1 = 1
        #regparam2 = 1
        
        K_train1, K_train2, Y_train, K_test1, K_test2, Y_test, X_train1, X_train2, X_test1, X_test2 \
            = self.generate_xortask()
        Y_train = Y_train.ravel(order = 'F')
        Y_test = Y_test.ravel(order = 'F')
        train_rows, train_columns = K_train1.shape[0], K_train2.shape[0]
        #print K_train1.shape, K_train2.shape, K_test1.shape, K_test2.shape, train_rows, train_columns
        trainlabelcount = train_rows * train_columns
        
        #Train linear two-step RLS with data-matrices
        params = {}
        params["regparam1"] = regparam1
        params["regparam2"] = regparam2
        params["X1"] = X_train1
        params["X2"] = X_train2
        params["Y"] = Y_train
        linear_two_step_learner = TwoStepRLS(**params)
        linear_twostepoutofsampleloo = linear_two_step_learner.out_of_sample_loo().reshape((train_rows, train_columns), order = 'F')
        linear_lro = linear_two_step_learner.leave_x1_out()
        linear_lco = linear_two_step_learner.leave_x2_out()
                                                              
        #Train kernel two-step RLS with pre-computed kernel matrices
        params = {}
        params["regparam1"] = regparam1
        params["regparam2"] = regparam2
        params["K1"] = K_train1
        params["K2"] = K_train2
        params["Y"] = Y_train
        kernel_two_step_learner = TwoStepRLS(**params)
        kernel_twostepoutofsampleloo = kernel_two_step_learner.out_of_sample_loo().reshape((train_rows, train_columns), order = 'F')
        kernel_lro = kernel_two_step_learner.leave_x1_out()
        kernel_lco = kernel_two_step_learner.leave_x2_out()
        tspred = kernel_two_step_learner.predict(K_train1, K_train2)
        
        #Train ordinary linear RLS in two steps for a reference
        params = {}
        params["regparam"] = regparam2
        params["X"] = X_train2
        params["Y"] = Y_train.reshape((train_rows, train_columns), order = 'F').T
        params['bias'] = 0
        ordinary_linear_rls_first_step = RLS(**params)
        firststeploo = ordinary_linear_rls_first_step.leave_one_out().T
        params = {}
        params["regparam"] = regparam1
        params["X"] = X_train1
        params["Y"] = firststeploo.reshape((train_rows, train_columns), order = 'F')
        params['bias'] = 0
        ordinary_linear_rls_second_step = RLS(**params)
        secondsteploo_linear_rls = ordinary_linear_rls_second_step.leave_one_out()
        
        #Train ordinary kernel RLS in two steps for a reference
        params = {}
        params["regparam"] = regparam2
        params["X"] = K_train2
        params['kernel'] = 'PrecomputedKernel'
        params["Y"] = Y_train.reshape((train_rows, train_columns), order = 'F').T
        ordinary_kernel_rls_first_step = RLS(**params)
        firststeploo = ordinary_kernel_rls_first_step.leave_one_out().T
        params = {}
        params["regparam"] = regparam1
        params["X"] = K_train1
        params["kernel"] = "PrecomputedKernel"
        params["Y"] = firststeploo.reshape((train_rows, train_columns), order = 'F')
        ordinary_kernel_rls_second_step = RLS(**params)
        secondsteploo_kernel_rls = ordinary_kernel_rls_second_step.leave_one_out()
        
        #Train ordinary kernel RLS in one step with the crazy kernel for a reference
        params = {}
        params["regparam"] = 1.
        crazykernel = la.inv(regparam1 * regparam2 * np.kron(la.inv(K_train2), la.inv(K_train1))
                       + regparam1 * np.kron(np.eye(K_train2.shape[0]), la.inv(K_train1))
                       + regparam2 * np.kron(la.inv(K_train2), np.eye(K_train1.shape[0])))
        params["X"] = crazykernel
        params['kernel'] = 'PrecomputedKernel'
        params["Y"] = Y_train
        ordinary_one_step_kernel_rls_with_crazy_kernel = RLS(**params)
        fooloo = ordinary_one_step_kernel_rls_with_crazy_kernel.leave_one_out()[0]
        allinds = np.arange(trainlabelcount)
        allinds_fortran_shaped = allinds.reshape((train_rows, train_columns), order = 'F')
        hoinds = sorted(allinds_fortran_shaped[0].tolist() + allinds_fortran_shaped[1:, 0].tolist())
        hocompl = sorted(list(set(allinds)-set(hoinds)))
        fooholdout = ordinary_one_step_kernel_rls_with_crazy_kernel.holdout(hoinds)[0]
        params = {}
        params["regparam"] = 1.
        params["X"] = crazykernel[np.ix_(hocompl, hocompl)]
        params['kernel'] = 'PrecomputedKernel'
        params["Y"] = Y_train[hocompl]
        ordinary_one_step_kernel_rls_with_crazy_kernel = RLS(**params)
        barholdout = ordinary_one_step_kernel_rls_with_crazy_kernel.predict(crazykernel[np.ix_([0], hocompl)])
        params = {}
        params["regparam"] = 1.
        K_train1_cut = K_train1[np.ix_(range(1, K_train1.shape[0]), range(1, K_train1.shape[1]))]
        K_train2_cut = K_train2[np.ix_(range(1, K_train2.shape[0]), range(1, K_train2.shape[1]))]
        crazykernel_cut = la.inv(regparam1 * regparam2 * np.kron(la.inv(K_train2_cut), la.inv(K_train1_cut))
                       + regparam1 * np.kron(np.eye(K_train2_cut.shape[0]), la.inv(K_train1_cut))
                       + regparam2 * np.kron(la.inv(K_train2_cut), np.eye(K_train1_cut.shape[0])))
        params["X"] = crazykernel_cut
        params['kernel'] = 'PrecomputedKernel'
        #params["Y"] = Y_train[hocompl]
        params["Y"] = Y_train.reshape((train_rows, train_columns), order = 'F')[np.ix_(range(1, train_rows), range(1, train_columns))].ravel(order = 'F')
        ordinary_one_step_kernel_rls_with_crazy_kernel = RLS(**params)
        bazholdout = ordinary_one_step_kernel_rls_with_crazy_kernel.predict(
            np.dot(np.dot(np.kron(K_train2[np.ix_([0], range(1, K_train2.shape[1]))], K_train1[np.ix_([0], range(1, K_train1.shape[1]))]),
                           la.inv(np.kron(K_train2_cut, K_train1_cut))),
                           crazykernel_cut))
        #print fooholdout, 'fooholdout', barholdout, bazholdout
        
        #Train linear two-step RLS without out-of-sample rows or columns for [0,0]
        params = {}
        params["regparam1"] = regparam1
        params["regparam2"] = regparam2
        params["X1"] = X_train1[range(1, X_train1.shape[0])]
        params["X2"] = X_train2[range(1, X_train2.shape[0])]
        params["Y"] = Y_train.reshape((train_rows, train_columns), order = 'F')[np.ix_(range(1, train_rows), range(1, train_columns))].ravel(order = 'F')
        linear_two_step_learner_00 = TwoStepRLS(**params)
        linear_two_step_testpred_00 = linear_two_step_learner_00.predict(X_train1[0], X_train2[0])
        
        #Train linear two-step RLS without out-of-sample rows or columns for [2,4]
        params = {}
        params["regparam1"] = regparam1
        params["regparam2"] = regparam2
        params["X1"] = X_train1[[0, 1] + list(range(3, K_train1.shape[0]))]
        params["X2"] = X_train2[[0, 1, 2, 3] + list(range(5, K_train2.shape[0]))]
        params["Y"] = Y_train.reshape((train_rows, train_columns), order = 'F')[np.ix_([0, 1] + list(range(3, train_rows)), [0, 1, 2, 3] + list(range(5, train_columns)))].ravel(order = 'F')
        linear_two_step_learner_24 = TwoStepRLS(**params)
        linear_two_step_testpred_24 = linear_two_step_learner_24.predict(X_train1[2], X_train2[4])
        
        #Train kernel two-step RLS without out-of-sample rows or columns for [0,0]
        params = {}
        params["regparam1"] = regparam1
        params["regparam2"] = regparam2
        params["K1"] = K_train1[np.ix_(range(1, K_train1.shape[0]), range(1, K_train1.shape[1]))]
        params["K2"] = K_train2[np.ix_(range(1, K_train2.shape[0]), range(1, K_train2.shape[1]))]
        params["Y"] = Y_train.reshape((train_rows, train_columns), order = 'F')[np.ix_(range(1, train_rows), range(1, train_columns))].ravel(order = 'F')
        kernel_two_step_learner_00 = TwoStepRLS(**params)
        kernel_two_step_testpred_00 = kernel_two_step_learner_00.predict(K_train1[range(1, K_train1.shape[0]), 0], K_train2[0, range(1, K_train2.shape[0])])
        
        #Train kernel two-step RLS without out-of-sample rows or columns for [2,4]
        params = {}
        params["regparam1"] = regparam1
        params["regparam2"] = regparam2
        params["K1"] = K_train1[np.ix_([0, 1] + list(range(3, K_train1.shape[0])), [0, 1] + list(range(3, K_train1.shape[0])))]
        params["K2"] = K_train2[np.ix_([0, 1, 2, 3] + list(range(5, K_train2.shape[0])), [0, 1, 2, 3] + list(range(5, K_train2.shape[0])))]
        params["Y"] = Y_train.reshape((train_rows, train_columns), order = 'F')[np.ix_([0, 1] + list(range(3, train_rows)), [0, 1, 2, 3] + list(range(5, train_columns)))].ravel(order = 'F')
        kernel_two_step_learner_24 = TwoStepRLS(**params)
        kernel_two_step_testpred_24 = kernel_two_step_learner_24.predict(K_train1[[0, 1] + list(range(3, K_train1.shape[0])), 2], K_train2[4, [0, 1, 2, 3] + list(range(5, K_train2.shape[0]))])
        
        #Train kernel two-step RLS without out-of-sample row 0
        params = {}
        params["regparam1"] = regparam1
        params["regparam2"] = regparam2
        params["K1"] = K_train1[np.ix_(range(1, K_train1.shape[0]), range(1, K_train1.shape[1]))]
        params["K2"] = K_train2
        params["Y"] = Y_train.reshape((train_rows, train_columns), order = 'F')[range(1, train_rows)].ravel(order = 'F')
        kernel_two_step_learner_lro_0 = TwoStepRLS(**params)
        kernel_two_step_testpred_lro_0 = kernel_two_step_learner_lro_0.predict(K_train1[range(1, K_train1.shape[0]), 0], K_train2)
        print('')
        print('Leave-row-out with linear two-step RLS:')
        print(linear_lro.reshape((train_rows, train_columns), order = 'F')[0])
        print('Leave-row-out with kernel two-step RLS:')
        print(kernel_lro.reshape((train_rows, train_columns), order = 'F')[0])
        print('Two-step RLS trained without the held-out row predictions for the row:')
        print(kernel_two_step_testpred_lro_0)
        np.testing.assert_almost_equal(linear_lro.reshape((train_rows, train_columns), order = 'F')[0], kernel_two_step_testpred_lro_0)
        np.testing.assert_almost_equal(kernel_lro.reshape((train_rows, train_columns), order = 'F')[0], kernel_two_step_testpred_lro_0)
        
        
        #Train kernel two-step RLS without out-of-sample column 0
        params = {}
        params["regparam1"] = regparam1
        params["regparam2"] = regparam2
        params["K1"] = K_train1
        params["K2"] = K_train2[np.ix_(range(1, K_train2.shape[0]), range(1, K_train2.shape[1]))]
        params["Y"] = Y_train.reshape((train_rows, train_columns), order = 'F')[:, range(1, train_columns)].ravel(order = 'F')
        kernel_two_step_learner_lco_0 = TwoStepRLS(**params)
        kernel_two_step_testpred_lco_0 = kernel_two_step_learner_lco_0.predict(K_train1, K_train2[range(1, K_train2.shape[0]), 0])
        print('')
        print('Leave-column-out with linear two-step RLS:')
        print(linear_lco[range(train_rows)])
        print('Leave-column-out with kernel two-step RLS:')
        print(kernel_lco[range(train_rows)])
        print('Two-step RLS trained without the held-out column predictions for the column:')
        print(kernel_two_step_testpred_lco_0)
        np.testing.assert_almost_equal(linear_lco[range(train_rows)], kernel_two_step_testpred_lco_0)
        np.testing.assert_almost_equal(kernel_lco[range(train_rows)], kernel_two_step_testpred_lco_0)
        
        print('')
        print('Out-of-sample LOO: Stacked ordinary linear RLS LOO, Stacked ordinary kernel RLS LOO, linear two-step RLS OOSLOO, kernel two-step RLS OOSLOO, linear two-step RLS OOS-pred, kernel two-step RLS OOS-pred')
        print('[0, 0]: ' + str(secondsteploo_linear_rls[0, 0])
                         + ' ' + str(secondsteploo_kernel_rls[0, 0])
                         + ' ' + str(linear_two_step_testpred_00)
                         + ' ' + str(kernel_two_step_testpred_00)
                         + ' ' + str(linear_twostepoutofsampleloo[0, 0])
                         + ' ' + str(kernel_twostepoutofsampleloo[0, 0]))
        print('[2, 4]: ' + str(secondsteploo_linear_rls[2, 4])
                         + ' ' + str(secondsteploo_kernel_rls[2, 4])
                         + ' ' + str(linear_two_step_testpred_24)
                         + ' ' + str(kernel_two_step_testpred_24)
                         + ' ' + str(linear_twostepoutofsampleloo[2, 4])
                         + ' ' + str(kernel_twostepoutofsampleloo[2, 4]))
        np.testing.assert_almost_equal(secondsteploo_linear_rls, secondsteploo_kernel_rls)
        np.testing.assert_almost_equal(secondsteploo_linear_rls, linear_twostepoutofsampleloo)
        np.testing.assert_almost_equal(secondsteploo_linear_rls, kernel_twostepoutofsampleloo)
        np.testing.assert_almost_equal(secondsteploo_linear_rls[0, 0], linear_two_step_testpred_00)
        np.testing.assert_almost_equal(secondsteploo_linear_rls[0, 0], kernel_two_step_testpred_00)
        
        
        #Train kernel two-step RLS with pre-computed kernel matrices and with output at position [2, 4] changed
        Y_24 = Y_train.copy()
        Y_24 = Y_24.reshape((train_rows, train_columns), order = 'F')
        Y_24[2, 4] = 55.
        Y_24 = Y_24.ravel(order = 'F')
        params = {}
        params["regparam1"] = regparam1
        params["regparam2"] = regparam2
        params["K1"] = K_train1
        params["K2"] = K_train2
        params["Y"] = Y_24
        kernel_two_step_learner_Y_24 = TwoStepRLS(**params)
        kernel_two_step_testpred_Y_24 = kernel_two_step_learner_Y_24.predict(K_test1, K_test2)
        
        
        kernel_two_step_learner_inSampleLOO_24a = kernel_two_step_learner.in_sample_loo().reshape((train_rows, train_columns), order = 'F')[2, 4]
        kernel_two_step_learner_inSampleLOO_24b = kernel_two_step_learner_Y_24.in_sample_loo().reshape((train_rows, train_columns), order = 'F')[2, 4]
        print('')
        print('In-sample LOO: Kernel two-step RLS ISLOO with original outputs, Kernel two-step RLS ISLOO with modified output at [2, 4]')
        print('[2, 4] ' + str(kernel_two_step_learner_inSampleLOO_24a) + ' ' + str(kernel_two_step_learner_inSampleLOO_24b))
        np.testing.assert_almost_equal(kernel_two_step_learner_inSampleLOO_24a, kernel_two_step_learner_inSampleLOO_24b)
        
        
        
        #Train kernel two-step RLS with pre-computed kernel matrices and with output at position [1, 1] changed
        Y_00 = Y_train.copy()
        Y_00 = Y_00.reshape((train_rows, train_columns), order = 'F')
        Y_00[0, 0] = 55.
        Y_00 = Y_00.ravel(order = 'F')
        params = {}
        params["regparam1"] = regparam1
        params["regparam2"] = regparam2
        params["K1"] = K_train1
        params["K2"] = K_train2
        params["Y"] = Y_00
        kernel_two_step_learner_Y_00 = TwoStepRLS(**params)
        kernel_two_step_testpred_Y_00 = kernel_two_step_learner_Y_00.predict(K_test1, K_test2)
        
        kernel_two_step_learner_inSampleLOO_00a = kernel_two_step_learner.in_sample_loo()[0]
        kernel_two_step_learner_inSampleLOO_00b = kernel_two_step_learner_Y_00.in_sample_loo()[0]
        print('')
        print('In-sample LOO: Kernel two-step RLS ISLOO with original outputs, Kernel two-step RLS ISLOO with modified output at [0, 0]')
        print('[0, 0] ' + str(kernel_two_step_learner_inSampleLOO_00a) + ' ' + str(kernel_two_step_learner_inSampleLOO_00b))
        np.testing.assert_almost_equal(kernel_two_step_learner_inSampleLOO_00a, kernel_two_step_learner_inSampleLOO_00b)
        
        
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
        
        Y_train = Y_train.ravel(order = 'F')
        Y_test = Y_test.ravel(order = 'F')
        train_rows, train_columns = K_train1.shape[0], K_train2.shape[0]
        test_rows, test_columns = K_test1.shape[0], K_test2.shape[0]
        trainlabelcount = train_rows * train_columns
        
        #Train symmetric kernel two-step RLS with pre-computed kernel matrices
        params = {}
        params["regparam1"] = regparam2
        params["regparam2"] = regparam2
        params["K1"] = K_train1
        params["K2"] = K_train2
        params["Y"] = Y_train
        kernel_two_step_learner = TwoStepRLS(**params)
        kernel_two_step_testpred = kernel_two_step_learner.predict(K_test1, K_test2).reshape((test_rows, test_columns), order = 'F')
        
        #Train two-step RLS without out-of-sample rows or columns
        rowind, colind = 2, 4
        trainrowinds = list(range(K_train1.shape[0]))
        trainrowinds.remove(rowind)
        trainrowinds.remove(colind)
        traincolinds = list(range(K_train2.shape[0]))
        traincolinds.remove(rowind)
        traincolinds.remove(colind)
        
        params = {}
        params["regparam1"] = regparam2
        params["regparam2"] = regparam2
        params["K1"] = K_train1[np.ix_(trainrowinds, trainrowinds)]
        params["K2"] = K_train2[np.ix_(traincolinds, traincolinds)]
        params["Y"] = Y_train.reshape((train_rows, train_columns), order = 'F')[np.ix_(trainrowinds, traincolinds)].ravel(order = 'F')
        kernel_kron_learner = TwoStepRLS(**params)
        kernel_kron_testpred = kernel_kron_learner.predict(K_train1[np.ix_([rowind], trainrowinds)], K_train2[np.ix_([colind], traincolinds)]).reshape((1, 1), order = 'F')
        
        fcsho = kernel_two_step_learner.out_of_sample_loo_symmetric().reshape((train_rows, train_columns), order = 'F')
        
        print('')
        print('Symmetric double out-of-sample LOO: Test prediction, LOO')
        print('[2, 4]: ' + str(kernel_kron_testpred[0, 0]) + ' ' + str(fcsho[2, 4]))
        np.testing.assert_almost_equal(kernel_kron_testpred[0, 0], fcsho[2, 4])
        
        



