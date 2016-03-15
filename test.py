import unittest

def testKernels():
    names = ["test_gaussian", "test_linear", "test_polynomial"]
    names = ["rlscore.test.test_kernel." +x for x in names ]
    loader = unittest.TestLoader()
    all_tests = loader.loadTestsFromNames(names)
    return all_tests
        
def testLearners():
    names = ["test_cg_kron_rls", "test_cg_rankrls", "test_cg_rls", "test_global_rankrls", "test_greedy_rls", "test_interactive_rls_classifier",
             "test_kronecker_rls", "test_kronsvm", "test_mmc", "test_query_rankrls", "test_rankrls_with_pairwise_preferences",
             "test_reduced_set_approximation", "test_rls", "test_two_step_rls"]
    names = ["rlscore.test.test_learner." +x for x in names ]
    loader = unittest.TestLoader()
    all_tests = loader.loadTestsFromNames(names)
    return all_tests

def testMeasures():
    names = ["test_accuracy", "test_auc", "test_cindex", "test_multiaccuracy", "test_sqerror"]
    names = ["rlscore.test.test_measure." +x for x in names ]
    loader = unittest.TestLoader()
    all_tests = loader.loadTestsFromNames(names)
    return all_tests


def testModels():
    from rlscore.test.test_model import Test
    suite = unittest.TestLoader().loadTestsFromTestCase(Test)
    return suite


def testUtilities():
    from rlscore.test.test_utility.test_sampled_kronecker_products import Test
    suite = unittest.TestLoader().loadTestsFromTestCase(Test)
    return suite
        
if __name__ == "__main__":
    all_tests = []
    for func in [testLearners, testMeasures, testKernels, testModels, testUtilities]:
        all_tests.append(func())
    combo = unittest.TestSuite(all_tests)
    unittest.TextTestRunner(verbosity=2).run(combo)
