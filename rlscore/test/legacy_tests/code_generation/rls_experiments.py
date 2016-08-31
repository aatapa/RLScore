cfiles = {"train_features":'./legacy_tests/data/class_train.features',
                          "train_labels":'./legacy_tests/data/class_train.labels',
                          "test_features":'./legacy_tests/data/class_test.features',
                          "test_labels":'./legacy_tests/data/class_test.labels'}

rfiles = {"train_features":'./legacy_tests/data/reg_train.features',
                          "train_labels":'./legacy_tests/data/reg_train.labels',
                          "test_features":'./legacy_tests/data/reg_test.features',
                          "test_labels":'./legacy_tests/data/reg_test.labels'}

defparams = {"X":"train_features", "Y":"train_labels"}

experiments = {"rls_defparams":{
                "learner":"RLS",
                "lpath":"rlscore.learner.rls",
                "measure":"auc",
                "lfparams": defparams,
                "lparams": {"regparam":1},
                "files": cfiles},
               
               "rls_classification":{
                "learner":"LeaveOneOutRLS",
                "lpath":"rlscore.learner.rls",
                "measure":"auc",
                "lfparams": defparams,
                "lparams": {},
                "files": cfiles,
                "selection":True},
                
                "rls_regression":{
                "learner":"LeaveOneOutRLS",
                "lpath":"rlscore.learner.rls",
                "measure":"sqerror",
                "lfparams": defparams,
                "lparams": {},
                "files": rfiles,
                "selection":True},

                "rls_lpocv":{
                 "learner":"LeavePairOutRLS",
                 "lpath":"rlscore.learner.rls",
                 "measure":"auc",
                "lfparams": dict(defparams.items() + [("folds","folds")]),
                "lparams": {},
                "files": dict(cfiles.items()+[("folds",'./legacy_tests/data/folds.txt')]),
                "selection":True},
                
                "rls_nfold":{
                 "learner":"KfoldRLS",
                 "lpath":"rlscore.learner.rls",
                 "measure":"auc",
                "lfparams": dict(defparams.items() + [("folds","folds")]),
                "lparams": {},
                "files": dict(cfiles.items()+[("folds",'./legacy_tests/data/folds.txt')]),
                "selection":True},
               
                "rls_gaussian":{
                "learner":"LeaveOneOutRLS",
                "lpath":"rlscore.learner.rls",
                "measure":"auc",
                "lfparams": defparams,
                "lparams": {"kernel":"GaussianKernel", "gamma":0.01},
                "files": cfiles,
                "selection":True},
               
               "rls_polynomial":{
                "learner":"LeaveOneOutRLS",
                "lpath":"rlscore.learner.rls",
                "measure":"auc",
                "lfparams": defparams,
                "lparams": {"kernel":"PolynomialKernel", "gamma":2, "coef0":1, "degree":3},
                "files": cfiles,
                "selection":True},
               
               "rls_reduced":{
                "learner":"LeaveOneOutRLS",
                "lpath":"rlscore.learner.rls",
                "measure":"auc",
                "lfparams": dict(defparams.items()+[("basis_vectors","basis_vectors")]),
                "lparams": {"kernel":"PolynomialKernel", "gamma":0.01},
                "files": dict(cfiles.items()+[("basis_vectors",'./legacy_tests/data/bvectors.indices')]),
                "selection":True},
               
               "rls_reduced_linear":{
                "learner":"LeaveOneOutRLS",
                "lpath":"rlscore.learner.rls",
                "measure":"auc",
                "lfparams": dict(defparams.items()+[("basis_vectors","basis_vectors")]),
                "lparams": {},
                "files": dict(cfiles.items()+[("basis_vectors",'./legacy_tests/data/bvectors.indices')]),
                "selection":True},
               
               "cg_rls":{
                "learner":"CGRLS",
                "lpath":"rlscore.learner.cg_rls",
                "measure":"auc",
                "lfparams": defparams,
                "lparams": {"regparam":1},
                "files": cfiles,
               }}
