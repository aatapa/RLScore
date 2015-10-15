cfiles = {"train_features":'./examples/data/class_train.features',
                          "train_labels":'./examples/data/class_train.labels',
                          "test_features":'./examples/data/class_test.features',
                          "test_labels":'./examples/data/class_test.labels'}

rfiles = {"train_features":'./examples/data/reg_train.features',
                          "train_labels":'./examples/data/reg_train.labels',
                          "test_features":'./examples/data/reg_test.features',
                          "test_labels":'./examples/data/reg_test.labels'}

defparams = {"X":"train_features", "Y":"train_labels"}

experiments = {"rls_defparams":{
                "learner":"RLS",
                "lpath":"rlscore.learner.rls",
                "measure":"auc",
                "lfparams": defparams,
                "lparams": {"regparam":1},
                "files": cfiles},
               
               "rls_classification":{
                "learner":"RLS",
                "lpath":"rlscore.learner.rls",
                "measure":"auc",
                "selector":"LOOCV",
                "sparams":{"measure":"auc"},
                "lfparams": defparams,
                "lparams": {"regparam":1},
                "files": cfiles},
                
                "rls_regression":{
                "learner":"RLS",
                "lpath":"rlscore.learner.rls",
                "measure":"sqerror",
                "selector":"LOOCV",
                "sparams":{"measure":"sqerror"},
                "lfparams": defparams,
                "lparams": {"regparam":1},
                "files": rfiles},
                
                "rls_nfold":{
                 "learner":"RLS",
                 "lpath":"rlscore.learner.rls",
                 "measure":"auc",
                 "selector":"NfoldCV",
                 "sparams":{"measure":"auc",
                            "folds":"folds"},
                "lfparams": defparams,
                "lparams": {"regparam":1},
                "files": dict(cfiles.items()+[("folds",'./examples/data/folds.txt')])},
               
                "rls_gaussian":{
                "learner":"RLS",
                "lpath":"rlscore.learner.rls",
                "measure":"auc",
                "selector":"LOOCV",
                "sparams":{"measure":"auc"},
                "lfparams": defparams,
                "lparams": {"regparam":1, "kernel":"GaussianKernel", "gamma":0.01},
                "files": cfiles},
               
               "rls_polynomial":{
                "learner":"RLS",
                "lpath":"rlscore.learner.rls",
                "measure":"auc",
                "selector":"LOOCV",
                "sparams":{"measure":"auc"},
                "lfparams": defparams,
                "lparams": {"regparam":1, "kernel":"PolynomialKernel", "gamma":2, "coef0":1, "degree":3},
                "files": cfiles},
               
               "rls_reduced":{
                "learner":"RLS",
                "lpath":"rlscore.learner.rls",
                "measure":"auc",
                "selector":"LOOCV",
                "sparams":{"measure":"auc"},
                "lfparams": dict(defparams.items()+[("basis_vectors","basis_vectors")]),
                "lparams": {"regparam":1, "kernel":"PolynomialKernel", "gamma":0.01},
                "files": dict(cfiles.items()+[("basis_vectors",'./examples/data/bvectors.indices')])},
               
               "rls_reduced_linear":{
                "learner":"RLS",
                "lpath":"rlscore.learner.rls",
                "measure":"auc",
                "selector":"LOOCV",
                "sparams":{"measure":"auc"},
                "lfparams": dict(defparams.items()+[("basis_vectors","basis_vectors")]),
                "lparams": {"regparam":1, "kernel":"LinearKernel"},
                "files": dict(cfiles.items()+[("basis_vectors",'./examples/data/bvectors.indices')])},
               
               "cg_rls":{
                "learner":"CGRLS",
                "lpath":"rlscore.learner.cg_rls",
                "measure":"auc",
                "sparams":{"measure":"auc"},
                "lfparams": defparams,
                "lparams": {"regparam":1},
                "files": cfiles,
               }}
