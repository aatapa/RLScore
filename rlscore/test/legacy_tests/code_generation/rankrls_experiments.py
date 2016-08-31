rfiles = {"train_features":'./legacy_tests/data/rank_train.features',
                          "train_labels":'./legacy_tests/data/rank_train.labels',
                          "test_features":'./legacy_tests/data/rank_test.features',
                          "test_labels":'./legacy_tests/data/rank_test.labels'}

rqfiles = {"train_features":'./legacy_tests/data/rank_train.features',
           "train_labels":'./legacy_tests/data/rank_train.labels',
           "train_qids":'./legacy_tests/data/rank_train.qids',
           "test_features":'./legacy_tests/data/rank_test.features',
            "test_labels":'./legacy_tests/data/rank_test.labels',
            "test_qids":'./legacy_tests/data/rank_test.qids',
            }

defparams = {"X":"train_features", "Y":"train_labels"}
defqparams = {"X":"train_features", "Y":"train_labels", "qids":"train_qids"}

experiments = {"rankrls_defparams":{
                "learner":"GlobalRankRLS",
                "lpath":"rlscore.learner.global_rankrls",
                "measure":"cindex",
                "lfparams": defparams,
                "lparams": {"regparam":1},
                "files": rfiles},
               
               "rankrls_lpo":{
                "learner":"LeavePairOutRankRLS",
                "lpath":"rlscore.learner.global_rankrls",
                "measure":"cindex",
                "lfparams": defparams,
                "lparams": {},
                "selection":True,
                "files": rfiles},

               "rankrls_kfold":{
                "learner":"KfoldRankRLS",
                "lpath":"rlscore.learner.global_rankrls",
                "measure":"cindex",
                "lfparams": dict(defparams.items() + [("folds","folds")]),
                "lparams": {},
                "selection":True,
                "files": dict(rfiles.items()+[("folds",'./legacy_tests/data/folds.txt')])},
               
                "rankrls_rset":{
                "learner":"LeavePairOutRankRLS",
                "lpath":"rlscore.learner.global_rankrls",
                "measure":"cindex",
                "lfparams": dict(defparams.items()+[("basis_vectors","basis_vectors")]),
                "lparams": {"kernel":"PolynomialKernel", "gamma":2, "coef0":1, "degree":3},
                "files": dict(rfiles.items()+[("basis_vectors",'./legacy_tests/data/bvectors.indices')]),
                "selection":True},
               
               "rankrls_qids":{
                "learner":"QueryRankRLS",
                "lpath":"rlscore.learner.query_rankrls",
                "measure":"cindex",
                "lfparams": defqparams,
                "lparams": {"regparam":1},
                "files": rqfiles},
               
                "rankrls_lqo":{
                "learner":"LeaveQueryOutRankRLS",
                "lpath":"rlscore.learner.query_rankrls",
                "measure":"cindex",
                "lfparams": defqparams,
                "lparams": {},
                "files": rqfiles,
                "selection":True},
               
                "rankrls_cg":{
                "learner":"CGRankRLS",
                "lpath":"rlscore.learner.cg_rankrls",
                "measure":"cindex",
                "lfparams": defparams,
                "lparams": {"regparam":1},
                "files": rfiles},
               
                "rankrls_cg_qids":{
                "learner":"CGRankRLS",
                "lpath":"rlscore.learner.cg_rankrls",
                "measure":"cindex",
                "lfparams": defqparams,
                "lparams": {"regparam":1},
                "files": rqfiles},
               
                "rankrls_cg_preferences":{
                "learner":"PCGRankRLS",
                "lpath":"rlscore.learner.cg_rankrls",
                "measure":"cindex",
                "lfparams": {"X":"train_features", "train_preferences":"train_preferences"},
                "lparams": {"regparam":1},
                "files": dict(rfiles.items() + [("train_preferences", './legacy_tests/data/rank_train.preferences') ])},                
               
               }
