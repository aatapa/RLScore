rfiles = {"train_features":'./examples/data/rank_train.features',
                          "train_labels":'./examples/data/rank_train.labels',
                          "test_features":'./examples/data/rank_test.features',
                          "test_labels":'./examples/data/rank_test.labels'}

rqfiles = {"train_features":'./examples/data/rank_train.features',
           "train_labels":'./examples/data/rank_train.labels',
           "train_qids":'./examples/data/rank_train.qids',
           "test_features":'./examples/data/rank_test.features',
            "test_labels":'./examples/data/rank_test.labels',
            "test_qids":'./examples/data/rank_test.qids',
            }

defparams = {"train_features":"train_features", "train_labels":"train_labels"}
defqparams = {"train_features":"train_features", "train_labels":"train_labels", "train_qids":"train_qids"}

experiments = {"rankrls_defparams":{
                "learner":"AllPairsRankRLS",
                "lpath":"rlscore.learner.all_pairs_rankrls",
                "measure":"cindex",
                "lfparams": defparams,
                "lparams": {"regparam":1},
                "files": rfiles},
               
               "rankrls_lpo":{
                "learner":"AllPairsRankRLS",
                "lpath":"rlscore.learner.all_pairs_rankrls",
                "measure":"cindex",
                "selector":"LPOCV",
                "sparams":{"measure":"cindex"},
                "lfparams": defparams,
                "lparams": {"regparam":1},
                "files": rfiles},
               
                "rankrls_rset":{
                "learner":"AllPairsRankRLS",
                "lpath":"rlscore.learner.all_pairs_rankrls",
                "measure":"cindex",
                "selector":"LPOCV",
                "sparams":{"measure":"cindex"},
                "lfparams": dict(defparams.items()+[("bvectors","bvectors")]),
                "lparams": {"regparam":1, "kernel":"PolynomialKernel", "gamma":2, "coef0":1, "degree":3},
                "files": dict(rfiles.items()+[("bvectors",'./examples/data/bvectors.indices')])},
               
               "rankrls_qids":{
                "learner":"LabelRankRLS",
                "lpath":"rlscore.learner.label_rankrls",
                "measure":"cindex",
                "lfparams": defqparams,
                "lparams": {"regparam":1},
                "files": rqfiles},
               
                "rankrls_lqo":{
                "learner":"LabelRankRLS",
                "lpath":"rlscore.learner.label_rankrls",
                "measure":"cindex",
                "selector":"LQOCV",
                "sparams":{"measure":"cindex"},
                "lfparams": defqparams,
                "lparams": {"regparam":1},
                "files": rqfiles},
               
                "rankrls_cg":{
                "learner":"CGRankRLS",
                "lpath":"rlscore.learner.cg_rankrls",
                "measure":"cindex",
                "sparams":{"measure":"cindex"},
                "lfparams": defparams,
                "lparams": {"regparam":1},
                "files": rfiles},
               
                "rankrls_cg_qids":{
                "learner":"CGRankRLS",
                "lpath":"rlscore.learner.cg_rankrls",
                "measure":"cindex",
                "sparams":{"measure":"cindex"},
                "lfparams": defqparams,
                "lparams": {"regparam":1},
                "files": rqfiles},
               
                "rankrls_cg_preferences":{
                "learner":"CGRankRLS",
                "lpath":"rlscore.learner.cg_rankrls",
                "measure":"cindex",
                "sparams":{"measure":"cindex"},
                "lfparams": dict(defparams.items() + [("train_preferences", "train_preferences")]),
                "lparams": {"regparam":1},
                "files": dict(rfiles.items() + [("train_preferences", './examples/data/rank_train.preferences') ])},                
               
               }