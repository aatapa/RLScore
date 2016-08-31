cfiles = {"train_features":'./legacy_tests/data/class_train.features',
                          "train_labels":'./legacy_tests/data/class_train.labels',
                          "test_features":'./legacy_tests/data/class_test.features',
                          "test_labels":'./legacy_tests/data/class_test.labels'}

defparams = {"X":"train_features", "Y":"train_labels"}

experiments = {"mmc_defparams":{
                "learner":"MMC",
                "lpath":"rlscore.learner.mmc",
                "measure":"auc",
                "lfparams": defparams,
                "lparams": {"regparam":1},
                "files": cfiles},
               }