import numpy as np
from rlscore.learner.cg_rankrls import CGRankRLS
from rlscore.utilities.reader import read_sparse
from rlscore.measure import cindex
train_labels = np.loadtxt("./legacy_tests/data/rank_train.labels")
test_labels = np.loadtxt("./legacy_tests/data/rank_test.labels")
train_features = read_sparse("./legacy_tests/data/rank_train.features")
test_features = read_sparse("./legacy_tests/data/rank_test.features")
kwargs = {}
kwargs["Y"] = train_labels
kwargs["X"] = train_features
kwargs["regparam"] = 1
learner = CGRankRLS(**kwargs)
P = learner.predict(test_features)
test_perf = cindex(test_labels, P)
print("test set performance: %f" %test_perf)
