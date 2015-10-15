import numpy as np
from rlscore.learner.cg_rankrls import PCGRankRLS
from rlscore.reader import read_sparse
from rlscore.reader import read_sparse
from rlscore.measure import cindex
train_labels = np.loadtxt("./examples/data/rank_train.labels")
test_labels = np.loadtxt("./examples/data/rank_test.labels")
train_preferences = np.loadtxt("./examples/data/rank_train.preferences")
train_features = read_sparse("./examples/data/rank_train.features")
test_features = read_sparse("./examples/data/rank_test.features")
kwargs = {}
kwargs["train_preferences"] = train_preferences
kwargs["X"] = train_features
kwargs["regparam"] = 1
learner = PCGRankRLS(**kwargs)
P = learner.predict(test_features)
test_perf = cindex(test_labels, P)
print "test set performance: %f" %test_perf
