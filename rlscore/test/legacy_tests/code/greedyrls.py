import numpy as np
from rlscore.learner.greedy_rls import GreedyRLS
from rlscore.utilities.reader import read_sparse
from rlscore.measure import auc
train_labels = np.loadtxt("./legacy_tests/data/class_train.labels")
test_labels = np.loadtxt("./legacy_tests/data/class_test.labels")
train_features = read_sparse("./legacy_tests/data/class_train.features")
test_features = read_sparse("./legacy_tests/data/class_test.features")
kwargs = {}
kwargs["Y"] = train_labels
kwargs["X"] = train_features
kwargs["test_labels"] = test_labels
kwargs["test_features"] = test_features
kwargs["use_default_callback"] = True
kwargs["regparam"] = 1
kwargs["bias"] = 1
kwargs["test_measure"] = "auc"
kwargs["subsetsize"] = 10
learner = GreedyRLS(**kwargs)
P = learner.predict(test_features)
test_perf = auc(test_labels, P)
print("test set performance: %f" %test_perf)
