import numpy as np
from rlscore.learner.cg_rls import CGRLS
from rlscore.learner.cg_rls import EarlyStopCB
from rlscore.utilities.reader import read_sparse
from rlscore.measure import auc
train_labels = np.loadtxt("./legacy_tests/data/class_train.labels")
test_labels = np.loadtxt("./legacy_tests/data/class_test.labels")
train_features = read_sparse("./legacy_tests/data/class_train.features")
test_features = read_sparse("./legacy_tests/data/class_test.features")
kwargs = {}
kwargs["Y"] = train_labels
kwargs["X"] = train_features
kwargs["regparam"] = 1
kwargs["callbackfun"] = EarlyStopCB(test_features, test_labels, measure=auc)
learner = CGRLS(**kwargs)
P = learner.predict(test_features)
test_perf = auc(test_labels, P)
print("test set performance: %f" %test_perf)