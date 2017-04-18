import numpy as np
from rlscore.learner.mmc import MMC
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
learner = MMC(**kwargs)
P = learner.predict(test_features)
test_perf = auc(test_labels, P)
print("test set performance: %f" %test_perf)
