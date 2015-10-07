import numpy as np
from rlscore.learner.mmc import MMC
from rlscore.reader import read_sparse
from rlscore.reader import read_sparse
from rlscore.measure import auc
train_labels = np.loadtxt("./examples/data/class_train.labels")
test_labels = np.loadtxt("./examples/data/class_test.labels")
train_features = read_sparse("./examples/data/class_train.features")
test_features = read_sparse("./examples/data/class_test.features")
kwargs = {}
kwargs["Y"] = train_labels
kwargs["X"] = train_features
kwargs["regparam"] = 1
learner = MMC.createLearner(**kwargs)
learner.train()
model = learner.getModel()
P = model.predict(test_features)
test_perf = auc(test_labels, P)
print "test set performance: %f" %test_perf
