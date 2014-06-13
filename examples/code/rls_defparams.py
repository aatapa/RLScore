import numpy as np
from rlscore.learner.rls import RLS
from rlscore.reader import read_sparse
from rlscore.reader import read_sparse
from rlscore.measure import auc
train_labels = np.loadtxt("./examples/data/class_train.labels")
test_labels = np.loadtxt("./examples/data/class_test.labels")
train_features = read_sparse("./examples/data/class_train.features")
test_features = read_sparse("./examples/data/class_test.features")
kwargs = {}
kwargs["train_labels"] = train_labels
kwargs["train_features"] = train_features
kwargs["regparam"] = 1
learner = RLS.createLearner(**kwargs)
learner.train()
model = learner.getModel()
P = model.predict(test_features)
test_perf = auc(test_labels, P)
print "test set performance: %f" %test_perf
