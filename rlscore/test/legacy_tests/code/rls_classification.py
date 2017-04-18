import numpy as np
from rlscore.learner.rls import LeaveOneOutRLS
from rlscore.utilities.reader import read_sparse
from rlscore.measure import auc
train_labels = np.loadtxt("./legacy_tests/data/class_train.labels")
test_labels = np.loadtxt("./legacy_tests/data/class_test.labels")
train_features = read_sparse("./legacy_tests/data/class_train.features")
test_features = read_sparse("./legacy_tests/data/class_test.features")
kwargs = {}
kwargs['measure']=auc
kwargs['regparams'] = [2**i for i in range(-10,11)]
kwargs["Y"] = train_labels
kwargs["X"] = train_features
learner = LeaveOneOutRLS(**kwargs)
grid = kwargs['regparams']
perfs = learner.cv_performances
for i in range(len(grid)):
    print("parameter %f cv_performance %f" %(grid[i], perfs[i]))
P = learner.predict(test_features)
test_perf = auc(test_labels, P)
print("test set performance: %f" %test_perf)
