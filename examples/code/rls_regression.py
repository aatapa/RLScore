import numpy as np
from rlscore.learner.rls import RLS
from rlscore.reader import read_sparse
from rlscore.reader import read_sparse
from rlscore.measure import sqerror
from rlscore.learner.rls import LOOCV
from rlscore.utilities.grid_search import grid_search
train_labels = np.loadtxt("./examples/data/reg_train.labels")
test_labels = np.loadtxt("./examples/data/reg_test.labels")
train_features = read_sparse("./examples/data/reg_train.features")
test_features = read_sparse("./examples/data/reg_test.features")
kwargs = {}
kwargs["Y"] = train_labels
kwargs["X"] = train_features
kwargs["regparam"] = 1
learner = RLS.createLearner(**kwargs)
learner.train()
kwargs = {}
kwargs["learner"] = learner
kwargs["measure"] = sqerror
crossvalidator = LOOCV(**kwargs)
grid = [2**i for i in range(-10,11)]
learner, perfs = grid_search(crossvalidator, grid)
for i in range(len(grid)):
    print "parameter %f cv_performance %f" %(grid[i], perfs[i])
model = learner.getModel()
P = model.predict(test_features)
test_perf = sqerror(test_labels, P)
print "test set performance: %f" %test_perf
