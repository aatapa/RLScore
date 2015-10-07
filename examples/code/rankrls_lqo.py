import numpy as np
from rlscore.learner.label_rankrls import LabelRankRLS
from rlscore.reader import read_qids
from rlscore.reader import read_sparse
from rlscore.reader import read_sparse
from rlscore.reader import read_qids
from rlscore.measure import cindex
from rlscore.learner.label_rankrls import LQOCV
from rlscore.utilities.grid_search import grid_search
train_labels = np.loadtxt("./examples/data/rank_train.labels")
test_labels = np.loadtxt("./examples/data/rank_test.labels")
train_qids = read_qids("./examples/data/rank_train.qids")
test_features = read_sparse("./examples/data/rank_test.features")
train_features = read_sparse("./examples/data/rank_train.features")
test_qids = read_qids("./examples/data/rank_test.qids")
kwargs = {}
kwargs["Y"] = train_labels
kwargs["X"] = train_features
kwargs["qids"] = train_qids
kwargs["regparam"] = 1
learner = LabelRankRLS(**kwargs)
learner.train()
kwargs = {}
kwargs["learner"] = learner
kwargs["measure"] = cindex
crossvalidator = LQOCV(**kwargs)
grid = [2**i for i in range(-10,11)]
learner, perfs = grid_search(crossvalidator, grid)
for i in range(len(grid)):
    print "parameter %f cv_performance %f" %(grid[i], perfs[i])
model = learner.getModel()
P = model.predict(test_features)
from rlscore.measure.measure_utilities import UndefinedPerformance
perfs = []
for query in test_qids:
    try:
        perf = cindex(test_labels[query], P[query])
        perfs.append(perf)
    except UndefinedPerformance:
        pass
test_perf = np.mean(perfs)
print "test set performance: %f" %test_perf
