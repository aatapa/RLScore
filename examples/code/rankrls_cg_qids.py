import numpy as np
from rlscore.learner.cg_rankrls import CGRankRLS
from rlscore.reader import read_qids
from rlscore.reader import read_sparse
from rlscore.reader import read_sparse
from rlscore.reader import read_qids
from rlscore.measure import cindex
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
learner = CGRankRLS.createLearner(**kwargs)
learner.train()
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
