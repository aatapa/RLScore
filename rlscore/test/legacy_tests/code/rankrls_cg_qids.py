import numpy as np
from rlscore.learner.cg_rankrls import CGRankRLS
from rlscore.utilities.reader import read_qids
from rlscore.utilities.reader import read_sparse
from rlscore.measure import cindex
train_labels = np.loadtxt("./legacy_tests/data/rank_train.labels")
test_labels = np.loadtxt("./legacy_tests/data/rank_test.labels")
train_qids = read_qids("./legacy_tests/data/rank_train.qids")
test_features = read_sparse("./legacy_tests/data/rank_test.features")
train_features = read_sparse("./legacy_tests/data/rank_train.features")
test_qids = read_qids("./legacy_tests/data/rank_test.qids")
kwargs = {}
kwargs["Y"] = train_labels
kwargs["X"] = train_features
kwargs["qids"] = train_qids
kwargs["regparam"] = 1
learner = CGRankRLS(**kwargs)
P = learner.predict(test_features)
from rlscore.measure.measure_utilities import UndefinedPerformance
from rlscore.measure.measure_utilities import qids_to_splits
test_qids = qids_to_splits(test_qids)
perfs = []
for query in test_qids:
    try:
        perf = cindex(test_labels[query], P[query])
        perfs.append(perf)
    except UndefinedPerformance:
        pass
test_perf = np.mean(perfs)
print("test set performance: %f" %test_perf)
