import numpy as np
from rlscore.learner.query_rankrls import LeaveQueryOutRankRLS
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
kwargs['measure']=cindex
kwargs['regparams'] = [2**i for i in range(-10,11)]
kwargs["Y"] = train_labels
kwargs["X"] = train_features
kwargs["qids"] = train_qids
learner = LeaveQueryOutRankRLS(**kwargs)
grid = kwargs['regparams']
perfs = learner.cv_performances
for i in range(len(grid)):
    print "parameter %f cv_performance %f" %(grid[i], perfs[i])
P = learner.predict(test_features)
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
