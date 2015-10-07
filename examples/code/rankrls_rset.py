import numpy as np
from rlscore.learner.all_pairs_rankrls import AllPairsRankRLS
from rlscore.reader import read_sparse
from rlscore.reader import read_sparse
from rlscore.measure import cindex
from rlscore.learner.all_pairs_rankrls import LPOCV
from rlscore.utilities.grid_search import grid_search
train_labels = np.loadtxt("./examples/data/rank_train.labels")
test_labels = np.loadtxt("./examples/data/rank_test.labels")
basis_vectors = np.loadtxt("./examples/data/bvectors.indices")
train_features = read_sparse("./examples/data/rank_train.features")
test_features = read_sparse("./examples/data/rank_test.features")
kwargs = {}
kwargs["Y"] = train_labels
kwargs["X"] = train_features
kwargs["basis_vectors"] = train_features[basis_vectors]
kwargs["regparam"] = 1
kwargs["coef0"] = 1
kwargs["degree"] = 3
kwargs["gamma"] = 2
kwargs["kernel"] = "PolynomialKernel"
learner = AllPairsRankRLS(**kwargs)
learner.train()
kwargs = {}
kwargs["learner"] = learner
kwargs["measure"] = cindex
crossvalidator = LPOCV(**kwargs)
grid = [2**i for i in range(-10,11)]
learner, perfs = grid_search(crossvalidator, grid)
for i in range(len(grid)):
    print "parameter %f cv_performance %f" %(grid[i], perfs[i])
model = learner.getModel()
P = model.predict(test_features)
test_perf = cindex(test_labels, P)
print "test set performance: %f" %test_perf
