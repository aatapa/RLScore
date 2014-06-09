from rlscore.learner import KronRLS
from rlscore import reader
from rlscore import writer
from rlscore.measure import sqerror

kwargs = {}
kwargs['train_labels'] = reader.read_dense('./examples/data/Kron_train.labels')
kwargs['kmatrix1'] = reader.read_dense('./examples/data/Kron_train_1.kernelm')
kwargs['kmatrix2'] = reader.read_dense('./examples/data/Kron_train_2.kernelm')
K_test1 = reader.read_dense('./examples/data/Kron_test_1.kernelm')
K_test2 = reader.read_dense('./examples/data/Kron_test_2.kernelm')
test_labels = reader.read_dense('./examples/data/Kron_test.labels')
kwargs['regparam'] = 0.001
learner = KronRLS.createLearner(**kwargs)
learner.train()
kronmodel = learner.getModel()
kronpred = kronmodel.predictWithKernelMatrices(K_test1, K_test2)
print sqerror(test_labels, kronpred)
