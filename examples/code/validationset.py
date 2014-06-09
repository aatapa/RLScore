from rlscore import core
from rlscore import reader
from rlscore import writer
from rlscore.measure import auc

kwargs = {}
kwargs['train_labels'] = reader.loadtxt('./examples/data/class_train.labels')
kwargs['validation_labels'] = reader.loadtxt('./examples/data/class_test.labels')
kwargs['validation_features'] = reader.read_sparse('./examples/data/class_test.features')
kwargs['basis_vectors'] = reader.loadtxtint('./examples/data/bvectors.indices')
kwargs['train_features'] = reader.read_sparse('./examples/data/class_train.features')
kwargs['reggrid'] = '0.001 1 100 10000 1000000000'
kwargs['bias'] = '1'
kwargs['gamma'] = '0.001'
kwargs['kernel'] = 'GaussianKernel'
kwargs['learner'] = 'RLS'
kwargs['measure'] = auc
kwargs['mselection'] = 'ValidationSetSelection'
trainresults = core.trainModel(**kwargs)
model = trainresults['model']
writer.write_dense('./examples/misc/mselect_perfs.txt', trainresults['mselection_performances'])
