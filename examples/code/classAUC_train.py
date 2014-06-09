from rlscore import core
from rlscore import reader
from rlscore import writer
from rlscore.measure import auc

kwargs = {}
kwargs['train_labels'] = reader.loadtxt('./examples/data/class_train.labels')
kwargs['train_features'] = reader.read_sparse('./examples/data/class_train.features')
kwargs['reggrid'] = '-10_10'
kwargs['bias'] = '1'
kwargs['kernel'] = 'LinearKernel'
kwargs['learner'] = 'AllPairsRankRLS'
kwargs['measure'] = auc
kwargs['mselection'] = 'LPOSelection'
trainresults = core.trainModel(**kwargs)
model = trainresults['model']
writer.write_pickle('./examples/models/classAUC.model', model)
