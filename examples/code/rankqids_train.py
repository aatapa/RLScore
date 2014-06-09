from rlscore import core
from rlscore import reader
from rlscore import writer
from rlscore.measure import disagreement

kwargs = {}
kwargs['train_labels'] = reader.loadtxt('./examples/data/rank_train.labels')
kwargs['train_qids'] = reader.read_qids('./examples/data/rank_train.qids')
kwargs['train_features'] = reader.read_sparse('./examples/data/rank_train.features')
kwargs['reggrid'] = '-10_10'
kwargs['bias'] = '1'
kwargs['kernel'] = 'LinearKernel'
kwargs['learner'] = 'LabelRankRLS'
kwargs['measure'] = disagreement
kwargs['mselection'] = 'NfoldSelection'
trainresults = core.trainModel(**kwargs)
model = trainresults['model']
writer.write_pickle('./examples/models/rankqids.model', model)
