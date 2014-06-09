from rlscore import core
from rlscore import reader
from rlscore import writer
from rlscore.measure import disagreement

kwargs = {}
kwargs['train_labels'] = reader.loadtxt('./examples/data/rank_train.labels')
kwargs['validation_labels'] = reader.loadtxt('./examples/data/rank_test.labels')
kwargs['train_qids'] = reader.read_qids('./examples/data/rank_train.qids')
kwargs['validation_qids'] = reader.read_qids('./examples/data/rank_test.qids')
kwargs['validation_features'] = reader.read_sparse('./examples/data/rank_test.features')
kwargs['train_features'] = reader.read_sparse('./examples/data/rank_train.features')
kwargs['reggrid'] = '0.001 0.1 10 1000'
kwargs['learner'] = 'CGRankRLS'
kwargs['measure'] = disagreement
kwargs['mselection'] = 'ValidationSetSelection'
trainresults = core.trainModel(**kwargs)
model = trainresults['model']
writer.write_pickle('./examples/models/rankqids.model', model)
