from rlscore import core
from rlscore import reader
from rlscore import writer
from rlscore.measure import disagreement
from numpy import mean

kwargs = {}
kwargs['train_labels'] = reader.loadtxt('./examples/data/rank_train.labels')
test_labels = reader.loadtxt('./examples/data/rank_test.labels')
kwargs['train_qids'] = reader.read_qids('./examples/data/rank_train.qids')
prediction_features = reader.read_sparse('./examples/data/rank_test.features')
kwargs['train_features'] = reader.read_sparse('./examples/data/rank_train.features')
test_qids = reader.read_qids('./examples/data/rank_test.qids')
kwargs['reggrid'] = '-10_10'
kwargs['bias'] = '1'
kwargs['kernel'] = 'LinearKernel'
kwargs['learner'] = 'LabelRankRLS'
kwargs['measure'] = disagreement
kwargs['mselection'] = 'NfoldSelection'
trainresults = core.trainModel(**kwargs)
model = trainresults['model']
print 'Making predictions on test data'
predicted_labels = model.predict(prediction_features)
print 'calculating performance as averages over queries'
performances = []
for query in test_qids:
    performances.append(disagreement(test_labels[query], predicted_labels[query]))
performance = mean(performances)
print 'Performance: %f %s' % (performance, disagreement.__name__)
writer.write_pickle('./examples/models/rankqids.model', model)
writer.write_dense('./examples/predictions/rankqids.predictions', predicted_labels)
