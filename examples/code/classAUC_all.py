from rlscore import core
from rlscore import reader
from rlscore import writer
from rlscore.measure import auc

kwargs = {}
kwargs['train_labels'] = reader.loadtxt('./examples/data/class_train.labels')
prediction_features = reader.read_sparse('./examples/data/class_test.features')
kwargs['train_features'] = reader.read_sparse('./examples/data/class_train.features')
test_labels = reader.loadtxt('./examples/data/class_test.labels')
kwargs['reggrid'] = '-10_10'
kwargs['bias'] = '1'
kwargs['kernel'] = 'LinearKernel'
kwargs['learner'] = 'AllPairsRankRLS'
kwargs['measure'] = auc
kwargs['mselection'] = 'LPOSelection'
trainresults = core.trainModel(**kwargs)
model = trainresults['model']
print 'Making predictions on test data'
predicted_labels = model.predict(prediction_features)
performance = auc(test_labels, predicted_labels)
print 'Performance: %f %s' % (performance, auc.__name__)
writer.write_pickle('./examples/models/classAUC.model', model)
writer.write_dense('./examples/predictions/classAUC.predictions', predicted_labels)
