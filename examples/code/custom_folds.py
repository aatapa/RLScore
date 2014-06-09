from rlscore import core
from rlscore import reader
from rlscore import writer
from rlscore.measure import sqerror

kwargs = {}
kwargs['train_labels'] = reader.loadtxt('./examples/data/reg_train.labels')
prediction_features = reader.read_sparse('./examples/data/reg_test.features')
kwargs['cross-validation_folds'] = reader.read_folds('./examples/data/folds.txt')
kwargs['train_features'] = reader.read_sparse('./examples/data/reg_train.features')
test_labels = reader.loadtxt('./examples/data/reg_test.labels')
kwargs['reggrid'] = '-10_10'
kwargs['bias'] = '1'
kwargs['kernel'] = 'LinearKernel'
kwargs['learner'] = 'RLS'
kwargs['measure'] = sqerror
kwargs['mselection'] = 'NfoldSelection'
trainresults = core.trainModel(**kwargs)
model = trainresults['model']
print 'Making predictions on test data'
predicted_labels = model.predict(prediction_features)
performance = sqerror(test_labels, predicted_labels)
print 'Performance: %f %s' % (performance, sqerror.__name__)
writer.write_pickle('./examples/models/reg.model', model)
writer.write_dense('./examples/predictions/reg.predictions', predicted_labels)
