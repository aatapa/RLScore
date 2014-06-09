from rlscore import core
from rlscore import reader
from rlscore import writer
from rlscore.measure import accuracy

kwargs = {}
kwargs['train_labels'] = reader.loadtxt('./examples/data/class_train.labels')
prediction_features = reader.read_sparse('./examples/data/class_test.features')
kwargs['train_features'] = reader.read_sparse('./examples/data/class_train.features')
test_labels = reader.loadtxt('./examples/data/class_test.labels')
kwargs['regparam'] = '1'
kwargs['subsetsize'] = '3'
kwargs['bias'] = '1'
kwargs['learner'] = 'GreedyRLS'
kwargs['measure'] = accuracy
mselector = None
trainresults = core.trainModel(**kwargs)
model = trainresults['model']
print 'Making predictions on test data'
predicted_labels = model.predict(prediction_features)
performance = accuracy(test_labels, predicted_labels)
print 'Performance: %f %s' % (performance, accuracy.__name__)
writer.write_pickle('./examples/models/sparse.model', model)
writer.write_dense('./examples/predictions/GreedyRLS_LOO.performance', trainresults['GreedyRLS_LOO_performances'])
writer.write_ints('./examples/predictions/selected.findices', trainresults['selected_features'])
