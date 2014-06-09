from rlscore import core
from rlscore import reader
from rlscore import writer

kwargs = {}
prediction_features = reader.read_sparse('./examples/data/reg_test.features')
model = reader.read_pickle('./examples/models/reg.model')
mselector = None
print 'Making predictions on test data'
predicted_labels = model.predict(prediction_features)
writer.write_dense('./examples/predictions/reg.predictions', predicted_labels)
