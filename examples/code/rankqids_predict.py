from rlscore import core
from rlscore import reader
from rlscore import writer

kwargs = {}
prediction_features = reader.read_sparse('./examples/data/rank_test.features')
model = reader.load('./examples/models/rankqids.model')
mselector = None
print 'Making predictions on test data'
predicted_labels = model.predict(prediction_features)
writer.write_dense('./examples/predictions/rankqids.predictions', predicted_labels)
