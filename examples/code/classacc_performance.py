from rlscore import core
from rlscore import reader
from rlscore import writer
from rlscore.measure import accuracy

kwargs = {}
test_labels = reader.loadtxt('./examples/data/class_test.labels')
predicted_labels = reader.loadtxt('./examples/predictions/classacc.predictions')
kwargs['measure'] = accuracy
mselector = None
performance = accuracy(test_labels, predicted_labels)
print 'Performance: %f %s' % (performance, accuracy.__name__)
