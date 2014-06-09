from rlscore import core
from rlscore import reader
from rlscore import writer
from rlscore.measure import auc

kwargs = {}
test_labels = reader.loadtxt('./examples/data/class_test.labels')
predicted_labels = reader.loadtxt('./examples/predictions/classAUC.predictions')
kwargs['measure'] = auc
mselector = None
performance = auc(test_labels, predicted_labels)
print 'Performance: %f %s' % (performance, auc.__name__)
