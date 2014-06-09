from rlscore import core
from rlscore import reader
from rlscore import writer
from rlscore.measure import sqerror

kwargs = {}
test_labels = reader.loadtxt('./examples/data/reg_test.labels')
predicted_labels = reader.loadtxt('./examples/predictions/reg.predictions')
kwargs['measure'] = sqerror
mselector = None
performance = sqerror(test_labels, predicted_labels)
print 'Performance: %f %s' % (performance, sqerror.__name__)
