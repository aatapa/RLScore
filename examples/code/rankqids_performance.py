from rlscore import core
from rlscore import reader
from rlscore import writer
from rlscore.measure import disagreement
from numpy import mean

kwargs = {}
test_labels = reader.loadtxt('./examples/data/rank_test.labels')
predicted_labels = reader.loadtxt('./examples/predictions/rankqids.predictions')
test_qids = reader.read_qids('./examples/data/rank_test.qids')
kwargs['measure'] = disagreement
mselector = None
print 'calculating performance as averages over queries'
performances = []
for query in test_qids:
    performances.append(disagreement(test_labels[query], predicted_labels[query]))
performance = mean(performances)
print 'Performance: %f %s' % (performance, disagreement.__name__)
