from rlscore import core
from rlscore import reader
from rlscore import writer
from rlscore.measure import accuracy

kwargs = {}
kwargs['train_labels'] = reader.loadtxt('./examples/data/class_train.labels')
kwargs['train_features'] = reader.read_sparse('./examples/data/class_train.features')
kwargs['reggrid'] = '-10_10'
kwargs['bias'] = '1'
kwargs['kernel'] = 'LinearKernel'
kwargs['learner'] = 'RLS'
kwargs['measure'] = accuracy
kwargs['mselection'] = 'LOOSelection'
trainresults = core.trainModel(**kwargs)
model = trainresults['model']
writer.write_pickle('./examples/models/classacc.model', model)
