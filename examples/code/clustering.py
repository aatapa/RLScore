from rlscore import core
from rlscore import reader
from rlscore import writer

kwargs = {}
kwargs['train_features'] = reader.read_sparse('./examples/data/class_train.features')
kwargs['regparam'] = '1'
kwargs['bias'] = '1'
kwargs['number_of_clusters'] = '2'
kwargs['learner'] = 'MMC'
mselector = None
trainresults = core.trainModel(**kwargs)
model = trainresults['model']
writer.write_ints('./examples/predictions/clusters.txt', trainresults['predicted_clusters_for_training_data'])
