from random import sample
import numpy as np

from rlscore.mselection.abstract_selection import AbstractSelection
from rlscore.measure import measure_utilities
from rlscore.measure.measure_utilities import UndefinedPerformance

class NfoldSelection(AbstractSelection):
    """N-fold cross-validation for model selection"""

    def __init__(self):
        AbstractSelection.__init__(self)
        self.folds = None
        
    def loadResources(self):
        """Loads in the resources in resource pool. If folds are present
        in the resource pool, they will be used instead of randomly
        split tenfold.
        """
        AbstractSelection.loadResources(self)
        if self.resource_pool.has_key('cross-validation_folds'):
            #fs = self.resource_pool['cross-validation_folds']
            #self.folds = fs.readFolds()
            self.folds = self.resource_pool['cross-validation_folds']
        elif self.resource_pool.has_key('train_qids'):
            self.folds = self.resource_pool['train_qids']
            #self.folds = qsource.readFolds()

    def setFolds(self, folds):
        """Sets user supplied folds
        
        @param folds: a list of lists, where each inner list contains the indices of examples belonging to one of the holdout sets
        @type folds: list of lists of integers"""
        self.folds = folds

    def setRandomFolds(self, foldcount):
        """Sets a randomized fold partition
        
        @param foldcount: the number of folds
        @type foldcount: integer"""
        self.folds = []
        indices = set(range(self.Y.shape[0]))
        foldsize = self.Y.shape[0] / foldcount
        leftover = self.Y.shape[0] % foldcount
        for i in range(foldcount):
            sample_size = foldsize
            if leftover > 0:
                sample_size += 1
                leftover -= 1
            fold = sample(indices, sample_size)
            indices = indices.difference(fold)
            self.folds.append(fold)

    def estimatePerformance(self, learner):
        """Estimates the expected N-fold cross-validation performance of a given model.
        By default 10-fold cross-validation with randomized fold partition is used.
        Another schemes can be used by setting the desired number of randomized or
        user defined folds with setFolds or setRandomFolds prior to calling this
        method.
 
        @param learner: trained learner object
        @type learner: RLS
        @return: estimated performance for the model
        @rtype: float"""
        #Default behaviour: random tenfold partition
        if not self.folds:
            self.setRandomFolds(10)
        self.Y_folds = []
        for fold in self.folds:
            self.Y_folds.append(self.Y[fold,:])
        performances = []
        for i in range(len(self.folds)):
            Y_pred = learner.computeHO(self.folds[i])
            #performance = self.measure.getPerformance(self.Y_folds[i], Y_pred)
            #performances.append(measure_utilities.aggregate(performance))
            try:
                performance = self.measure(self.Y_folds[i], Y_pred)
                performances.append(performance)
            except UndefinedPerformance, e:
                pass
        #performance = measure_utilities.aggregate(performances)
        performance = np.mean(performances)
        return performance
        

