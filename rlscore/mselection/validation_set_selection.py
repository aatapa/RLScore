import numpy as np

from rlscore.mselection.abstract_selection import AbstractSelection
from rlscore import data_sources
from rlscore import model
from rlscore.measure import measure_utilities


class ValidationSetSelection(AbstractSelection):
    """Model selection using a validation set"""
    
    def loadResources(self):
        """Loads in the resources in resource pool. If folds are present
        in the resource pool, they will be used instead of randomly
        split tenfold.
        """
        AbstractSelection.loadResources(self)
        if not self.resource_pool.has_key(data_sources.VALIDATION_FEATURES):
            raise Exception("ValidationSetSelection cannot be initialized without validation set features provided")
        self.validation_X = self.resource_pool[data_sources.VALIDATION_FEATURES]
        if not self.resource_pool.has_key(data_sources.VALIDATION_LABELS):
            raise Exception("ValidationSetSelection cannot be initialized without validation set labels provided")
        self.validation_Y = self.resource_pool[data_sources.VALIDATION_LABELS]
        if self.resource_pool.has_key(data_sources.VALIDATION_QIDS):
            self.validation_qids = self.resource_pool[data_sources.VALIDATION_QIDS]
        else:
            self.validation_qids =  None
        self.K = None
    
    def estimatePerformance(self, learner):
        """Returns the performance on validation set
        
        @param model: trained learner object
        @type model: RLS
        @return: estimated performance for the model
        @rtype: float"""
        #A hack to avoid re-computing the kernel matrix for the validation set
        #for each regparam value
        mod= learner.getModel()
        if isinstance(mod, model.DualModel):
            if self.K == None:
                self.K = mod.kernel.getKM(self.validation_X)
            P = np.dot(self.K.T, mod.A)
        else:        
            P = mod.predict(self.validation_X)
        #performance = self.measure.multiVariatePerformance(self.validation_Y, P, self.validation_qids)
        if self.validation_qids != None:
            perfs = []
            for query in self.validation_qids:
                perf = self.measure(self.validation_Y[query], P[query])
                perfs.append(perf)
            performance = np.mean(perfs)
            #performance = measure_utilities.wrapper(self.measure, self.validation_Y, P, self.validation_qids)
        else:
            performance = self.measure(self.validation_Y, P)
            #performance = self.measure.getPerformance(self.validation_Y, P)
        self.predictions.append(P)
        #performance = measure_utilities.aggregate(performance)
        return performance