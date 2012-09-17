from rlscore.mselection.abstract_selection import AbstractSelection
from rlscore.measure import measure_utilities

class LOOSelection(AbstractSelection):
    """Leave-one-out cross-validation for model selection"""

    def estimatePerformance(self, model):
        """Returns the leave-one-out estimate
        
        @param model: trained learner object
        @type model: RLS
        @return: estimated performance for the model
        @rtype: float"""
        Y_pred = model.computeLOO()
        #performance = self.measure.multiOutputPerformance(self.Y, Y_pred)
        #performance = self.measure.getPerformance(self.Y, Y_pred)
        #performance = measure_utilities.aggregate(performance)
        performance = self.measure(self.Y, Y_pred)
        self.predictions.append(Y_pred)
        return performance
