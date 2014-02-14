import numpy as np

from rlscore import data_sources
from rlscore.utilities import creators
from rlscore.kernel import LinearKernel
from rlscore.utilities.adapter import SvdAdapter
from rlscore.utilities.adapter import LinearSvdAdapter
from rlscore.utilities.adapter import PreloadedKernelMatrixSvdAdapter
from rlscore.utilities import array_tools


class AbstractLearner(object):
    '''Base class for learning algorithms'''
    
    
    def createLearner(cls, **kwargs):
        learner = cls()
        learner.resource_pool = kwargs
        learner.loadResources()
        return learner
    createLearner = classmethod(createLearner)
    
    
    def loadResources(self):
        pass
    
    
    def train(self):
        """Trains the learning algorithm.
        
        After the learner is trained, one can call the method getModel
        to get the trained model
        """
        pass
    
    
    def getModel(self):
        """Returns the trained model, call this only after training.
        
        Returns
        -------
        model : {LinearModel, DualModel}
            prediction function
        """
        raise Exception("AbstractLearner does not have an implemented getModel function.")

class AbstractSupervisedLearner(AbstractLearner):
    '''Base class for supervised learning algorithms'''
    
    
    def loadResources(self):
        Y = self.resource_pool[data_sources.TRAIN_LABELS]
        self.setLabels(Y)
    
    
    def setLabels(self, Y):
        self.Y = array_tools.as_labelmatrix(Y)
        self.size = self.Y.shape[0]
        self.ysize = self.Y.shape[1]


class AbstractSvdLearner(AbstractLearner):
    """Base class for singular value decomposition based learners"""
    
    def loadResources(self):
        #THE GREAT SVD MONOLITH!!!
        if self.resource_pool.has_key(data_sources.KMATRIX):
            self.svdad = PreloadedKernelMatrixSvdAdapter.createAdapter(**self.resource_pool)
        else:
            if not self.resource_pool.has_key(data_sources.KERNEL_OBJ):
                if not self.resource_pool.has_key("kernel"):
                    self.resource_pool["kernel"] = "LinearKernel"
                self.resource_pool[data_sources.KERNEL_OBJ] = creators.createKernelByModuleName(**self.resource_pool)
            if isinstance(self.resource_pool[data_sources.KERNEL_OBJ], LinearKernel):
                self.svdad = LinearSvdAdapter.createAdapter(**self.resource_pool)
            else:
                self.svdad = SvdAdapter.createAdapter(**self.resource_pool)
        self.svals = self.svdad.svals
        self.svecs = self.svdad.rsvecs
        if not self.resource_pool.has_key(data_sources.TIKHONOV_REGULARIZATION_PARAMETER):
            self.resource_pool[data_sources.TIKHONOV_REGULARIZATION_PARAMETER] = 1.
        self.size = self.svecs.shape[0]
    
    
    def getModel(self):
        """Returns the trained model, call this only after training.
        
        Returns
        -------
        model : {LinearModel, DualModel}
            prediction function
        """
        model = self.svdad.createModel(self)
        return model


class AbstractSvdSupervisedLearner(AbstractSupervisedLearner,AbstractSvdLearner):
    """Base class for supervised singular value decomposition based learners"""
    
    
    def loadResources(self):
        
        AbstractSupervisedLearner.loadResources(self)
        AbstractSvdLearner.loadResources(self)
        
        if self.size != self.svecs.shape[0]:
            tivc = str(self.svecs.shape[0])
            tlc = str(self.size)
            raise Exception('The number ' + tivc + ' of training feature vectors is different from the number ' + tlc + ' of training labels.')
    
    
    def train(self):
        """Trains the learning algorithm.
        
        After the learner is trained, one can call the method getModel
        to get the trained model
        """
        #regparam = float(self.resource_pool[data_sources.TIKHONOV_REGULARIZATION_PARAMETER])
        regparam = self.regparam
        self.solve(regparam)
        
    
    def solve(self, regparam):
        """Trains the learning algorithm, using the given regularization parameter.
        
        Parameters
        ----------
        regparam: float (regparam > 0)
            regularization parameter
        """
        pass
    
class CallbackFunction(object):
    
    def callback(self, learner):
        pass
    
    
    def finished(self, learner):
        pass



class AbstractIterativeLearner(AbstractLearner):
    """Base class for iterative learners"""
    
    
    def loadResources(self):
        if self.resource_pool.has_key(data_sources.CALLBACK_FUNCTION):
            self.callbackfun = self.resource_pool[data_sources.CALLBACK_FUNCTION]
        else:
            self.callbackfun = None

    
    
    def callback(self):
        if not self.callbackfun == None:
            self.callbackfun.callback(self)
    
    
    def finished(self):
        if not self.callbackfun == None:
            self.callbackfun.finished(self)
