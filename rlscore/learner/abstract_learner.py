import numpy as np

from rlscore.utilities import creators
from rlscore.kernel import LinearKernel
from rlscore.utilities.adapter import SvdAdapter
from rlscore.utilities.adapter import LinearSvdAdapter
from rlscore.utilities.adapter import PreloadedKernelMatrixSvdAdapter
from rlscore.utilities import array_tools


class AbstractLearner(object):
    '''Base class for learning algorithms'''
    
    def __init__(self, **kwargs):
        super(AbstractLearner, self).__init__()
    
    
    def createLearner(cls, **kwargs):
        learner = cls(**kwargs)
        return learner
    createLearner = classmethod(createLearner)
    
    
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
    
    def __init__(self, **kwargs):
        super(AbstractSupervisedLearner, self).__init__(**kwargs)
        Y = kwargs['train_labels']
        self.Y = array_tools.as_labelmatrix(Y)
        self.size = self.Y.shape[0]
        self.ysize = self.Y.shape[1]


class AbstractSvdLearner(AbstractLearner):
    """Base class for singular value decomposition based learners"""
    
    def __init__(self, **kwargs):
        super(AbstractSvdLearner, self).__init__(**kwargs)
        #THE GREAT SVD MONOLITH!!!
        if kwargs.has_key('kernel_matrix'):
            self.svdad = PreloadedKernelMatrixSvdAdapter.createAdapter(**kwargs)
        else:
            if not kwargs.has_key('kernel_obj'):
                if not kwargs.has_key("kernel"):
                    kwargs["kernel"] = "LinearKernel"
                kwargs['kernel_obj'] = creators.createKernelByModuleName(**kwargs)
            if isinstance(kwargs['kernel_obj'], LinearKernel):
                self.svdad = LinearSvdAdapter.createAdapter(**kwargs)
            else:
                self.svdad = SvdAdapter.createAdapter(**kwargs)
        self.svals = self.svdad.svals
        self.svecs = self.svdad.rsvecs
        #if not kwargs.has_key('regparam'):
        #    kwargs['regparam'] = 1.
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
    
    def __init__(self, **kwargs):
        super(AbstractSvdSupervisedLearner, self).__init__(**kwargs)
        
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
    
    def __init__(self, **kwargs):
        super(AbstractIterativeLearner, self).__init__(**kwargs)
        if kwargs.has_key('callback'):
            self.callbackfun = kwargs['callback']
        else:
            self.callbackfun = None
    
    
    def callback(self):
        if not self.callbackfun == None:
            self.callbackfun.callback(self)
    
    
    def finished(self):
        if not self.callbackfun == None:
            self.callbackfun.finished(self)
