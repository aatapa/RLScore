from rlscore.utilities import creators
from rlscore.kernel import LinearKernel
from rlscore.utilities.adapter import SvdAdapter
from rlscore.utilities.adapter import LinearSvdAdapter
from rlscore.utilities.adapter import PreloadedKernelMatrixSvdAdapter




class AbstractSvdLearner(object):
    """Base class for singular value decomposition based learners"""
    
    def __init__(self, X, kernel="LinearKernel", **kwargs):
        #THE GREAT SVD MONOLITH!!!
        kwargs["kernel"] = kernel
        if kernel == "precomputed":
            kwargs["kernel_matrix"] = X
            self.svdad = PreloadedKernelMatrixSvdAdapter.createAdapter(**kwargs)        
        else:
            kwargs['X'] = X
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
    
class CallbackFunction(object):
    
    def callback(self, learner):
        pass
    
    
    def finished(self, learner):
        pass
