from rlscore.kernel import LinearKernel
from rlscore.utilities.adapter import SvdAdapter
from rlscore.utilities.adapter import LinearSvdAdapter
from rlscore.utilities.adapter import PreloadedKernelMatrixSvdAdapter
import inspect

KERNEL_NAME = 'kernel'

def createKernelByModuleName(**kwargs):
    kname = kwargs[KERNEL_NAME]
    exec "from ..kernel import " + kname
    kernelclazz = eval(kname)
    #get kernel arguments
    args = inspect.getargspec(kernelclazz.__init__)[0]
    args = set(kwargs.keys()).intersection(set(args))
    #filter unnecessary arguments
    new_kwargs = {}
    for key in kwargs.keys():
        if key in args:
            new_kwargs[key] = kwargs[key]
    #initialize kernel
    kernel = kernelclazz(**new_kwargs)
    return kernel

def createSVDAdapter(**kwargs):
    if kwargs.has_key(KERNEL_NAME):
        kernel = createKernelByModuleName(**kwargs)
        kwargs['kernel_obj'] = kernel
    if kwargs.has_key('kernel_matrix'):
        svdad = PreloadedKernelMatrixSvdAdapter.createAdapter(**kwargs)
    else:
        if not kwargs.has_key('kernel_obj'):
            if not kwargs.has_key("kernel"):
                kwargs["kernel"] = "LinearKernel"
            kwargs['kernel_obj'] = createKernelByModuleName(**kwargs)
        if isinstance(kwargs['kernel_obj'], LinearKernel):
            svdad = LinearSvdAdapter.createAdapter(**kwargs)
        else:
            svdad = SvdAdapter.createAdapter(**kwargs)   
    return svdad

def createLearnerByModuleName(**kwargs):
    lname = kwargs['learner']
    exec "from rlscore.learner import " + lname
    learnerclazz = eval(lname)
    learner = learnerclazz(**kwargs)
    return learner