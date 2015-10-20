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

def createSVDAdapter(X, kernel="LinearKernel", **kwargs):
        kwargs[KERNEL_NAME] = kernel
        if kernel == "precomputed":
            kwargs["kernel_matrix"] = X
            svdad = PreloadedKernelMatrixSvdAdapter.createAdapter(**kwargs)        
        else:
            kwargs['X'] = X
            kwargs['kernel_obj'] = createKernelByModuleName(**kwargs)
            if isinstance(kwargs['kernel_obj'], LinearKernel):
                svdad = LinearSvdAdapter.createAdapter(**kwargs)
            else:
                svdad = SvdAdapter.createAdapter(**kwargs)
        return svdad