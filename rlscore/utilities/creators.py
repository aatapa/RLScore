from rlscore import data_sources
from rlscore.kernel import LinearKernel
from rlscore.utilities.adapter import SvdAdapter
from rlscore.utilities.adapter import LinearSvdAdapter
from rlscore.utilities.adapter import PreloadedKernelMatrixSvdAdapter

KERNEL_NAME = 'kernel'

def createKernelByModuleName(**kwargs):
    kname = kwargs[KERNEL_NAME]
    exec "from ..kernel import " + kname
    pcgstr = "kernel."
    kernelclazz = eval(kname)
    kernel = kernelclazz.createKernel(**kwargs)
    return kernel

def createSVDAdapter(**kwargs):
    if kwargs.has_key(KERNEL_NAME):
        kernel = createKernelByModuleName(**kwargs)
        kwargs[data_sources.KERNEL_OBJ] = kernel
    if kwargs.has_key(data_sources.KMATRIX):
        svdad = PreloadedKernelMatrixSvdAdapter.createAdapter(**kwargs)
    else:
        if not kwargs.has_key(data_sources.KERNEL_OBJ):
            if not kwargs.has_key("kernel"):
                kwargs["kernel"] = "LinearKernel"
            kwargs[data_sources.KERNEL_OBJ] = createKernelByModuleName(**kwargs)
        if isinstance(kwargs[data_sources.KERNEL_OBJ], LinearKernel):
            svdad = LinearSvdAdapter.createAdapter(**kwargs)
        else:
            svdad = SvdAdapter.createAdapter(**kwargs)   
    return svdad

def createLearnerByModuleName(**kwargs):
    lname = kwargs['learner']
    exec "from rlscore.learner import " + lname
    learnerclazz = eval(lname)
    learner = learnerclazz.createLearner(**kwargs)
    return learner