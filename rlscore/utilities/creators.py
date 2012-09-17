from rlscore import data_sources

KERNEL_NAME = 'kernel'

def createKernelByModuleName(**kwargs):
    kname = kwargs[KERNEL_NAME]
    exec "from ..kernel import " + kname
    pcgstr = "kernel."
    kernelclazz = eval(kname)
    kernel = kernelclazz.createKernel(**kwargs)
    return kernel
