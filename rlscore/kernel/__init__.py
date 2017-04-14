from .gaussian_kernel import GaussianKernel
from .linear_kernel import LinearKernel
from .polynomial_kernel import PolynomialKernel
from .rset_kernel import RsetKernel

def createKernelByModuleName(**kwargs):
    import inspect
    kname = kwargs["kernel"]
    exec("from ..kernel import " + kname)
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
    if "basis_vectors" in kwargs:
        new_kwargs['X'] = kwargs['basis_vectors']
    kernel = kernelclazz(**new_kwargs)
    return kernel