from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np

ext_modules = [Extension("rlscore.utilities.swapped",["rlscore/utilities/swapped.pyx"], include_dirs=[np.get_include()] ),
    Extension("rlscore.learner.cython_mmc",["rlscore/learner/cython_mmc.pyx"], include_dirs=[np.get_include()]),
    Extension("rlscore.learner.cython_greedy_rls",["rlscore/learner/cython_greedy_rls.pyx"], include_dirs=[np.get_include()]),
    Extension("rlscore.utilities.sparse_kronecker_multiplication_tools",["rlscore/utilities/sparse_kronecker_multiplication_tools.pyx"], include_dirs=[np.get_include()])]

setup(
    name = 'cmodules',
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules,
    )

