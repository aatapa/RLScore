from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np

ext_modules = [Extension(
    name="rlscore.utilities.swapped",
    sources=["ext_src/swapped.pyx", "ext_src/c_swapped.c"],
    language="c", include_dirs=[np.get_include()]),
    Extension("rlscore.learner.cython_mmc",["rlscore/learner/cython_mmc.pyx"], include_dirs=[np.get_include()]),
    Extension("rlscore.learner.cython_greedy_rls",["rlscore/learner/cython_greedy_rls.pyx"], include_dirs=[np.get_include()]),
    Extension("rlscore.utilities.sparse_kronecker_multiplication_tools",["rlscore/utilities/sparse_kronecker_multiplication_tools.pyx"], include_dirs=[np.get_include()])]

setup(
    name = 'cmodules',
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules,
    )

