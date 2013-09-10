from distutils.core import setup
from distutils.extension import Extension
from distutils.command import build_ext
import numpy as np
import sys

sys.argv[1:] = ['build_ext', '--inplace']

ext_modules = [Extension(
    name="rlscore.utilities.swapped",
    sources=["ext_src/swapped.c", "ext_src/c_swapped.c"],
    language="c", include_dirs=[np.get_include()]),
    Extension("rlscore.learner.cython_mmc",["rlscore/learner/cython_mmc.c"], include_dirs=[np.get_include()]),
    Extension("rlscore.learner.cython_greedy_rls",["rlscore/learner/cython_greedy_rls.c"], include_dirs=[np.get_include()]),
    Extension("rlscore.utilities.sparse_kronecker_multiplication_tools",["rlscore/utilities/sparse_kronecker_multiplication_tools.c"], include_dirs=[np.get_include()])
]

setup(
    name = 'RLScore',
    ext_modules = ext_modules,
    )


