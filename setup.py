from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [Extension(
    name="rlscore.utilities.swapped",
    sources=["ext_src/swapped.pyx", "ext_src/c_swapped.c"],
    language="c",
    )]

setup(
    name = 'swapped',
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules,
    )

