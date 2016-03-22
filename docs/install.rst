============
Installation
============

Requirements
============

Python 2.7x, NumPy and SciPy packages, and a working c-compiler for building extensions are needed for installing RLScore. `Anaconda <https://www.continuum.io/downloads>`_ is a recommended Python distribution, available for Windows, OS X and Linux, that includes all the required dependencies.

Installation
============

Easiest way to install RLScore is to use pip (TO BE IMPLEMENTED) ::

    pip install rlscore 

For bleeding edge versions, you can download as zip package, or clone using git the source code from `https://github.com/aatapa/RLScore <https://github.com/aatapa/RLScore>`_, and install using setupy.py, for example:

global installation ::

    python setup.py install 

installing to home directory ::

    python setup.py install --home=<dir>

compile extensions inside source folder (you need to add RLScore folder to PYTHONPATH to use it) ::

    python setup.py build_ext --inplace

