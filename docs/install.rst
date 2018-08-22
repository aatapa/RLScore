============
Installation
============

Requirements
============

Python 3.5 or newer, NumPy and SciPy packages, and a working c-compiler for building extensions are needed for installing RLScore.

For example `Anaconda <https://www.continuum.io/downloads>`_ is a Python distribution, available for Windows, OS X and Linux, that includes the required dependencies. For Windows, you need to also install `Microsoft Visual C++ Compiler for Python <https://wiki.python.org/moin/WindowsCompilers>`_.

Installation
============

For bleeding edge versions, we recommend cloning the git repository:
`https://github.com/aatapa/RLScore <https://github.com/aatapa/RLScore>`_,
and install using setupy.py, for example:

global installation ::

    python setup.py install 

installing to home directory ::

    python setup.py install --home=<dir>

compile extensions inside source folder (you need to add RLScore folder to PYTHONPATH to use it) ::

    python setup.py build_ext --inplace

Alternatively, it is possible to install the latest pip indexed release:

    pip install rlscore 

