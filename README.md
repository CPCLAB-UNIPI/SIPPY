# Welcome to SIPPY!
### Systems Identification Package for PYthon (SIPPY)

The main objective of this code is to provide different identification methods
to build linear models of dynamic systems, starting from input-output collected
data. The models can be built as transfer functions or state-space models in
discrete-time domain. The Python user has many options in terms of identification
algorithms and in terms of settings to look for the best model.
It is originally developed by Giuseppe Armenise at the Department of Civil and Industrial Engineering of University of Pisa under supervision of [Prof. Gabriele Pannocchia](https://people.unipi.it/gabriele_pannocchia/). The identification code is distributed under the LGPL license, meaning the code can be used royalty-free even in commercial applications.
The developed code is quite simple to use and, having default settings, it can
be used by beginners but also by experts, having many adjustable settings that
can be changed according to the particular case. Furthermore, there are some
functions that the user can use, e.g. to test if the identified system follows the
plant data.
The linear model to be identified can be chosen between:
* input-output structures: FIR, ARX, ARMAX, ARMA, ARARX, ARARMAX, OE, BJ, GEN;
* state-space structures: N4SID, MOESP, CVA, PARSIM-P, PARSIM-S or PARSIM-K.
All the proposed structures are available both in the SISO case, for which the information criteria
are available, and in the MIMO case.

### Installation and package content
The code has been implemented in Python 2.7, compatible with Python 3.7, (download it [here](https://www.python.org/downloads/)) and requires the following packages:
NumPy, SciPy, control (version >= 0.8.2), math, Slycot, Future (See installation instruction [here](http://python-future.org/quickstart.html#installation)), CasADi (see [here](https://web.casadi.org/get/)).
The Slycot package is available [here](https://pypi.python.org/pypi/slycot/0.2.0) or alternatively the binaries can be found [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/).

In order to make the installation easier, the user can simply use the quick command
`python setup.py install`
in order to gather all the required packages all together.

SIPPY is distributed as packed file SIPPY.zip (download it from [here](https://github.com/CPCLAB-UNIPI/SIPPY)) that contains the following items:
* `user_guide.pdf`: documentation for Identification_code usage.
* `sippy/__init__.py`: main file containing the function that has to be recalled to perform the
identifications.
* `Examples/Ex_ARMAX_MIMO.py`: example of usage of the Identification_code for ARMAX systems (multi input-multi output case).
* `Examples/Ex_ARX_MIMO.py`: example of usage of the Identification_code for ARX systems (multi input-multi output case).
* `Examples/Ex_ARMAX.py`: example of usage of the Identification_code for ARMAX systems (single input-single output case, using the information criteria).
* `Examples/SS.py`: example of usage of the Identification_code for State-space systems.
* `Examples/Ex_OPT_GEN-INOUT.py`: example of usage of the Identification_code for the input-output structures using optimization methods.
* `Examples/Ex_RECURSIVE.py`: example of usage of the Identification_code for the input-output structures using recursive methods.
* `Examples/Ex_CST.py`: example of usage of the Identification_code for a Continuous Stirred Tank system.
* `sippy/functionset.py`:  file containing most of the functions used by the identification functions and other useful functions (see the user_guide for the usage).
* `sippy/functionset_OPT.py`: file containing the nonlinear optimization problem used by some of the identification methods.
* `sippy/functionsetSIM.py`: additional functions used by the Subspace identification functions and other useful functions for state space models (see the user_guide for the usage).

In the folder `sippy/` there are other files `.py`, that are called by the main file, so the user has
not to use them.
