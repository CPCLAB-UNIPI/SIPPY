# Welcome to SIPPY

## Systems Identification Package for PYthon (SIPPY)

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

## System dependencies

The code has been implemented in [Python 3.10](https://www.python.org/downloads/) and requires [CasADi](https://web.casadi.org/get/).
The Slycot package is optional and can speed up validation. It is available at [PyPI](https://pypi.python.org/pypi/slycot/0.2.0) or alternatively as [binaries](https://www.lfd.uci.edu/~gohlke/pythonlibs/).

## Installation

To install from PyPI, run the following command:

```bash
pip install sippy_unipi
```

To install from the source code, run the following:

```bash
git clone https://github.com/CPCLAB-UNIPI/SIPPY.git
cd SIPPY
pip install -e .
```

We recommend using uv (<https://docs.astral.sh/uv/>) to install the package.

```bash
uv sync
```

## Structure

SIPPY is distributed as package, with following notable items in the structure:

```plaintext
├── user_guide.pdf
│   └── Documentation for Identification_code usage.
├── sippy/
│   ├── __init__.py
│   │   └── Main entry point: contains the function to perform identifications.
│   ├── functionset.py
│   │   └── Core functions used by identification routines and other utilities (see user_guide for details).
│   ├── functionset_OPT.py
│   │   └── Nonlinear optimization routines used by some identification methods.
│   └── functionsetSIM.py
│       └── Additional functions for Subspace identification and state-space model utilities (see user_guide for details).
└── Examples/
    |   └── Examples of usage of the SIPPY package (available with installation from source)
    ├── Ex_ARMAX_MIMO.py
    │   └── Example: Identification_code usage for ARMAX systems (multi input-multi output).
    ├── Ex_ARX_MIMO.py
    │   └── Example: Identification_code usage for ARX systems (multi input-multi output).
    ├── Ex_ARMAX.py
    │   └── Example: Identification_code usage for ARMAX systems (single input-single output, with information criteria).
    ├── SS.py
    │   └── Example: Identification_code usage for State-space systems.
    ├── Ex_OPT_GEN-INOUT.py
    │   └── Example: Input-output structure identification using optimization methods.
    ├── Ex_RECURSIVE.py
    │   └── Example: Input-output structure identification using recursive methods.
    └── Ex_CST.py
        └── Example: Identification_code usage for a Continuous Stirred Tank system.
```
