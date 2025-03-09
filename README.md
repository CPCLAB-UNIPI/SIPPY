# Welcome to SIPPY

[![Supported versions](https://img.shields.io/pypi/pyversions/sippy_unipi.svg?style=)](https://pypi.org/project/sippy_unipi/)
[![PyPI Package latest release](https://img.shields.io/pypi/v/sippy_unipi.svg?style=)](https://pypi.org/project/sippy_unipi/)
[![PyPI Package download count (per month)](https://img.shields.io/pypi/dm/sippy_unipi?style=)](https://pypi.org/project/sippy_unipi/)
[![Quality and Tests](https://github.com/CPCLAB-UNIPI/SIPPY/actions/workflows/ci.yml/badge.svg)](https://github.com/CPCLAB-UNIPI/SIPPY/actions/workflows/ci.yml)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-green?style=&logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![codecov](https://codecov.io/gh/CPCLAB-UNIPI/SIPPY/branch/master/graph/badge.svg?token=BIS0A7CF1F)](https://codecov.io/gh/CPCLAB-UNIPI/SIPPY)

## Systems Identification Package for PYthon (SIPPY)

SIPPY is a library for linear model identification of dynamic systems. It aims to be the most user-friendly and comprehensive library for system identification in Python. 

Originally developed by Giuseppe Armenise at the Department of Civil and Industrial Engineering of University of Pisa under supervision of [Prof. Gabriele Pannocchia](https://people.unipi.it/gabriele_pannocchia/).

## ⚡️ Quickstart

To identify system as Auto-Regressive with eXogenous Inputs model (ARX) using Linear Least Squares  (LLS) on example data, simply run:

```python
from sippy_unipi import system_identification
from sippy_unipi.datasets import load_sample_siso

Y, U = load_sample_siso()

Id_ARX = system_identification(
    Y,
    U,
    "ARX",
    *([4], [[3]], [2], [[11]]),
    id_mode="LLS",
)
```

Get your hand on the algorithms using following Jupyter notebooks and play around with open-spource example data:

* [ARX systems (multi input-multi output case)](https://github.com/CPCLAB-UNIPI/SIPPY/blob/master/Examples/Ex_ARX_MIMO.py)
* [ARMAX systems (single input-single output case)](https://github.com/CPCLAB-UNIPI/SIPPY/blob/master/Examples/Ex_ARMAX.py)
* [ARMAX systems (multi input-multi output case)](https://github.com/CPCLAB-UNIPI/SIPPY/blob/master/Examples/Ex_ARMAX_MIMO.py)
* [Input-output structures (using optimization methods)](https://github.com/CPCLAB-UNIPI/SIPPY/blob/master/Examples/Ex_ARMAX_MIMO.py)
* [Input-output structures (using recursive methods)](https://github.com/CPCLAB-UNIPI/SIPPY/blob/master/Examples/Ex_OPT_GEN-INOUT.py)
* [State space system (multi input-multi output case)](https://github.com/CPCLAB-UNIPI/SIPPY/blob/master/Examples/Ex_SS.py)
* [Continuous Stirred Tank Reactor](https://github.com/CPCLAB-UNIPI/SIPPY/blob/master/Examples/Ex_CST.py)

## 🛠 Installation

Intended to work with Python 3.10 and above. Building project dependencies requires C compiler (euther CMake or Ninja). Pre-build wheels are currently not available (feel free to contribute).

Simply run:

```bash
pip install sippy_unipi
```

To install from source, use poetry:

```bash
poetry install
```

Alternatively, you can use Docker to set up the environment. Follow these steps:

1. Build the Docker image:

```bash
docker build -t sippy .
```

2. Run the Docker container:

```bash
docker run -it --rm sippy
```

## 🔮 Features

SIPPY provides implementations of the following:

**Input-Output Models**

- FIR
- ARX
- ARMAX
- ARMA
- ARARX
- ARARMAX
- OE
- BJ
- GEN

**State-Space Models**

- N4SID
- MOESP
- CVA
- PARSIM_P
- PARSIM_S
- PARSIM_K

## 👐 Contributing

Feel free to contribute in any way you like, we're always open to new ideas and
approaches.

* Feel welcome to
[open an issue](https://github.com/CPCLAB-UNIPI/SIPPY/issues/new/choose)
if you think you've spotted a bug or a performance issue.

## 💬 Citation

If the service or the algorithm has been useful to you and you would like to cite it in an scientific publication, please refer to the
[paper]():

<!-- ```bibtex
@article{sippy,
}
``` -->


## 📝 License

This algorithm is free and open-source software licensed under the [LGPL](https://github.com/CPCLAB-UNIPI/SIPPY/blob/master/LICENSE). license, meaning the code can be used royalty-free even in commercial applications.