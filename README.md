# Welcome to SIPPY!
### Systems Identification Package for PYthon (SIPPY) - Modern Architecture

The main objective of this code is to provide different identification methods
to build linear models of dynamic systems, starting from input-output collected
data. The models can be built as transfer functions or state-space models in
discrete-time domain using modern object-oriented architecture.

It is originally developed by Giuseppe Armenise at the Department of Civil and Industrial Engineering of University of Pisa under supervision of [Prof. Gabriele Pannocchia](https://people.unipi.it/gabriele_pannocchia/). The identification code is distributed under the LGPL license, meaning the code can be used royalty-free even in commercial applications.

SIPPY provides a modern, object-oriented interface with:
- **Factory pattern** for extensible algorithm registration
- **Clean API** with fluent configuration
- **Integrated analysis tools** built into model objects
- **Self-contained implementation** with no external dependencies beyond scientific libraries
- **Enhanced maintainability** and type safety

The linear model to be identified can be chosen between:
* **State-space structures**: N4SID, MOESP, CVA, PARSIM-K, PARSIM-S, PARSIM-P
* **Input-output methods**: ARX, ARMAX, ARARX, ARARMAX, FIR, OE, BJ
* Available for both SISO and MIMO cases

**Algorithm Status Notes:**
- N4SID, MOESP, CVA: Fully validated and production-ready
- ARX, FIR, ARMAX: Fully validated and production-ready
- PARSIM family: Reimplemented with TDD, in progress (see PARSIM status in CLAUDE.md for details)
- OE, BJ, ARARMAX: Simplified implementations for performance (see CLAUDE.md for details)

## Quick Start

```python
import numpy as np
from sippy.identification import SystemIdentification, SystemIdentificationConfig

# Generate sample data
y = np.random.randn(1, 1000)  # 1 output, 1000 time steps
u = np.random.randn(2, 1000)  # 2 inputs, 1000 time steps

# Configure and identify
config = SystemIdentificationConfig(method='N4SID', ss_f=20, ss_fixed_order=2)
identifier = SystemIdentification(config)
model = identifier.identify(y, u)

# Model now has built-in analysis capabilities
print(f"System order: {model.n}")
print(f"System stable: {model.is_stable()}")

# Get FIR coefficients and step responses
fir_model = model.get_fir_coefficients(['input1', 'input2'], ['output1'], 1.0, 60)
step_response = model.get_step_response(['input1', 'input2'], ['output1'])

# Simulate system
x, y_sim = model.simulate(u)
```

## Installation

### Modern Python (3.7+)

```bash
# Using uv (recommended)
pip install uv
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .

# Or using pip
pip install -e .
```

### Requirements
- **Python**: 3.7+ (modern architecture removes Python 2.7 support)
- **Core dependencies**: NumPy, SciPy, harold
- **Optional**: matplotlib for plotting, pandas for data handling

## Architecture Overview

SIPPY now features a clean, modular architecture:

```
src/sippy/
├── identification/           # Core identification algorithms
│   ├── algorithms/         # N4SID, MOESP, CVA implementations
│   │   └── subspace_core.py # Core SVD-based algorithms
│   ├── base.py             # StateSpaceModel with analysis methods
│   ├── factory.py          # Algorithm factory pattern
│   └── __main__.py         # Main identification interface
├── utils/                  # Signal processing and simulation utilities
│   ├── signal_utils.py     # GBN_seq, white_noise_var, etc.
│   └── simulation_utils.py # get_fir_coef, get_step_response, etc.
└── __init__.py
```

### Key Features

1. **Object-Oriented Design**: Clear separation of concerns with abstract base classes
2. **Factory Pattern**: Extensible algorithm registration and discovery
3. **Integrated Analysis**: StateSpaceModel objects include built-in analysis methods
4. **Self-Contained**: No external sysidbox dependencies required

## Migration from Legacy SIPPY

The new architecture maintains backward compatibility while providing cleaner interfaces:

### Old API (Legacy)
```python
from sysidbox.subspace import system_identification
from sysidbox.functionsetSIM import get_fir_coef

model = system_identification(y, u, 'N4SID', SS_fixed_order=2)
fir = get_fir_coef(model.G, inputs, outputs, sampling, tss)
```

### New API (Modern)
```python
from sippy.identification import system_identification

model = system_identification(y, u, 'N4SID', ss_fixed_order=2)
fir = model.get_fir_coefficients(inputs, outputs, sampling, tss)
```

## Package Contents

- `Examples/`: Updated example scripts demonstrating the new architecture
- `src/sippy/identification/`: Core identification algorithms and interfaces
- `src/sippy/utils/`: Signal processing and simulation utilities
- `detrend/`: Signal preprocessing filters (unchanged)

The new architecture provides cleaner integration, enhanced maintainability, and improved developer experience while maintaining the powerful identification algorithms from the original SIPPY package.
