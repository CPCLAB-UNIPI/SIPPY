# How to Fix Algorithm Signature Incompatibility

This document provides step-by-step instructions for fixing the 6 algorithms with signature incompatibility.

## Problem

6 algorithms use the old signature:
```python
def identify(self, data, config):
```

But the base class and `SystemIdentification` interface expect:
```python
def identify(self, y=None, u=None, iddata=None, **kwargs) -> StateSpaceModel:
```

## Affected Files

1. `/Users/josephj/Workspace/SIPPY/src/sippy/identification/algorithms/armax.py`
2. `/Users/josephj/Workspace/SIPPY/src/sippy/identification/algorithms/ararx.py`
3. `/Users/josephj/Workspace/SIPPY/src/sippy/identification/algorithms/fir.py`
4. `/Users/josephj/Workspace/SIPPY/src/sippy/identification/algorithms/oe.py`
5. `/Users/josephj/Workspace/SIPPY/src/sippy/identification/algorithms/bj.py`
6. `/Users/josephj/Workspace/SIPPY/src/sippy/identification/algorithms/arma.py`

## Reference Implementation

**Use ARX as template:** `/Users/josephj/Workspace/SIPPY/src/sippy/identification/algorithms/arx.py` (lines 92-211)

ARX was recently fixed and shows the correct pattern.

## Step-by-Step Fix (Using ARMAX as Example)

### Step 1: Update Method Signature

**Before:**
```python
def identify(self, data, config):
    """
    Identify ARMAX model from input-output data using the selected algorithm mode.

    Parameters:
    -----------
    data : IDData
        Input-output data
    config : SystemIdentificationConfig or dict
        Configuration parameters including na, nb, nc, nk, armx_mode
    """
```

**After:**
```python
def identify(
    self,
    y: Optional[np.ndarray] = None,
    u: Optional[np.ndarray] = None,
    iddata: Optional["IDData"] = None,
    **kwargs,
) -> StateSpaceModel:
    """
    Identify ARMAX model from input-output data using the selected algorithm mode.

    Parameters:
    -----------
    y : np.ndarray, optional
        Output data (outputs x time_steps)
    u : np.ndarray, optional
        Input data (inputs x time_steps)
    iddata : IDData, optional
        Input-output data container
    **kwargs : dict
        Configuration parameters including na, nb, nc, nk, armx_mode, max_iterations, etc.

    Returns:
    --------
    model : StateSpaceModel
        Identified state-space model

    Note:
        Either (y, u) or iddata should be provided, but not both.
    """
```

### Step 2: Add Input Validation (at start of identify method)

```python
# Validate input arguments
if iddata is not None and (y is not None or u is not None):
    raise ValueError("Provide either iddata or (y, u), but not both")
if iddata is None and (y is None or u is None):
    raise ValueError("Must provide either iddata or both y and u")
```

### Step 3: Extract Data from Either Source

**Replace:**
```python
# Extract data from IDData object
u = data.get_input_array()
y = data.get_output_array()
```

**With:**
```python
# Extract data if IDData is provided
if iddata is not None:
    u = iddata.get_input_array()
    y = iddata.get_output_array()
    sample_time = iddata.sample_time
else:
    # Ensure arrays are 2D
    y = np.atleast_2d(y)
    u = np.atleast_2d(u)
    sample_time = kwargs.get("tsample", 1.0)
```

### Step 4: Extract Configuration from kwargs Instead of config

**Replace:**
```python
# Extract configuration parameters (support both object and dict config)
if hasattr(config, "__dict__"):
    # Object config
    na = getattr(config, "na", 1)
    nb = getattr(config, "nb", 1)
    nc = getattr(config, "nc", 1)
    # ... etc
else:
    # Dict config
    na = config.get("na", 1)
    nb = config.get("nb", 1)
    nc = config.get("nc", 1)
    # ... etc
```

**With:**
```python
# Extract configuration parameters from kwargs
na = kwargs.get("na", 1)
nb = kwargs.get("nb", 1)
nc = kwargs.get("nc", 1)
nk = kwargs.get("nk", 1)
max_iterations = kwargs.get("max_iterations", 200)
convergence_tolerance = kwargs.get("convergence_tolerance", 1e-6)

# For ARMAX-specific: check if mode override is provided
armx_mode = kwargs.get("armx_mode", None)
if armx_mode is not None and armx_mode != self.mode:
    self.mode = armx_mode.upper()
    self.handler = get_armax_handler(self.mode)
```

### Step 5: Use sample_time Instead of data.sample_time

**Throughout the method, replace:**
```python
data.sample_time  # OLD
```

**With:**
```python
sample_time  # NEW (extracted in Step 3)
```

### Step 6: Add Type Hints to Imports (at top of file)

```python
from typing import Optional

from ..base import IdentificationAlgorithm, StateSpaceModel
```

## Testing the Fix

After making changes to an algorithm, test it with:

```python
import numpy as np
from sippy.identification import SystemIdentification, SystemIdentificationConfig

# Generate test data
np.random.seed(42)
u = np.random.randn(1, 500)
y = np.zeros((1, 500))
for i in range(1, 500):
    y[0, i] = 0.8*y[0, i-1] + 0.5*u[0, i-1] + 0.1*np.random.randn()

# Test the algorithm
config = SystemIdentificationConfig(method="ARMAX_ILLS")  # or whichever you fixed
config.na = 1
config.nb = 1
config.nc = 1
config.nk = 1
identifier = SystemIdentification(config)
model = identifier.identify(y=y, u=u)

# Verify it worked
assert model is not None
assert model.A.shape[0] > 0
print(f"Success! Model created with A.shape = {model.A.shape}")
```

## Run Full Test Suite

After fixing, run:
```bash
uv run pytest src/sippy/identification/tests/test_algorithm_differentiation.py -v
```

The tests for your algorithm should change from FAILED to PASSED.

## Common Pitfalls

### 1. Forgetting to handle both iddata and (y, u) inputs
**Wrong:**
```python
u = iddata.get_input_array()  # What if iddata is None?
```

**Right:**
```python
if iddata is not None:
    u = iddata.get_input_array()
else:
    u = np.atleast_2d(u)
```

### 2. Not extracting sample_time correctly
**Wrong:**
```python
sample_time = config.get("tsample", 1.0)  # config doesn't exist anymore
```

**Right:**
```python
if iddata is not None:
    sample_time = iddata.sample_time
else:
    sample_time = kwargs.get("tsample", 1.0)
```

### 3. Leaving references to `data` or `config` parameters
After the fix, search for:
- `data.` (should be replaced with extracted variables)
- `config.get(` or `getattr(config,` (should be `kwargs.get(`)

### 4. Not updating docstrings
Make sure docstring reflects new signature with y, u, iddata, **kwargs

## Checklist for Each Algorithm

- [ ] Update method signature to `identify(self, y, u, iddata, **kwargs)`
- [ ] Add input validation (y/u vs iddata)
- [ ] Extract data from either source
- [ ] Extract all config params from kwargs
- [ ] Replace `data.sample_time` with `sample_time` variable
- [ ] Remove all references to old `data` and `config` parameters
- [ ] Update docstring
- [ ] Add Optional type hints
- [ ] Test manually with script above
- [ ] Run pytest test suite
- [ ] Verify algorithm still produces correct results

## Expected Impact

Once fixed, these algorithms will:
- ✅ Work through `SystemIdentification` interface
- ✅ Accept both IDData objects and numpy arrays
- ✅ Be consistent with all other algorithms
- ✅ Pass all differentiation tests
- ✅ Be fully compatible with the factory pattern

## Recommended Order of Fixes

**Priority 1 (Most Used):**
1. ARMAX - Most common input-output method
2. FIR - Widely used for simple models

**Priority 2 (Commonly Used):**
3. ARARX - Adaptive ARX variant
4. OE - Output-error method

**Priority 3 (Specialized):**
5. BJ - Box-Jenkins method
6. ARMA - Pure time-series (no exogenous input)

---

**Questions?** Reference ARX implementation or investigation reports:
- Reference: `/Users/josephj/Workspace/SIPPY/src/sippy/identification/algorithms/arx.py`
- Report: `/Users/josephj/Workspace/SIPPY/ALGORITHM_DIFFERENTIATION_INVESTIGATION.md`
- Summary: `/Users/josephj/Workspace/SIPPY/ALGORITHM_DIFFERENTIATION_SUMMARY.md`
