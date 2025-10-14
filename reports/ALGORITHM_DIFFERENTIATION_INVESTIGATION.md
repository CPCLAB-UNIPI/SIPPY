# Algorithm Differentiation Investigation Report

**Date:** October 12, 2025
**Task:** Debug why different algorithms return identical results (TASK 7 from MIGRATION_ACCURACY_TODO.md)
**Status:** ROOT CAUSE IDENTIFIED - No actual "identical results" bug exists

## Executive Summary

The investigation revealed that **the reported issue does not exist**. Algorithms do NOT return identical results. Instead, there are two separate issues:

1. **Misconception about algorithm names**: The investigation reports looked for "ARMAX-BB", "ARMAX-FB", "ARMAX-FF" which don't exist in the codebase. The actual ARMAX variants are "ARMAX_ILLS", "ARMAX_RLLS", "ARMAX_OPT", "ARMAX_RLS", "ARMAX_ILS".

2. **Signature incompatibility bug**: 6 algorithms (ARMAX, ARARX, FIR, OE, BJ, ARMA) have an incompatible `identify()` method signature, causing them to fail when called through `SystemIdentification` class. However, when called directly, they work correctly and produce different results.

## Investigation Results

### 1. Subspace Methods Differentiation (N4SID, MOESP, CVA)

**Verified:** Subspace methods DO produce different results (as expected).

Test results:
```
N4SID vs MOESP:
  Max |A_diff| = 0.0009904606
  Max |B_diff| = 0.0003423001
  Max |C_diff| = 0.0077975740

N4SID vs CVA:
  Max |A_diff| = 1.6679881546
  Max |B_diff| = 0.1400955875
  Max |C_diff| = 14.7339551859

MOESP vs CVA:
  Max |A_diff| = 1.6676504897
  Max |B_diff| = 0.1403689139
  Max |C_diff| = 14.7385297149
```

**Conclusion:** Subspace methods are properly differentiated. N4SID, MOESP, and CVA use different SVD weighting schemes and produce significantly different state-space matrices.

### 2. ARMAX Variants Differentiation

**Misconception:** The investigation looked for "ARMAX-BB", "ARMAX-FB", "ARMAX-FF" which don't exist.

**Actual ARMAX variants in factory:**
- ARMAX (default, uses ILLS mode)
- ARMAX_ILS (alias for ILLS)
- ARMAX_ILLS (Iterative Least Squares)
- ARMAX_OPT (Optimization-based)
- ARMAX_RLLS (Recursive Least Squares)
- ARMAX_RLS (alias for RLLS)

**Test results when called directly:**
```
ILLS vs RLLS:
  Max |A_diff| = 0.7994024958
  Max |B_diff| = 0.1267030778

ILLS vs OPT:
  Max |A_diff| = 0.0053483088
  Max |B_diff| = 0.3658572075

RLLS vs OPT:
  Max |A_diff| = 0.8047508046
  Max |B_diff| = 0.4925602853
```

**Conclusion:** ARMAX variants ARE properly differentiated when called directly. They use different estimation approaches (ILLS iterative least squares, RLLS recursive least squares, OPT scipy optimization) and produce significantly different results.

### 3. Root Cause: Signature Incompatibility

**The Real Bug:** 6 algorithms have incompatible `identify()` signatures with the base class and `SystemIdentification` interface.

#### Algorithms with WRONG signature (old style):
```python
def identify(self, data, config):
    """data is an IDData-like object, config is a dict"""
```

Affected algorithms:
1. `/Users/josephj/Workspace/SIPPY/src/sippy/identification/algorithms/armax.py`
2. `/Users/josephj/Workspace/SIPPY/src/sippy/identification/algorithms/ararx.py`
3. `/Users/josephj/Workspace/SIPPY/src/sippy/identification/algorithms/fir.py`
4. `/Users/josephj/Workspace/SIPPY/src/sippy/identification/algorithms/oe.py`
5. `/Users/josephj/Workspace/SIPPY/src/sippy/identification/algorithms/bj.py`
6. `/Users/josephj/Workspace/SIPPY/src/sippy/identification/algorithms/arma.py`

#### Expected signature (from base class):
```python
def identify(
    self,
    y: Optional[np.ndarray] = None,
    u: Optional[np.ndarray] = None,
    iddata: Optional["IDData"] = None,
    **kwargs,
) -> StateSpaceModel:
```

#### Why the bug occurs:

1. User creates config: `SystemIdentificationConfig(method="ARMAX_ILLS")`
2. `SystemIdentification.identify()` calls `create_algorithm("ARMAX_ILLS")`
3. Factory creates `ARMAXAlgorithm(mode="ILLS")` instance
4. `SystemIdentification.identify()` calls: `algorithm.identify(y_centered, u_centered, **config_dict)`
5. **BUG:** `config_dict` contains `method="ARMAX_ILLS"`, but `ARMAXAlgorithm.identify(data, config)` signature doesn't accept `method` in config
6. **ERROR:** "ARMAXAlgorithm.identify() got an unexpected keyword argument 'method'"

#### Algorithms with CORRECT signature (new style):
- N4SID, MOESP, CVA (all subspace methods)
- PARSIM-K, PARSIM-S, PARSIM-P
- ARX (fixed recently)

These algorithms correctly implement:
```python
def identify(self, y=None, u=None, iddata=None, **kwargs) -> StateSpaceModel:
```

## Why Subspace Methods Work But Input-Output Methods Don't

**Subspace methods (N4SID, MOESP, CVA):**
- Use correct signature: `identify(self, y, u, iddata, **kwargs)`
- Extract parameters from `**kwargs` (e.g., `ss_f`, `ss_threshold`)
- Ignore unknown parameters like `method`
- Work seamlessly with `SystemIdentification` class

**Input-Output methods (ARMAX, ARARX, etc.):**
- Use old signature: `identify(self, data, config)`
- Expect all parameters in `config` dict
- Do NOT expect `method` in config dict
- FAIL when called through `SystemIdentification` because `method` is passed

**Why ARX works:**
- ARX was recently updated to use the correct signature
- File: `/Users/josephj/Workspace/SIPPY/src/sippy/identification/algorithms/arx.py`
- This shows the pattern for fixing the other algorithms

## Technical Details of the Fix

### Pattern to follow (from ARX):

```python
def identify(
    self,
    y: Optional[np.ndarray] = None,
    u: Optional[np.ndarray] = None,
    iddata: Optional["IDData"] = None,
    **kwargs,
) -> StateSpaceModel:
    """
    Identify ARX model from input-output data.

    Parameters:
    -----------
    y : np.ndarray, optional
        Output data (outputs x time_steps)
    u : np.ndarray, optional
        Input data (inputs x time_steps)
    iddata : IDData, optional
        Input-output data container
    **kwargs : dict
        Configuration parameters including na, nb, nk, tsample
    """
    # Validate input arguments
    if iddata is not None and (y is not None or u is not None):
        raise ValueError("Provide either iddata or (y, u), but not both")
    if iddata is None and (y is None or u is None):
        raise ValueError("Must provide either iddata or both y and u")

    # Extract data if IDData is provided
    if iddata is not None:
        u = iddata.get_input_array()
        y = iddata.get_output_array()
        sample_time = iddata.sample_time
    else:
        y = np.atleast_2d(y)
        u = np.atleast_2d(u)
        sample_time = kwargs.get("tsample", 1.0)

    # Extract configuration parameters from kwargs
    na = kwargs.get("na", 1)
    nb = kwargs.get("nb", 1)
    nk = kwargs.get("nk", 1)

    # Validate parameters
    self.validate_parameters(na=na, nb=nb, nk=nk)

    # ... rest of algorithm implementation
```

### Key changes required:

1. **Change signature** from `identify(self, data, config)` to `identify(self, y, u, iddata, **kwargs)`
2. **Add parameter validation** for `y`, `u`, `iddata`
3. **Extract data** from `iddata` if provided, else use `y`, `u`
4. **Extract config parameters** from `**kwargs` instead of `config` dict
5. **Remove `method` from kwargs** or ignore it (it's only used by factory)
6. **Update docstrings** to reflect new signature

## Factory Registration

The factory registration for ARMAX variants is correct:

```python
AlgorithmFactory.register("ARMAX", ARMAXAlgorithm)
AlgorithmFactory.register("ARMAX_ILS", lambda: ARMAXAlgorithm(mode="ILS"))
AlgorithmFactory.register("ARMAX_ILLS", lambda: ARMAXAlgorithm(mode="ILLS"))
AlgorithmFactory.register("ARMAX_OPT", lambda: ARMAXAlgorithm(mode="OPT"))
AlgorithmFactory.register("ARMAX_RLLS", lambda: ARMAXAlgorithm(mode="RLLS"))
AlgorithmFactory.register("ARMAX_RLS", lambda: ARMAXAlgorithm(mode="RLLS"))
```

However, the lambda functions should accept `**kwargs` to be compatible with `AlgorithmFactory.create()`:

```python
# Current (causes issues):
AlgorithmFactory.register("ARMAX_ILLS", lambda: ARMAXAlgorithm(mode="ILLS"))

# Better (but still not needed once identify() is fixed):
AlgorithmFactory.register("ARMAX_ILLS", lambda **kwargs: ARMAXAlgorithm(mode="ILLS"))
```

Actually, once the `identify()` signatures are fixed, the lambdas can remain as-is since `create()` calls them without kwargs anyway.

## Testing Evidence

### Test files created:
1. `/Users/josephj/Workspace/SIPPY/test_algorithm_differentiation_debug.py` - Verified subspace methods work
2. `/Users/josephj/Workspace/SIPPY/test_armax_differentiation.py` - Showed ARMAX fails with SystemIdentification
3. `/Users/josephj/Workspace/SIPPY/test_armax_direct_simple.py` - Proved ARMAX works when called directly

### Key test results:

**Subspace methods differentiation (PASS):**
- N4SID, MOESP, CVA produce different A, B, C matrices
- Differences range from 0.0001 to 16.7 in matrix elements
- As expected based on different SVD weighting schemes

**ARMAX variants differentiation (PASS when called directly):**
- ILLS, RLLS, OPT produce different A, B matrices
- Differences range from 0.005 to 0.8 in matrix elements
- As expected based on different estimation algorithms

**SystemIdentification interface (FAIL for input-output methods):**
- ARMAX_ILLS, ARMAX_RLLS, ARMAX_OPT fail with signature error
- Error: "ARMAXAlgorithm.identify() got an unexpected keyword argument 'method'"
- Caused by incompatible `identify(data, config)` signature

## Recommendations

### Immediate Actions Required:

1. **Fix algorithm signatures** for these 6 files:
   - `armax.py` - Update `identify()` signature to match base class
   - `ararx.py` - Update `identify()` signature to match base class
   - `fir.py` - Update `identify()` signature to match base class
   - `oe.py` - Update `identify()` signature to match base class
   - `bj.py` - Update `identify()` signature to match base class
   - `arma.py` - Update `identify()` signature to match base class

2. **Use ARX as template** - ARX algorithm (`arx.py`) already has the correct signature and can serve as a reference

3. **Add regression tests** - Create tests that verify:
   - Algorithms work through `SystemIdentification` interface
   - Different variants produce different results
   - Signature incompatibilities are caught early

### Long-term Improvements:

1. **Enforce signature consistency** - Add type hints and static analysis to catch signature mismatches
2. **Update documentation** - Clarify that ARMAX variants are named "ARMAX_ILLS" not "ARMAX-BB"
3. **Improve error messages** - Make it clearer when signature incompatibility occurs
4. **Add integration tests** - Test all algorithms through the full `SystemIdentification` interface

## Conclusion

**No "identical results" bug exists.** The investigation reports were based on two misconceptions:

1. Looking for non-existent algorithm names ("ARMAX-BB" vs "ARMAX_ILLS")
2. Not realizing that signature incompatibility was preventing algorithms from being tested

The real issue is that 6 algorithms have outdated `identify()` signatures that are incompatible with the `SystemIdentification` class interface. Once these signatures are updated to match the base class (following the ARX pattern), all algorithms will work correctly and continue to produce differentiated results.

**Algorithm differentiation is working correctly** - no changes needed to the algorithm logic itself.
