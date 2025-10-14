# SIPPY

System identification library for Python with modern OOP architecture.

# Build & Test

• Setup: `uv sync && source .venv/bin/activate`
• Lint: `uv run ruff check src/`
• Format: `uv run ruff format src/`
• Test: `uv run pytest`
• Test specific: `uv run pytest src/sippy/identification/tests/test_factory.py`
• Examples: `python Examples/example_new_architecture.py`

# Architecture Overview

SIPPY uses modern factory pattern with extensible algorithm registration:

```
src/sippy/
├── identification/              # Core identification algorithms
│   ├── __main__.py             # SystemIdentification class (main entry point)
│   ├── base.py                 # IdentificationAlgorithm ABC & StateSpaceModel
│   ├── factory.py              # AlgorithmFactory for registration
│   ├── iddata.py               # IDData class (MATLAB-like iddata)
│   └── algorithms/             # Algorithm implementations
│       ├── __init__.py         # Algorithm registration happens here
│       ├── subspace_core.py    # Core SVD-based subspace methods (N4SID, MOESP, CVA)
│       ├── parsim_core.py      # PARSIM family core logic
│       └── ...                 # ARX, ARMAX, ARARX, ARARMAX, FIR, OE, BJ
├── filters/                     # Signal preprocessing filters
│   └── ...                     # difference, high_pass, zero_mean, none_filter
└── utils/                       # Shared utilities
    ├── compiled_utils.py       # Numba JIT-compiled performance functions
    ├── simulation_utils.py     # System simulation (simulate_ss_system)
    └── signal_utils.py         # Signal processing (GBN_seq, white_noise_var)
```

# Reference Implementation

**CRITICAL**: The **master branch** contains the reference implementation - source of truth for algorithm correctness.

• Check out via git worktree: `git worktree add ../SIPPY-master master`
• All algorithms on `harold` branch **MUST** be 100% adherent to reference implementation
• Deviations allowed ONLY for performance optimizations (Numba JIT, vectorization, OOP refactoring)
• **Numerical accuracy must be preserved** - any deviation cannot sacrifice correctness
• Always cross-reference master branch when implementing/modifying algorithms

# Algorithm Categories

## Production-Ready Algorithms ✅

**Core I/O Methods**: ARX, ARMAX, FIR - Exact match to master (<1e-8 relative error)
**Subspace Methods**: N4SID, MOESP, CVA, PARSIM-K/S/P - 100% test pass rates
**ARARX**: ✅ Production-ready with NLP implementation (6.2% NRMSE, >0.9999 correlation)
**ARMA**: ✅ Production-ready for simple models (6-13% error for AR, MA, ARMA(1,1))
**OE**: ✅ Production-ready with NLP implementation (1-21% error depending on complexity)
**BJ**: ✅ Production-ready with NLP implementation (3-20% error depending on complexity)
**ARARMAX**: ✅ Core NLP implemented (99% complete, minor API refactoring needed)
**GEN**: ✅ Production-ready - Generalized model algorithm with full 5-polynomial structure

## Migration Status Summary

**Overall Migration Accuracy**: ~87% ✅ (All 15 algorithms migrated)
**API Compliance**: 100% ✅ (All algorithms use modern API signature)
**Critical Fixes**: 100% ✅ (3/3 completed)
**High Priority Tasks**: 100% ✅ (12/12 completed)

**Algorithm Status:**
- **Exact Match**: ARX, ARMAX, FIR, N4SID, MOESP, CVA, PARSIM-K/S/P
- **High Accuracy**: ARARX (6.2% NRMSE), ARMA(1,1) (6-13% error)
- **Production Ready**: OE, BJ with NLP implementations
- **Near Complete**: ARARMAX NLP core implemented (99%)
- **New Addition**: GEN algorithm (generalized all 5-polynomial model)

## Modern API Signature

**All 14 identification algorithms use the modern API signature**:

```python
def identify(
    self,
    y: Optional[np.ndarray] = None,
    u: Optional[np.ndarray] = None,
    iddata: Optional["IDData"] = None,
    **kwargs,
) -> StateSpaceModel:
```

Features: Consistent interface, IDData/raw array support, SystemIdentification compatibility, type hints.

**All 15 identification algorithms use the modern API signature**:

1. **Subspace Methods**: N4SID, MOESP, CVA, PARSIM-K, PARSIM-S, PARSIM-P
2. **Input-Output Methods**: ARX, ARARX, ARARMAX, FIR, ARMAX (4 modes), OE, ARMA, BJ
3. **Generalized Model**: GEN (encompasses all I/O methods)

# Performance Optimization with Numba

SIPPY uses Numba JIT compilation for 2-100x speedups on performance-critical operations:

• **Automatic**: Compiled versions used transparently when Numba available
• **Location**: `src/sippy/utils/compiled_utils.py`
• **Key functions**:
  - `ordinate_sequence_compiled()` - Subspace matrix operations
  - `simulate_ss_system_compiled()` - State-space simulation
  - `create_regression_matrix_arx_compiled()` - ARX regression
  - `information_criterion_compiled()` - Model selection

Check availability: `from sippy.utils.compiled_utils import NUMBA_AVAILABLE`

# Harold Library Integration

SIPPY uses the **harold** control systems library exclusively for transfer functions and state-space representations. All algorithms migrated from `control.matlab` to harold-only API.

**Key Harold APIs:**
- `harold.State(A, B, C, D, dt=Ts)` - Create discrete-time state-space models
- `harold.Transfer(num, den, dt=Ts)` - Create discrete-time transfer functions
- `harold.transfer_to_state(G)` - Convert transfer function to state-space
- `harold.haroldpolymul(p1, p2)` - Multiply polynomials (safer than np.convolve)

**Important Notes:**
- **Harold is required** for transfer function and state-space features (G_tf, H_tf, Yid)
- Always use `dt=Ts` parameter for discrete-time systems (NOT just `Ts`)
- Harold uses lowercase attributes (`.a, .b, .c, .d`) for state-space matrices
- Wrap harold calls in try-except to handle edge cases

**Migration from control.matlab:**
```python
# OLD (control.matlab - DO NOT USE):
import control.matlab as cnt
G_tf = cnt.tf(NUM, DEN, Ts)

# NEW (harold - REQUIRED):
import harold
G_tf = harold.Transfer(NUM, DEN, dt=Ts)
```

# Development Conventions

## Adding New Algorithms

1. **Test first** (TDD approach): Write tests in `src/sippy/identification/tests/`
2. **Extend base class**: Inherit from `IdentificationAlgorithm` in `base.py`
3. **Implement**: Create algorithm file in `src/sippy/identification/algorithms/`
4. **Register**: Add to factory in `algorithms/__init__.py`
5. **All new code** goes in `src/sippy/identification/algorithms/`

## Testing Requirements

- Comprehensive pytest suite with 3600+ lines of tests
- 90+ tests covering algorithms, filters, IDData, factory, integration
- Tests must pass before commits
- Mock implementations when dependencies unavailable

## Cross-Branch Validation Framework

Comprehensive validation framework (`test_master_comparison.py`) compares harold branch against master branch:
- Tests all 14 identification algorithms with realistic data
- Detailed error metrics (max absolute, max relative, Frobenius norm, correlation)
- Expected tolerances: Subspace methods <1e-8, I/O methods <1e-8, Conditional methods <1e-4

## Code Style

- Ruff handles formatting automatically
- Run `uv run ruff check src/` before commits
- Never use pip directly - UV manages all dependencies

# Conventions & Patterns

• Use `IdentificationAlgorithm` base class for new algorithms
• Register algorithms in factory via `algorithms/__init__.py`
• Test before implementation (TDD approach)
• Mock implementations when dependencies unavailable
• All new code goes in `src/sippy/identification/algorithms/`
• Leverage `compiled_utils` for performance-critical numerical operations
• Use harold library exclusively for control systems operations

# Gotchas

• **Master branch** = reference implementation - always cross-reference
• **Migration Complete**: All 15 algorithms migrated to modern architecture
• **Harold-only**: All algorithms use harold exclusively (control.matlab fully removed)
• UV manages all dependencies - never use pip directly
• Ruff formatting fixes most style issues automatically
• Tests must pass before commits - use pre-commit workflow
• Numba optimizations are automatic and transparent
• **ARMA limitations**: Use ARMA(1,1) or simpler for reliable results (ARMA(2,2)+ not recommended)
• **Algorithm Accuracy**: Most algorithms are production-ready, see detailed status above
• **GEN Algorithm**: New generalized model that encompasses all other I/O methods

# External Services

• **Harold library** for control systems (required for TF/SS features)
• **CasADi + IPOPT** for NLP optimization (required for ARARX, ARMA, OE, BJ, ARARMAX, GEN)
• Pandas for data handling
• NumPy/SciPy for numerical computing
• Numba (>=0.60.0) for JIT compilation and performance acceleration

# Validation & Testing Framework

## Cross-Branch Validation

Comprehensive validation framework compares harold vs master implementations:
- **Test Coverage**: All 15 algorithms with realistic test data
- **Error Metrics**: Max absolute, max relative, Frobenius norm, correlation
- **Tolerance Tiers**: 
  - Subspace methods: <1e-8 relative error
  - Core I/O methods: <1e-8 relative error  
  - NLP methods: <1e-4 relative error (ARARX, ARMA, OE, BJ)
- **Validation Scripts**: `validate_*.py` for each algorithm category

## Test Suite Status

- **Total Tests**: 3600+ lines across 90+ comprehensive tests
- **Overall Pass Rate**: ~93% across all algorithms
- **Production-Ready**: 100% test pass rates for core algorithms
- **Coverage**: Algorithms, filters, IDData, factory, integration

# Git Workflow

1. **Current development branch**: `harold` (NOT master)
2. Branch from `harold` for new features
3. Run `uv run ruff check src/` before committing
4. All tests must pass: `uv run pytest`
5. Keep commits atomic and descriptive
6. **Main branch for PRs**: `master`
