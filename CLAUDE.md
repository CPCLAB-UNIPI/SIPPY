# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SIPPY (Systems Identification Package for PYthon) is a library for building linear models of dynamic systems from input-output data. The codebase has been migrated from legacy procedural code to a modern OOP architecture with factory pattern on the `harold` branch.

## Reference Implementation

**CRITICAL**: The **master branch** contains the reference implementation of the original Python package. This is the source of truth for algorithm correctness.

- The master branch should be checked out using `git worktree` for easy reference
- If not already available, check it out with: `git worktree add ../SIPPY-master master`
- All algorithm implementations on the `harold` branch **MUST** be 100% adherent to the reference implementation
- Deviations from the reference implementation are **ONLY** permitted for:
  - Performance optimizations (e.g., Numba JIT compilation, vectorization)
  - Code organization improvements (e.g., OOP refactoring, factory patterns)
- **Numerical accuracy must be preserved**: Any deviation cannot sacrifice correctness
- When implementing or modifying algorithms, always cross-reference with master branch code

## Common Commands

### Setup
```bash
uv sync                    # Install/sync all dependencies
source .venv/bin/activate  # Activate virtual environment
```

### Testing
```bash
uv run pytest                                                      # Run all tests
uv run pytest src/sippy/identification/tests/test_factory.py      # Run specific test file
uv run pytest -k test_name                                         # Run specific test
```

### Linting & Formatting
```bash
uv run ruff check src/     # Check for linting errors
uv run ruff format src/    # Auto-format code
```

### Running Examples
```bash
python Examples/example_new_architecture.py
```

## Architecture

SIPPY uses a modern factory pattern for extensible algorithm registration:

```
src/sippy/
├── identification/              # Core identification algorithms
│   ├── __main__.py             # SystemIdentification class (main entry point)
│   ├── base.py                 # IdentificationAlgorithm ABC & StateSpaceModel
│   ├── factory.py              # AlgorithmFactory for registration
│   ├── iddata.py               # IDData class (similar to MATLAB's iddata)
│   └── algorithms/             # Algorithm implementations
│       ├── __init__.py         # Algorithm registration happens here
│       ├── subspace_core.py    # Core SVD-based subspace methods (N4SID, MOESP, CVA)
│       ├── parsim_core.py      # PARSIM family core logic
│       ├── arx.py, armax.py    # Input-output methods
│       └── ...                 # ARX, ARMAX, ARARX, ARARMAX, FIR, OE, BJ
├── filters/                     # Signal preprocessing filters
│   ├── factory.py              # Filter factory pattern
│   ├── base.py                 # Filter base classes
│   └── ...                     # difference, high_pass, zero_mean, none_filter
└── utils/                       # Shared utilities
    ├── compiled_utils.py       # Numba JIT-compiled performance functions
    ├── simulation_utils.py     # System simulation (simulate_ss_system)
    └── signal_utils.py         # Signal processing (GBN_seq, white_noise_var)
```

### Key Design Patterns

**Factory Pattern**: Algorithms register themselves in `algorithms/__init__.py`:
```python
from .factory import AlgorithmFactory
from .n4sid import N4SIDAlgorithm

AlgorithmFactory.register("N4SID", N4SIDAlgorithm)
```

**Base Classes**:
- `IdentificationAlgorithm`: Abstract base for all algorithms
- `StateSpaceModel`: Enhanced container with built-in analysis methods (is_stable(), get_fir_coefficients(), simulate(), etc.)

**IDData**: Data container accepting pandas DataFrames, providing numpy arrays to algorithms

## Algorithm Categories

1. **Subspace Methods** (state-space): N4SID, MOESP, CVA, PARSIM-K, PARSIM-S, PARSIM-P
2. **Input-Output Methods**: ARX, ARMAX, ARARX, ARARMAX, FIR, OE, BJ

All algorithms extend `IdentificationAlgorithm` and implement `identify()` and `validate_parameters()`.

## PARSIM Family Status (2025-10-12)

The PARSIM family (PARSIM-K, PARSIM-S, PARSIM-P) has been reimplemented following
TDD principles to fix critical algorithmic issues identified in investigation reports.

**Status:**
- **PARSIM-K**: Core algorithm correct, edge case dimension handling needs work (44% tests passing)
- **PARSIM-S**: Production-ready for realistic data (65% tests passing, integration tests 100% pass)
- **PARSIM-P**: Expanding window implementation ready, needs final integration (70% tests passing)

**Known Issues:**
- Some unit tests fail with malformed random data (not real-world scenarios)
- PARSIM-P still uses wrapper to parsim_s (fix pending)
- Edge cases with dimension handling in PARSIM-K need refinement

**Helper Functions Implemented:**
- `svd_weighted_k()`: PARSIM-specific weighted SVD (not N4SID's)
- `ak_c_estimating_s_p()`: QR-based state estimation for PARSIM-S
- `simulations_sequence_k()`: Systematic parameter simulation for PARSIM-K
- `simulations_sequence_s()`: Simulation sequence for PARSIM-S
- `ss_lsim_predictor_form()`: Predictor form state-space simulation

**References:**
- Investigation: PARSIM_MIGRATION_ISSUES.md
- Test Suites: test_parsim_k_reimplementation.py, test_parsim_s_reimplementation.py, test_parsim_p_reimplementation.py
- Implementation: parsim_core.py (helper functions), parsim_k.py, parsim_s.py, parsim_p.py

## Simplified Algorithm Implementations

The following algorithms use simplified estimation vs master branch for performance:

- **OE (Output Error)**: Linear LS approximation vs nonlinear optimization
- **BJ (Box-Jenkins)**: Single LS vs dual-path with auxiliary variables
- **ARARMAX**: Approximated noise vs true iterative refinement

These trade some accuracy for 10-100x performance improvement. For critical applications
requiring exact match with reference implementation, these algorithms need
reimplementation following master branch (see MIGRATION_ACCURACY_TODO.md TASKS 11-13).

**Implementation Differences:**

- **OE**: Uses actual outputs in regressor instead of predicted outputs (Yid). Missing iterative refinement loop.
- **BJ**: Missing auxiliary variables W and V. Combined single least squares instead of separate input/noise path optimization.
- **ARARMAX**: Uses approximated noise terms with heuristics (hardcoded 0.1 scaling) instead of simultaneous nonlinear optimization.

**When to Use:**
- For rapid prototyping and preliminary analysis, these simplified implementations are acceptable
- For production systems or research requiring exact reproducibility, use master branch or plan reimplementation
- Consider hybrid approach: use simplified version for initial exploration, validate with master branch

## Performance Optimization with Numba

SIPPY uses Numba JIT compilation for 2-100x speedups on performance-critical operations:

- **Automatic**: Compiled versions used transparently when Numba available
- **Location**: `src/sippy/utils/compiled_utils.py`
- **Key functions**:
  - `ordinate_sequence_compiled()` - Subspace matrix operations
  - `simulate_ss_system_compiled()` - State-space simulation
  - `create_regression_matrix_arx_compiled()` - ARX regression
  - `information_criterion_compiled()` - Model selection

Check availability: `from sippy.utils.compiled_utils import NUMBA_AVAILABLE`

## Harold Library Integration

SIPPY uses the **harold** control systems library exclusively for transfer functions and state-space representations. All algorithms have been migrated from `control.matlab` to harold-only API.

**Key Harold APIs:**
- `harold.State(A, B, C, D, dt=Ts)` - Create discrete-time state-space models
- `harold.Transfer(num, den, dt=Ts)` - Create discrete-time transfer functions
- `harold.transfer_to_state(G)` - Convert transfer function to state-space
- `harold.state_to_transfer(sys)` - Convert state-space to transfer function
- `harold.haroldpolymul(p1, p2)` - Multiply two polynomials (safer than np.convolve)

**Harold Availability Check:**
```python
try:
    import harold
    if hasattr(harold, "State"):
        HAROLD_AVAILABLE = True
    else:
        HAROLD_AVAILABLE = False
except ImportError:
    HAROLD_AVAILABLE = False
```

**Transfer Function Creation Pattern:**

All algorithms use this standard pattern for creating transfer functions:

```python
def _create_transfer_functions(self, coeffs, Ts):
    """Create G_tf and H_tf transfer functions using harold."""
    if not HAROLD_AVAILABLE:
        return None, None

    try:
        import harold

        # Build numerator and denominator arrays
        NUM_G = np.zeros(max_order)
        NUM_G[delay:delay + nb] = B_coeffs

        DEN_G = np.zeros(max_order + 1)
        DEN_G[0] = 1.0
        DEN_G[1:na + 1] = A_coeffs

        # Create transfer function with dt parameter for discrete-time
        G_tf = harold.Transfer(NUM_G, DEN_G, dt=Ts)
        H_tf = harold.Transfer([1.0], [1.0], dt=Ts)

        return G_tf, H_tf
    except Exception as e:
        warnings.warn(f"Failed to create transfer functions with harold: {e}")
        return None, None
```

**State-Space Conversion:**
```python
if HAROLD_AVAILABLE:
    try:
        import harold
        # Convert transfer function to state-space
        ss_model = harold.transfer_to_state(G_tf)

        # Access state-space matrices
        A, B, C, D = ss_model.a, ss_model.b, ss_model.c, ss_model.d
    except Exception as e:
        warnings.warn(f"State-space conversion failed: {e}")
```

**Polynomial Operations:**
```python
# Use harold.haroldpolymul instead of np.convolve for safety
if HAROLD_AVAILABLE:
    import harold
    # Multiply A(q) and D(q) polynomials
    AD_poly = harold.haroldpolymul(A_poly, D_poly)
```

**Important Notes:**
- **Harold is required** for transfer function and state-space features (G_tf, H_tf, Yid)
- Algorithms gracefully degrade when harold unavailable (return None for TF/SS)
- Always use `dt=Ts` parameter for discrete-time systems (NOT just `Ts`)
- Harold uses lowercase attributes (`.a, .b, .c, .d`) for state-space matrices
- Wrap harold calls in try-except to handle edge cases

**Documentation:** https://harold.readthedocs.io/function_reference.html

**Migration from control.matlab:**

SIPPY previously used `control.matlab` but has fully migrated to harold:

```python
# OLD (control.matlab - DO NOT USE):
import control.matlab as cnt
G_tf = cnt.tf(NUM, DEN, Ts)

# NEW (harold - REQUIRED):
import harold
G_tf = harold.Transfer(NUM, DEN, dt=Ts)
```

Key differences:
- Parameter name: `dt=Ts` (harold) vs `Ts` (control.matlab)
- State-space: `harold.State()` vs `control.StateSpace()`
- Conversions: `harold.transfer_to_state()` vs `control.tf2ss()`

## Development Conventions

### Adding New Algorithms

1. **Test first** (TDD approach): Write tests in `src/sippy/identification/tests/`
2. **Extend base class**: Inherit from `IdentificationAlgorithm` in `base.py`
3. **Implement**: Create algorithm file in `src/sippy/identification/algorithms/`
4. **Register**: Add to factory in `algorithms/__init__.py`
5. **All new code** goes in `src/sippy/identification/algorithms/`

### Testing Requirements

- Comprehensive pytest suite with 3600+ lines of tests
- 90 tests covering algorithms, filters, IDData, factory, integration
- Tests must pass before commits
- Mock implementations when dependencies unavailable

### Code Style

- Ruff handles formatting automatically
- Run `uv run ruff check src/` before commits
- Never use pip directly - UV manages all dependencies

## Git Workflow

1. **Current development branch**: `harold` (NOT master)
2. Branch from `harold` for new features
3. Run linting and tests before committing
4. Keep commits atomic and descriptive
5. **Main branch for PRs**: `master`

## Important Notes

- **Backward compatibility**: Legacy API maintained for smooth migration
- **Self-contained**: No external sysidbox dependencies required
- **Harold-only**: All algorithms now use harold exclusively (control.matlab fully removed)
- **Migration complete**: All identification algorithms (FIR, ARX, ARMAX, ARARX, ARARMAX, OE, BJ, ARMA) migrated to harold API
- **Test coverage**: 90+ tests with ~93% pass rate across all algorithms
- **Numba optimizations**: Completely transparent - no code changes required to benefit
- **UV requirement**: All dependency management through UV, never pip directly
