# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SIPPY (Systems Identification Package for PYthon) is a library for building linear models of dynamic systems from input-output data. The codebase has been migrated from legacy procedural code to a modern OOP architecture with factory pattern on the `harold` branch.

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
- **Harold integration**: Optional dependency for State objects (control systems)
- **Migration tracking**: See `MIGRATION_PROGRESS.md` for status (92% test pass rate)
- **Numba optimizations**: Completely transparent - no code changes required to benefit
- **UV requirement**: All dependency management through UV, never pip directly
