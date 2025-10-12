# SIPPY

System identification library for Python with modern OOP architecture.

# Build & Test

• Setup: `uv sync`
• Lint: `uv run ruff check src/`
• Format: `uv run ruff format src/`
• Test: `uv run pytest`
• Test specific: `uv run pytest src/sippy/identification/tests/test_factory.py`

# Architecture Overview

Core: `src/sippy/identification/` with factory pattern for algorithm registration.
Algorithms: State-space (N4SID, MOESP, CVA, PARSIM-K/S/P) and input-output (ARX, ARMAX, OE, FIR).
Utilities: Signal processing and simulation tools in `src/sippy/utils/` with Numba-compiled high-performance functions.
Tests: Comprehensive pytest suite in `src/sippy/identification/tests/`.

# Reference Implementation

**Master branch** = reference implementation (source of truth for correctness).

• Check out via git worktree: `git worktree add ../SIPPY-master master`
• All algorithms must be 100% adherent to reference implementation
• Deviations allowed ONLY for performance gains (Numba, vectorization, OOP refactoring)
• Numerical accuracy cannot be sacrificed
• Always cross-reference master branch when implementing/modifying algorithms

# Performance

SIPPY leverages Numba JIT compilation for significant performance improvements:

• **Automatic acceleration**: Critical functions use compiled versions transparently
• **Speedup range**: 2-100x faster depending on operation and data size
• **Key optimized areas**: State-space simulation, signal processing, ARX regression
• **Backward compatible**: Graceful fallback to pure Python when Numba unavailable

## Compiled Functions

High-performance implementations available in `src/sippy/utils/compiled_utils.py`:

• `ordinate_sequence_compiled()`: Subspace identification matrix operations
• `simulate_ss_system_compiled()`: State-space system simulation
• `create_regression_matrix_arx_compiled()`: ARX regression matrix construction
• `information_criterion_compiled()`: Model selection criteria
• `rescale_compiled()`, `white_noise_compiled()`: Signal processing utilities

## Usage

Optimizations are completely transparent:

```python
from sippy.utils.simulation_utils import simulate_ss_system

# Automatically uses JIT-compiled version when available
x, y = simulate_ss_system(A, B, C, D, u)

# Check Numba availability
from sippy.utils.compiled_utils import NUMBA_AVAILABLE
```

# Conventions & Patterns

• Use `IdentificationAlgorithm` base class for new algorithms
• Register algorithms in factory via `algorithms/__init__.py`
• Test before implementation (TDD approach)
• Mock implementations when dependencies unavailable
• All new code goes in `src/sippy/identification/algorithms/`
• Leverage `compiled_utils` for performance-critical numerical operations

# Gotchas

• UV manages all dependencies - never use pip directly
• Ruff formatting fixes most style issues automatically
• Tests must pass before commits - use pre-commit workflow
• Migration status tracked in `MIGRATION_PROGRESS.md`
• Numba optimizations are automatic and transparent - no code changes required
• Performance improvements scale with data size; largest gains with real-world datasets

# External Services

• Harold library for control systems (optional dependency)
• Pandas for data handling
• NumPy/SciPy for numerical computing
• Numba (>=0.60.0) for JIT compilation and performance acceleration

# Git Workflow

1. Branch from `harold` (current development branch)
2. Run `uv run ruff check src/` before committing
3. All tests must pass: `uv run pytest`
4. Keep commits atomic and descriptive
