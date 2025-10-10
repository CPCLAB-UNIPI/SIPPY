# SIPPY Codebase Notes for Agents

## Repository Layout
- `src/sippy/`: new modular architecture with object-oriented identification algorithms.
- `sysidbox/`: legacy core identification algorithms and utilities (maintained for backward compatibility).
- `detrend/`: signal preprocessing filters exposed via a factory pattern.
- `Examples/`: end-to-end scripts and notebooks consuming the library.
- `data/`: sample datasets used by the examples and tests.

## Core Identification Flow

### New Architecture (Preferred)
1. Preprocess raw signals with `detrend.DetrendingFilter` to remove trends, interpolate bad slices, and store results in the shared `FilterData` singleton.
2. Convert filtered data into numpy arrays and use `src.sippy.identification.SystemIdentification` with fluent API for modern object-oriented approach.
3. Choose from registered algorithms (N4SID, MOESP, CVA) via `AlgorithmFactory` with configuration through `SystemIdentificationConfig`.
4. Work with the returned `StateSpaceModel` wrapper which exposes matrices, covariance estimates, and helper methods.
5. Use `sysidbox.functionsetSIM` helpers for downstream analysis and visualization.

### Legacy Architecture (Backward Compatible)
1. Preprocess raw signals with `detrend.DetrendingFilter` to remove trends, interpolate bad slices, and store results in the shared `FilterData` singleton.
2. Convert filtered data into numpy arrays and call `sysidbox.subspace.system_identification` with the desired subspace method (N4SID, MOESP, or CVA) and configuration.
3. Work with the returned `SS_model` wrapper, which exposes the state-space matrices, covariance estimates, and helper gains.
4. Use `sysidbox.functionsetSIM` helpers (`get_fir_coef`, `get_step_response`, `get_model_uncertainty`, etc.) for downstream analysis and visualization.

## Notable Modules

### New Architecture
- `src/sippy/identification/__init__.py`: Public API exports and main interface.
- `src/sippy/identification/base.py`: Abstract base classes (`IdentificationAlgorithm`, `StateSpaceModel`).
- `src/sippy/identification/factory.py`: Algorithm factory pattern for extensible algorithm registration.
- `src/sippy/identification/algorithms/`: Concrete implementations (N4SID, MOESP, CVA).
- `src/sippy/identification/tests/`: Comprehensive pytest test suite with integration tests.

### Legacy Architecture
- `sysidbox/functionset.py`: low-level signal utilities (noise generation, scaling, validation metrics).
- `sysidbox/functionsetSIM.py`: simulation routines, FIR extraction, uncertainty analysis, and dead-time estimation.
- `sysidbox/OLSims_methods.py`: SVD-based subspace identification implementation plus the `SS_model` container.
- `sysidbox/tests/test_armax.py`: Legacy test focusing on ARMAX compatibility.

### Signal Processing
- `detrend/high_pass_filter.py`, `difference_filter.py`, `zero_mean_filter.py`, `none_filter.py`: concrete filter strategies created by `DetrendingFilter`.

### Examples
- `Examples/upstream_id.py`: comprehensive reference pipeline from raw data to identified model and plots.
- `example_new_architecture.py`: demonstration of new object-oriented architecture with fluent API.

## Tooling

### Dependency Management
- `uv.lock` and `pyproject.toml` manage modern Python dependencies with uv CLI (replaces requirements.txt approach).
- `setup.py` packages the modules under the `sippy` distribution name for backward compatibility.
- `uv` CLI is the primary dependency management tool (replaces pip/puthon approaches).

### Testing
- `pytest` is the primary testing framework with comprehensive test coverage:
  - `src/sippy/identification/tests/`: new architecture tests (algorithms, factory, integration)
  - `sysidbox/tests/`: legacy API compatibility tests
  - `.pytest_cache/`: pytest cache directory
- Run tests with: `uv run pytest` or `pytest` (if uv activates environment)

### Development Environment
- `Dockerfile` and `.devcontainer/` provide containerized development setup.
- `uv` manages Python virtual environments automatically.
- Laboratory environment includes: NumPy/SciPy, harold, control, pandas, slycot, JupyterLab.

### Key Files
- `example_new_architecture.py`: working demonstration of new API.
- `REFACTORING_SUMMARY.md`: detailed documentation of the architectural transformation.
- `uv.lock`: lockfile for reproducible dependency versions.
