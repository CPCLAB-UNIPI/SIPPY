# SIPPY Codebase Notes for Agents

## Repository Layout
- `src/sippy/`: modern modular architecture with object-oriented identification algorithms.
- `src/sippy/utils/`: signal processing and simulation utilities (FIR analysis, noise generation).
- `src/sippy/identification/algorithms/`: concrete subspace algorithm implementations (N4SID, MOESP, CVA).
- `detrend/`: signal preprocessing filters exposed via a factory pattern.
- `Examples/`: end-to-end scripts and notebooks consuming the library.
- `data/`: sample datasets used by the examples and tests.

## Core Identification Flow

1. Preprocess raw signals with `detrend.DetrendingFilter` to remove trends, interpolate bad slices, and store results in the shared `FilterData` singleton.
2. Convert filtered data into numpy arrays and use `src.sippy.identification.SystemIdentification` with fluent API for modern object-oriented approach.
3. Choose from registered algorithms (N4SID, MOESP, CVA) via `AlgorithmFactory` with configuration through `SystemIdentificationConfig`.
4. Work with the returned `StateSpaceModel` wrapper which exposes matrices, covariance estimates, and helper methods including:
   - `model.get_fir_coefficients()` for FIR coefficient extraction
   - `model.get_step_response()` for step response analysis
   - `model.get_model_uncertainty()` for confidence interval analysis
   - `model.simulate()` for system simulation

## Notable Modules

### Core Architecture
- `src/sippy/identification/__init__.py`: Public API exports and main interface.
- `src/sippy/identification/base.py`: Abstract base classes (`IdentificationAlgorithm`, `StateSpaceModel`) with integrated analysis methods.
- `src/sippy/identification/factory.py`: Algorithm factory pattern for extensible algorithm registration.
- `src/sippy/identification/algorithms/`: Concrete implementations (N4SID, MOESP, CVA) with native algorithm implementations.
- `src/sippy/identification/algorithms/subspace_core.py`: Core subspace identification algorithms (SVD-based implementations).
- `src/sippy/identification/tests/`: Comprehensive pytest test suite with integration tests.

### Utilities
- `src/sippy/utils/signal_utils.py`: Signal generation utilities (`GBN_seq`, `white_noise_var`) and analysis functions.
- `src/sippy/utils/simulation_utils.py`: Simulation and analysis utilities (`get_fir_coef`, `get_step_response`, `get_model_uncertainty`, `simulate_ss_system`).

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

## API Migration Guide

### From Legacy to New Architecture
- `from sysidbox.subspace import system_identification` → `from sippy.identification import system_identification`
- `from sysidbox.functionsetSIM import get_fir_coef` → `model.get_fir_coefficients()` method or `from sippy.utils import get_fir_coef`
- `from sysidbox.functionset import GBN_seq` → `from sippy.utils import GBN_seq`
- `sysidbox.OLSims_methods.OLSims()` → Built into new algorithm implementations via `SubspaceCoreAlgorithm.olsims()`

The new architecture is self-contained with no external sysidbox dependencies, providing cleaner integration and enhanced maintainability.
