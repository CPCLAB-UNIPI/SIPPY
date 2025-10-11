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
Utilities: Signal processing and simulation tools in `src/sippy/utils/`.
Tests: Comprehensive pytest suite in `src/sippy/identification/tests/`.

# Conventions & Patterns

• Use `IdentificationAlgorithm` base class for new algorithms
• Register algorithms in factory via `algorithms/__init__.py`
• Test before implementation (TDD approach)
• Mock implementations when dependencies unavailable
• All new code goes in `src/sippy/identification/algorithms/`

# Gotchas

• UV manages all dependencies - never use pip directly
• Ruff formatting fixes most style issues automatically
• Tests must pass before commits - use pre-commit workflow
• Migration status tracked in `MIGRATION_PROGRESS.md`

# External Services

• Harold library for control systems (optional dependency)
• Pandas for data handling
• NumPy/SciPy for numerical computing

# Git Workflow

1. Branch from `harold` (current development branch)
2. Run `uv run ruff check src/` before committing
3. All tests must pass: `uv run pytest`
4. Keep commits atomic and descriptive
