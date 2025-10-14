# Workspace Cleanup Plan

## File Categories

### 📁 Keep (Essential Files)
These should remain in the repository:

**Core Documentation**:
- `CLAUDE.md` - Development guidelines
- `README.md` - Main project documentation
- `LICENSE` - License file
- `USER_GUIDE.md` - User documentation
- `pyproject.toml` - Project configuration

**Core Scripts**:
- `benchmark_comprehensive.py` - Main benchmark suite
- `profile_algorithms.py` - Algorithm profiling
- `profile_flamegraph.py` - Flamegraph generation

**Key Reports**:
- `PHASE4_PROFILING_VALIDATION_REPORT.md` - Final optimization report
- `VALIDATION_SUMMARY.md` - Validation overview
- `MIGRATION_ACCURACY_TODO.md` - Migration tracking

### 🗂️ Archive (Move to subdirectories)
These should be organized but kept for reference:

**Debug Scripts** → `debug/`:
- `debug_*.py` (10 files)
- `analyze_*.py` (1 file)
- `check_*.py` (1 file)

**Test Scripts** → `tests/`:
- `test_*.py` (16 files) 
- `validate_*.py` (10 files)
- `verify_*.py` (1 file)

**Benchmark Scripts** → `benchmarks/`:
- `benchmark_*.py` (4 files, excluding main one)

**Investigation Reports** → `reports/`:
- `*REPORT.md` (30+ files)
- `*SUMMARY.md` (15+ files)
- `*INVESTIGATION*.md` (8+ files)

**Data Files** → `data/`:
- `*.json` (1 file)
- `*.png` (1 file)
- `*.txt` (1 file)
- `*.lprof` (1 file)

### 🗑️ Delete (Safe to Remove)
These are likely not needed anymore:

**Redundant/Duplicate Reports**:
- Very similar investigation reports
- Outdated summaries
- Temporary analysis files

**One-off Scripts**:
- `fix_recursive.py`
- `compare_arma_master.py` (if similar to validate scripts)

## Recommended Actions

### 1. Create Directory Structure
```bash
mkdir -p debug tests benchmarks reports data
```

### 2. Move Files to Appropriate Directories
```bash
# Debug scripts
mv debug_*.py debug/
mv analyze_fir_bottleneck.py debug/
mv check_arma_theory.py debug/

# Test scripts  
mv test_*.py tests/
mv validate_*.py tests/
mv verify_*.py tests/

# Benchmark scripts
mv benchmark_arma*.py benchmarks/
mv benchmark_armax*.py benchmarks/
mv benchmark_y_tilde.py benchmarks/
mv benchmark_comprehensive_numba.py benchmarks/

# Reports
mv *REPORT.md reports/
mv *SUMMARY.md reports/
mv *INVESTIGATION*.md reports/

# Data files
mv *.json data/
mv *.png data/
mv *.txt data/
mv *.lprof data/
```

### 3. Selective Cleanup
Review these categories individually before deletion:
- Agent-specific files (`AGENT*.md`, `agents.md`)
- Task-specific files (`TASK*.md`)
- Very similar reports (keep the most comprehensive)

## Benefits
- ✅ Cleaner workspace root
- ✅ Better file organization  
- ✅ Easier navigation
- ✅ Preserved important documentation
- ✅ Reduced clutter in git status
