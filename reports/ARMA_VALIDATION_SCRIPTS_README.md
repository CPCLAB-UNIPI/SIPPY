# ARMA Validation Scripts

**Created:** 2025-10-13  
**Purpose:** Cross-branch validation for ARMA NLP implementation  
**Status:** Ready to run after NLP implementation complete

---

## Overview

Two comprehensive validation scripts have been created following the ARARX validation pattern to test the ARMA NLP implementation against the master branch reference.

---

## Scripts Created

### 1. `validate_arma_nlp.py` - Comprehensive Validation

**Purpose:** Full validation suite with 4 test cases covering AR, MA, and ARMA models.

**Test Cases:**
- **Case 1: AR(1)** - Pure autoregressive (a1=0.7)
- **Case 2: MA(1)** - Pure moving average (c1=0.5)  
- **Case 3: ARMA(1,1)** - Simple full model (a1=0.7, c1=0.5)
- **Case 4: ARMA(2,2)** - Higher order (a=[0.6, 0.2], c=[0.4, 0.1])

**For Each Test Case:**
- Generate synthetic data with known parameters
- Identify with harold branch (NLP method)
- Identify with master branch (ARMAX with u=0 as proxy)
- Compare coefficients (AR and MA)
- Compare one-step predictions (Yid) - **PRIMARY METRIC**
- Compare H(z) transfer functions
- Compute metrics: MSE, MAE, NRMSE, Correlation

**Validation Metrics:**
- Coefficient errors: < 10% for AR, < 15% for MA (MA harder)
- Yid NRMSE: < 10% (primary acceptance criterion)
- Yid Correlation: > 0.95 (primary acceptance criterion)
- Stability: All poles within unit circle

**Success Criteria:**
- ✅ **EXCELLENT**: NRMSE < 5%, Correlation > 0.99
- ✅ **GOOD**: NRMSE < 10%, Correlation > 0.95
- ⚠️  **ACCEPTABLE**: NRMSE < 15%, Correlation > 0.90
- ❌ **FAIL**: NRMSE >= 15% or Correlation < 0.90

**Output:**
- Console: Detailed test results with pass/fail verdicts
- JSON: `arma_validation_results.json` with machine-readable metrics

**Usage:**
```bash
python validate_arma_nlp.py
```

**Expected Runtime:** 2-5 minutes (4 test cases with NLP optimization)

---

### 2. `debug_arma_nlp.py` - Quick Diagnostic

**Purpose:** Fast diagnostic script for quick verification during development.

**Test Case:** Single AR(1) process (y[k] = 0.7*y[k-1] + e[k])

**Checks:**
- AR coefficient accuracy (< 10% error)
- MA coefficient near zero (< 0.1 absolute)
- Transfer function H(z) comparison (< 1% error)
- One-step prediction Yid comparison (NRMSE < 10%, Corr > 0.95)
- Stability check (poles within unit circle)

**Output:**
- Console: Quick pass/fail verdict with diagnostic details
- No file output (fast iteration)

**Usage:**
```bash
python debug_arma_nlp.py
```

**Expected Runtime:** 10-30 seconds

---

## Validation Methodology

### Why One-Step Predictions (Yid)?

The ARMA NLP minimizes the **one-step prediction error**:

```
minimize (1/N) * sum((y[k] - Yid[k])^2)
```

where `Yid[k]` is the one-step-ahead prediction:

```
Yid[k] = -sum(a*y[k-i]) + sum(c*e[k-j])
```

Therefore, **Yid is the most direct and reliable validation metric**. If Yid matches between harold and master branches, the NLP implementation is correct.

### Why Not Step/Impulse Response?

Step and impulse responses are **not reliable for ARMA** because:
1. Different state-space realizations produce different transient behavior
2. Small coefficient differences cause exponentially diverging responses
3. ARMA can produce unstable systems for certain datasets (correct behavior)

### Master Branch Proxy

Since master branch has no pure ARMA function, we use:
```python
# Master ARMAX with zero inputs as proxy for ARMA
model_m = master_sysid(
    y, u_zeros, "ARMAX",
    na_ord=[na],
    nb_ord=[1],  # Minimal input order
    nc_ord=[nc],
    tsample=1.0
)
```

This is mathematically valid as ARMAX with u=0 degenerates to ARMA.

---

## Expected Results

### Coefficient Accuracy

**AR Coefficients:**
- ✅ **Excellent**: < 5% error
- ✅ **Good**: < 10% error
- ⚠️  **Acceptable**: < 15% error
- ❌ **Poor**: >= 15% error

**MA Coefficients:**
- ✅ **Excellent**: < 10% error
- ✅ **Good**: < 15% error
- ⚠️  **Acceptable**: < 20% error
- ❌ **Poor**: >= 20% error

MA terms are harder to estimate (depend on unobserved noise), so higher tolerance is expected.

### One-Step Prediction Accuracy (PRIMARY)

**Yid NRMSE:**
- ✅ **Excellent**: < 5%
- ✅ **Good**: < 10%
- ⚠️  **Acceptable**: < 15%
- ❌ **Poor**: >= 15%

**Yid Correlation:**
- ✅ **Excellent**: > 0.99
- ✅ **Good**: > 0.95
- ⚠️  **Acceptable**: > 0.90
- ❌ **Poor**: < 0.90

### Transfer Function Comparison

**H(z) = C(z)/A(z) Error:**
- ✅ **Excellent**: < 1e-4 (exact match)
- ✅ **Good**: < 1e-2 (1% error)
- ⚠️  **Acceptable**: < 0.1 (10% error)
- ❌ **Poor**: >= 0.1

---

## Troubleshooting Guide

### If AR Coefficients Are Wrong

**Diagnosis:** Sign convention or regression matrix error

**Check:**
- Line 298 in `arma.py`: `AR_coeffs[i, :] = -theta[:na]` (negation required)
- One-step prediction (line 320): Should use `-AR_coeffs[i, lag]`
- Regression matrix AR filling (lines 204-206)

### If MA Coefficients Are All Zero

**Diagnosis:** MA estimation not working

**Check:**
- Noise estimate update loop (lines 265-286 in `arma.py`)
- Regression matrix MA filling (lines 209-220)
- Coefficient extraction (line 299)

### If Algorithm Doesn't Converge

**Diagnosis:** Numerical instability or poor initialization

**Check:**
- Is lstsq failing? (try-except block, line 224)
- Is variance increasing? (binary search, lines 242-255)
- Is rcond too strict? (currently 1e-10, line 225)

### If Results Vary Across Runs

**Diagnosis:** Random initialization or numerical precision

**Check:**
- Set `np.random.seed()` for reproducibility
- Check if noise estimates are initialized consistently
- Verify no stochastic components in algorithm

### If Master Comparison Fails

**If harold vs. master error > 0.1:**
1. Check data preprocessing: Master branch may rescale data
2. Verify convergence: Both branches may have converged to different local minima
3. Test with simpler case: Start with AR(1) for debugging
4. Compare intermediate steps: Check regression matrices, residuals at each iteration

**If harold vs. master error 0.01 - 0.1:**
- This is **acceptable** - different optimization paths
- Both solutions are likely mathematically valid
- Verify both models have similar prediction accuracy

**If harold vs. master error < 0.01:**
- **Excellent** - algorithms converged to same solution
- High confidence in implementation correctness

---

## Production Readiness Criteria

To declare ARMA NLP production-ready, must achieve:

### Tier 1: Coefficient Accuracy (Synthetic Data) - MUST PASS
- ✅ AR coefficient relative error < 10% for all test cases
- ✅ MA coefficient relative error < 15% for all test cases
- ✅ High SNR case: AR < 5%, MA < 10%

### Tier 2: Prediction Accuracy - MUST PASS
- ✅ Normalized prediction RMSE < 10%
- ✅ Prediction fit percentage > 80%
- ✅ Yid NRMSE < 10%, Correlation > 0.95

### Tier 3: Convergence and Stability - MUST PASS
- ✅ Algorithm converges within 200 iterations
- ✅ No numerical errors (NaN, Inf)
- ✅ Estimated poles inside unit circle (stability)

### Tier 4: Cross-Branch Comparison - SHOULD PASS
- ⚠️  Transfer function coefficient error < 1e-2
- ⚠️  If error > 1e-2, investigate but may accept up to 0.1

**Overall Assessment:**
- **Production Ready**: All Tier 1, Tier 2, Tier 3 tests pass
- **Acceptable**: Tier 1 and Tier 2 pass, Tier 3 may have minor issues
- **Needs Work**: Any Tier 1 or Tier 2 failures

---

## Files Generated

After running `validate_arma_nlp.py`, the following file is created:

```
arma_validation_results.json
```

**Structure:**
```json
{
  "test_case_1_ar1": {
    "passed": true,
    "metrics": {
      "yid_mse": 2.15e-06,
      "yid_mae": 0.00146,
      "yid_nrmse": 0.009,
      "yid_corr": 0.999998,
      "harold_pred_mse": 0.0215,
      "harold_pred_mae": 0.116,
      "master_pred_mse": 0.0215,
      "master_pred_mae": 0.116
    }
  },
  ...
  "overall_status": "PRODUCTION_READY",
  "tests_passed": 4,
  "total_tests": 4
}
```

This can be used for:
- CI/CD integration
- Automated regression testing
- Performance tracking over time

---

## Integration with CI/CD

```bash
# Run validation and check exit code
python validate_arma_nlp.py
if [ $? -eq 0 ]; then
    echo "✅ ARMA validation passed"
else
    echo "❌ ARMA validation failed"
    exit 1
fi
```

Exit codes:
- `0`: All tests passed (production ready)
- `1`: One or more tests failed

---

## References

### Implementation
- `/Users/josephj/Workspace/SIPPY/src/sippy/identification/algorithms/arma.py` - ARMA algorithm

### Strategy Documents
- `ARMA_VALIDATION_STRATEGY.md` - Comprehensive validation approach
- `ARARX_NLP_VALIDATION_REPORT.md` - Reference validation example

### Validation Scripts
- `validate_arma_nlp.py` - Comprehensive validation (this file)
- `debug_arma_nlp.py` - Quick diagnostic

### Reference Implementations
- `validate_ararx_yid.py` - ARARX validation (pattern template)
- `debug_ararx_nlp.py` - ARARX diagnostic (pattern template)

---

**Document Author:** Claude Code (Anthropic)  
**Document Location:** `/Users/josephj/Workspace/SIPPY/ARMA_VALIDATION_SCRIPTS_README.md`  
**Last Updated:** 2025-10-13
