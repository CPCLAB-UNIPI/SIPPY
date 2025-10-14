# ARMA Validation Strategy

**Date:** 2025-10-13
**Purpose:** Comprehensive validation approach for ARMA algorithm implementation
**Status:** Draft Strategy Document

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [ARMA Background](#arma-background)
3. [Validation Challenges](#validation-challenges)
4. [Validation Metrics](#validation-metrics)
5. [Test Case Design](#test-case-design)
6. [Cross-Branch Validation](#cross-branch-validation)
7. [Validation Script Usage](#validation-script-usage)
8. [Acceptance Criteria](#acceptance-criteria)
9. [Interpretation Guide](#interpretation-guide)

---

## Executive Summary

This document outlines a comprehensive validation strategy for the ARMA (AutoRegressive Moving Average) algorithm implementation on the harold branch. ARMA is a time series model (no exogenous inputs) that requires special validation approaches.

### Key Challenges

1. **No Master Branch ARMA**: Master branch only has ARMAX (with inputs), not pure ARMA
2. **Time Series Only**: ARMA has no inputs, limiting validation approaches
3. **Iterative Estimation**: Extended least-squares convergence means non-deterministic results
4. **MA Term Complexity**: Moving Average terms are notoriously difficult to estimate accurately

### Validation Approach

**Primary Validation: Synthetic Data with Known Parameters**
- Generate ARMA data with known coefficients
- Estimate using harold ARMA implementation
- Compare estimated vs. true coefficients
- Assess one-step-ahead prediction accuracy

**Secondary Validation: ARMAX with u=0 as Reference**
- Use master branch ARMAX with zero inputs as proxy for ARMA
- Compare transfer functions H(q) = C(q)/A(q)
- Document acceptable tolerance due to algorithmic differences

**Tertiary Validation: Information Criteria**
- Use AIC/BIC for model order selection
- Validate model complexity vs. goodness of fit
- Ensure convergence properties

---

## ARMA Background

### Model Structure

ARMA model equation:
```
A(q) y(k) = C(q) e(k)
```

Where:
- `A(q) = 1 + a1*q^-1 + a2*q^-2 + ... + ana*q^-na` (AR polynomial)
- `C(q) = 1 + c1*q^-1 + c2*q^-2 + ... + cnc*q^-nc` (MA polynomial)
- `e(k)` is white noise

Time-domain form:
```
y[k] = -a1*y[k-1] - a2*y[k-2] - ... - ana*y[k-na]
       + e[k] + c1*e[k-1] + c2*e[k-2] + ... + cnc*e[k-nc]
```

### Implementation Method

Harold branch uses **Iterative Extended Least-Squares (ILLS)**:
1. Initialize noise estimates to zero
2. Build regression matrix with AR terms and lagged noise estimates
3. Solve least-squares for AR and MA coefficients
4. Update noise estimates using current coefficients
5. Repeat until convergence (max 100 iterations, tolerance 1e-6)

**Reference**: `/Users/josephj/Workspace/SIPPY/src/sippy/identification/algorithms/arma.py` lines 173-293

### Known Characteristics

- **AR coefficients**: Sign convention requires negation for transfer function form
- **MA coefficients**: Estimated iteratively using residuals from previous iteration
- **Convergence**: Typically 5-20 iterations for simple models
- **Numerical stability**: Added safeguards for higher-order models (rcond=1e-10, clipping)

**Current Test Results**:
- **Accuracy tests**: 3/3 passing (< 10% error)
- **Unit tests**: 13/13 passing (100%)
- **Production status**: Ready for ARMA(1,1) to ARMA(2,2), conditional for higher orders

**Reference**: `/Users/josephj/Workspace/SIPPY/ARMA_ACCURACY_IMPROVEMENT_REPORT.md`

---

## Validation Challenges

### Challenge 1: No Pure ARMA in Master Branch

**Problem**: Master branch (`sippy_unipi`) only has ARMAX (with inputs), not pure ARMA.

**Investigation**:
- Searched master branch: `./sippy_unipi/armaxMIMO.py` contains ARMAX only
- No standalone ARMA function found
- ARMAX requires both `y` and `u` inputs

**Workaround Options**:
1. **Use ARMAX with u=0** (zeros): Master ARMAX with zero inputs approximates ARMA
2. **Compare with theoretical expectations**: Validate against known ARMA processes
3. **Use literature benchmarks**: Compare with published ARMA results

**Recommended Approach**: Combination of synthetic data validation (primary) and ARMAX comparison (secondary)

### Challenge 2: MA Terms Are Hard to Estimate

**Problem**: Moving Average terms depend on unobserved past noise, requiring iterative estimation.

**Implications**:
- MA coefficients have higher uncertainty than AR coefficients
- Different optimization paths can converge to different solutions
- Acceptable tolerance must be higher for MA terms

**Validation Strategy**:
- **AR tolerance**: < 5% relative error expected
- **MA tolerance**: < 10% relative error acceptable
- Focus on prediction accuracy (Yid) as secondary validation

### Challenge 3: Non-Unique State-Space Realizations

**Problem**: Same ARMA model can have multiple equivalent state-space representations.

**Impact**:
- Direct A, B, C, D matrix comparison may fail even for correct models
- Transfer function comparison is more reliable
- Simulation-based metrics are most robust

**Validation Strategy**:
- **Primary**: Compare AR/MA coefficients (unique)
- **Secondary**: Compare transfer functions H(q) = C(q)/A(q)
- **Tertiary**: Compare one-step-ahead predictions (Yid)

### Challenge 4: Iterative Convergence

**Problem**: Iterative algorithms may converge to different local optima.

**Implications**:
- Results may vary slightly across runs (but deterministic with same seed)
- Master vs. harold may converge to different but equally valid solutions
- Tolerance must account for multiple convergence paths

**Validation Strategy**:
- Use fixed random seeds for reproducibility
- Accept solutions within tolerance (not exact match)
- Validate physical plausibility (stability, noise variance)

---

## Validation Metrics

### 1. Coefficient Accuracy (Primary Metric)

**AR Coefficients (a1, a2, ..., ana)**:

```python
ar_true = [-0.7, -0.2]  # Transfer function form
ar_estimated = model.AR_coeffs[0, :]

# Relative error for each coefficient
ar_error = np.abs(ar_estimated - ar_true) / np.abs(ar_true) * 100

# Acceptance criteria
assert ar_error.max() < 5%, "AR coefficient error exceeds 5%"
```

**MA Coefficients (c1, c2, ..., cnc)**:

```python
ma_true = [0.3, 0.1]
ma_estimated = model.MA_coeffs[0, :]

# Relative error for each coefficient
ma_error = np.abs(ma_estimated - ma_true) / np.abs(ma_true) * 100

# Acceptance criteria (more tolerant than AR)
assert ma_error.max() < 10%, "MA coefficient error exceeds 10%"
```

**Why This Metric**:
- Direct validation of estimated parameters
- Unique and unambiguous (unlike state-space matrices)
- Standard in time series literature

**Limitations**:
- Requires known true coefficients (synthetic data only)
- Doesn't test prediction quality

### 2. One-Step-Ahead Prediction Accuracy

**Prediction RMSE (Root Mean Squared Error)**:

```python
# Model provides Yid (one-step-ahead predictions)
prediction_error = model.Yid - y_true
rmse = np.sqrt(np.mean(prediction_error**2))

# Normalize by signal standard deviation
normalized_rmse = rmse / np.std(y_true) * 100

# Acceptance criteria
assert normalized_rmse < 15%, "Normalized prediction RMSE exceeds 15%"
```

**Prediction Fit Percentage**:

```python
# MATLAB-style fit percentage
fit_percent = 100 * (1 - np.linalg.norm(model.Yid - y_true) /
                         np.linalg.norm(y_true - np.mean(y_true)))

# Acceptance criteria
assert fit_percent > 80%, "Prediction fit < 80%"
```

**Why This Metric**:
- Tests actual model utility (not just parameter accuracy)
- Works with real data (no true coefficients needed)
- Standard validation in system identification

**Limitations**:
- Can be high even with wrong coefficients (overfitting)
- Doesn't test multi-step forecasting

### 3. Residual Analysis

**Residual Whiteness Test**:

```python
residuals = y - model.Yid

# Autocorrelation of residuals (should be white noise)
from statsmodels.stats.diagnostic import acorr_ljungbox

# Ljung-Box test for residual autocorrelation
lb_stat, lb_pvalue = acorr_ljungbox(residuals, lags=10)

# Acceptance criteria
assert lb_pvalue > 0.05, "Residuals are not white (autocorrelated)"
```

**Residual Variance**:

```python
residual_variance = np.var(residuals)

# Should match noise variance used in data generation
assert np.abs(residual_variance - true_noise_variance) / true_noise_variance < 0.2, \
    "Residual variance mismatch > 20%"
```

**Why This Metric**:
- Validates model adequacy (good models produce white residuals)
- Standard in time series analysis
- Detects model misspecification

**Limitations**:
- Requires longer time series (> 100 samples)
- May be affected by initial transients

### 4. Transfer Function Comparison

**H(q) = C(q)/A(q) Comparison**:

```python
# Extract transfer function from model
H_harold = model.H_tf

# Compare numerator (C polynomial)
num_harold = H_harold.num[0]
num_master = H_master.num[0][0]

# Normalize and compare
num_error = np.max(np.abs(num_harold - num_master))

# Compare denominator (A polynomial)
den_harold = H_harold.den[0]
den_master = H_master.den[0][0]

den_error = np.max(np.abs(den_harold - den_master))

# Acceptance criteria (relaxed for iterative algorithms)
assert num_error < 0.01, "Numerator error exceeds 0.01"
assert den_error < 0.01, "Denominator error exceeds 0.01"
```

**Why This Metric**:
- Enables comparison with ARMAX (master branch proxy)
- Invariant to state-space realization
- Standard in control theory

**Limitations**:
- Requires harold library
- Different algorithms may produce equivalent but not identical H(q)

### 5. Information Criteria

**AIC (Akaike Information Criterion)**:

```python
n = y.shape[1]  # Number of samples
k = na + nc  # Number of parameters
residual_variance = np.var(residuals)

AIC = n * np.log(residual_variance) + 2 * k

# Lower AIC is better (penalizes complexity)
```

**BIC (Bayesian Information Criterion)**:

```python
BIC = n * np.log(residual_variance) + k * np.log(n)

# Lower BIC is better (stronger penalty than AIC)
```

**Order Selection**:

```python
# Test multiple orders and select minimum AIC/BIC
orders = [(1,1), (2,1), (1,2), (2,2)]
results = []

for na, nc in orders:
    model = arma.identify(y=y, u=None, na=na, nc=nc)
    residuals = y - model.Yid
    aic = compute_aic(residuals, na, nc, y.shape[1])
    results.append((na, nc, aic))

# Best model has minimum AIC
best_na, best_nc, best_aic = min(results, key=lambda x: x[2])
```

**Why This Metric**:
- Validates model order selection
- Balances fit quality vs. complexity
- Standard in statistical modeling

**Limitations**:
- Doesn't validate parameter accuracy
- Assumes correct model structure

### 6. Spectral Analysis

**Power Spectral Density (PSD) Comparison**:

```python
from scipy import signal

# Compute PSD of true data
f_true, psd_true = signal.periodogram(y, fs=1/Ts)

# Simulate from estimated model and compute PSD
y_sim = simulate_arma(model, n_samples=y.shape[1], seed=42)
f_sim, psd_sim = signal.periodogram(y_sim, fs=1/Ts)

# Compare PSDs
psd_error = np.mean(np.abs(psd_true - psd_sim) / psd_true) * 100

# Acceptance criteria
assert psd_error < 20%, "PSD mismatch exceeds 20%"
```

**Why This Metric**:
- Tests frequency domain characteristics
- Validates overall model dynamics
- Robust to phase shifts

**Limitations**:
- Requires long time series
- Sensitive to noise

---

## Test Case Design

### Test Case 1: Simple AR(1) Process

**Purpose**: Validate pure AR estimation (baseline)

**Model**:
```
y[k] = 0.7 * y[k-1] + e[k]
```

**Transfer Function Form**:
```
A(q) = 1 - 0.7*q^-1
C(q) = 1
```

**True Parameters**:
- `na = 1`, `nc = 1`
- `ar_coeffs = [-0.7]` (TF form)
- `ma_coeffs = [0.0]` (no MA component, but nc=1 required by ARMA)

**Expected Behavior**:
- AR coefficient should be estimated accurately (< 2% error)
- MA coefficient should be near zero (< 5% absolute value)
- Prediction RMSE should be low (< 10% normalized)

**Validation**:
```python
model = arma.identify(y=y, u=None, na=1, nc=1, tsample=1.0)
assert abs(model.AR_coeffs[0, 0] - (-0.7)) / 0.7 < 0.02
assert abs(model.MA_coeffs[0, 0]) < 0.05
```

### Test Case 2: Simple MA(1) Process

**Purpose**: Validate pure MA estimation (challenging)

**Model**:
```
y[k] = e[k] + 0.5 * e[k-1]
```

**Transfer Function Form**:
```
A(q) = 1
C(q) = 1 + 0.5*q^-1
```

**True Parameters**:
- `na = 1`, `nc = 1`
- `ar_coeffs = [0.0]` (no AR component, but na=1 required)
- `ma_coeffs = [0.5]`

**Expected Behavior**:
- MA coefficient should be estimated reasonably (< 10% error)
- AR coefficient should be near zero (< 5% absolute value)
- Convergence may take more iterations (MA is harder)

**Validation**:
```python
model = arma.identify(y=y, u=None, na=1, nc=1, tsample=1.0)
assert abs(model.MA_coeffs[0, 0] - 0.5) / 0.5 < 0.10
assert abs(model.AR_coeffs[0, 0]) < 0.05
```

**Note**: Pure MA processes are notoriously difficult to estimate. 10% error is acceptable.

### Test Case 3: ARMA(2,2) Process

**Purpose**: Validate full ARMA with multiple lags

**Model**:
```
y[k] = 0.6*y[k-1] + 0.2*y[k-2] + e[k] + 0.4*e[k-1] + 0.1*e[k-2]
```

**Transfer Function Form**:
```
A(q) = 1 - 0.6*q^-1 - 0.2*q^-2
C(q) = 1 + 0.4*q^-1 + 0.1*q^-2
```

**True Parameters**:
- `na = 2`, `nc = 2`
- `ar_coeffs = [-0.6, -0.2]`
- `ma_coeffs = [0.4, 0.1]`

**Expected Behavior**:
- AR coefficients accurate (< 5% error each)
- MA coefficients reasonable (< 10% error each)
- Good prediction fit (> 80%)

**Validation**:
```python
model = arma.identify(y=y, u=None, na=2, nc=2, tsample=1.0)
ar_error = np.abs(model.AR_coeffs[0, :] - np.array([-0.6, -0.2])) / np.abs([-0.6, -0.2]) * 100
ma_error = np.abs(model.MA_coeffs[0, :] - np.array([0.4, 0.1])) / np.abs([0.4, 0.1]) * 100
assert ar_error.max() < 5
assert ma_error.max() < 10
```

### Test Case 4: Real-World Time Series (Sunspot Data)

**Purpose**: Validate on real data without known parameters

**Data Source**: Monthly sunspot numbers (classic time series benchmark)

**Approach**:
1. Load sunspot data (1749-present)
2. Fit ARMA models of various orders
3. Compare using AIC/BIC for order selection
4. Validate using out-of-sample prediction

**Expected Behavior**:
- Model should converge (no numerical errors)
- AIC/BIC should select reasonable order (typically ARMA(2,2) or ARMA(3,2))
- Out-of-sample fit should be moderate (40-60% typical for sunspots)

**Validation**:
```python
# Split data
y_train = sunspot[:800]
y_test = sunspot[800:]

# Fit ARMA
model = arma.identify(y=y_train, u=None, na=2, nc=2, tsample=1.0)

# Out-of-sample prediction (simplified - would need proper forecasting)
# This tests model doesn't diverge
y_pred = model.simulate(n_steps=len(y_test))
fit = compute_fit_percent(y_test, y_pred)

assert fit > 30, "Out-of-sample fit too poor"
```

**Note**: Real data validation focuses on robustness and convergence, not parameter accuracy.

### Test Case 5: High SNR (Low Noise)

**Purpose**: Validate accuracy under ideal conditions

**Model**: Same as Test Case 3 (ARMA(2,2))

**Noise Level**: `noise_std = 0.01` (very low)

**Expected Behavior**:
- Near-perfect parameter estimation (< 1% error)
- Very high prediction fit (> 95%)
- Few iterations needed (< 10)

**Validation**:
```python
y, ar_true, ma_true = generate_arma_data(
    ar_coeffs=[-0.6, -0.2],
    ma_coeffs=[0.4, 0.1],
    noise_std=0.01,  # Very low noise
    n_samples=2000
)

model = arma.identify(y=y, u=None, na=2, nc=2, tsample=1.0)

ar_error = np.abs(model.AR_coeffs[0, :] - ar_true) / np.abs(ar_true) * 100
ma_error = np.abs(model.MA_coeffs[0, :] - ma_true) / np.abs(ma_true) * 100

assert ar_error.max() < 1, "High SNR should give < 1% AR error"
assert ma_error.max() < 2, "High SNR should give < 2% MA error"
```

### Test Case 6: Model Order Mismatch

**Purpose**: Validate behavior when model order is incorrectly specified

**Data Generation**: True ARMA(2,1)
**Estimation**: Fit ARMA(1,1) (underspecified) and ARMA(3,3) (overspecified)

**Expected Behavior**:
- **Underspecified**: Higher residual variance, worse fit
- **Overspecified**: Similar fit, higher AIC/BIC (complexity penalty)

**Validation**:
```python
# Generate ARMA(2,1) data
y, _, _ = generate_arma_data(ar_coeffs=[-0.6, -0.2], ma_coeffs=[0.4])

# Fit correct order
model_correct = arma.identify(y=y, u=None, na=2, nc=1)
aic_correct = compute_aic(model_correct)

# Fit underspecified
model_under = arma.identify(y=y, u=None, na=1, nc=1)
aic_under = compute_aic(model_under)

# Fit overspecified
model_over = arma.identify(y=y, u=None, na=3, nc=2)
aic_over = compute_aic(model_over)

# Correct order should have lowest AIC
assert aic_correct < aic_under, "Underspecified model should have worse AIC"
assert aic_correct < aic_over, "Overspecified model should have worse AIC"
```

---

## Cross-Branch Validation

### Approach: Use ARMAX with u=0 as Proxy

Since master branch doesn't have pure ARMA, we use ARMAX with zero inputs as a reference:

**Master Branch Call**:
```python
from sippy_unipi import system_identification as master_sysid

# Create zero input
u_zeros = np.zeros((1, y.shape[1]))

# Call ARMAX with nc > 0 (ARMAX mode)
model_master = master_sysid(
    y,
    u_zeros,
    "ARMAX",
    na_ord=[na],
    nb_ord=[1],  # Minimal input order
    nc_ord=[nc],
    tsample=1.0
)

# Extract H(q) = C(q)/A(q) for comparison
H_master = model_master.H
```

**Harold Branch Call**:
```python
from sippy.identification import SystemIdentification, SystemIdentificationConfig

config = SystemIdentificationConfig(method="ARMA")
config.na = na
config.nc = nc

identifier = SystemIdentification(config)
model_harold = identifier.identify(y=y, u=None)

# Extract H(q)
H_harold = model_harold.H_tf
```

**Comparison**:
```python
# Extract transfer function coefficients
master_num = H_master.num[0][0]  # C polynomial
master_den = H_master.den[0][0]  # A polynomial

harold_num = H_harold.num[0]
harold_den = H_harold.den[0]

# Normalize by leading denominator coefficient
master_num_norm = master_num / master_den[0]
master_den_norm = master_den / master_den[0]

harold_num_norm = harold_num / harold_den[0]
harold_den_norm = harold_den / harold_den[0]

# Compute errors
num_error = np.max(np.abs(harold_num_norm - master_num_norm))
den_error = np.max(np.abs(harold_den_norm - master_den_norm))

print(f"Numerator (C polynomial) error: {num_error:.2e}")
print(f"Denominator (A polynomial) error: {den_error:.2e}")
```

### Expected Tolerance

**Acceptable Error Levels**:
- **Exact match (< 1e-7)**: Unlikely due to different algorithms
- **Excellent (< 1e-4)**: Both algorithms converged to same solution
- **Good (< 1e-2)**: Different convergence paths, mathematically equivalent
- **Acceptable (< 0.1)**: Different optimization, both valid models
- **Poor (> 0.1)**: Investigate for bugs

**Reference**: Based on ARMAX cross-validation (test_master_comparison.py lines 1031-1257)

### Known Differences

1. **Algorithm**: Harold uses extended LS, master uses optimization
2. **Initialization**: Different starting points for iterative estimation
3. **Convergence**: Different stopping criteria
4. **Data preprocessing**: Master may use rescaling (see ARMAX_ERROR_INVESTIGATION_REPORT.md)

**Conclusion**: Expect tolerance of 1e-2 to 1e-4 for cross-branch comparison.

---

## Validation Script Usage

### Running the Validation Script

```bash
# Run comprehensive validation
python validate_arma_template.py

# Run specific test case
python validate_arma_template.py --test case1  # AR(1)
python validate_arma_template.py --test case2  # MA(1)
python validate_arma_template.py --test case3  # ARMA(2,2)
python validate_arma_template.py --test case4  # Real data
python validate_arma_template.py --test case5  # High SNR

# Run with master branch comparison
python validate_arma_template.py --compare-master

# Run with custom parameters
python validate_arma_template.py --na 2 --nc 2 --noise-std 0.1 --n-samples 1000
```

### Output Interpretation

The script produces:

1. **Console Output**: Real-time progress and results
2. **Metrics Summary**: Table of errors and fit statistics
3. **Plots** (optional):
   - Time series: true vs. predicted
   - Residuals: autocorrelation plot
   - PSD: true vs. estimated model
4. **JSON Report**: Machine-readable results for CI/CD

**Example Output**:
```
================================================================================
ARMA VALIDATION REPORT
================================================================================

Test Case: ARMA(2,2) - Synthetic Data

TRUE PARAMETERS:
  AR coefficients (TF form): [-0.60, -0.20]
  MA coefficients (TF form): [ 0.40,  0.10]

ESTIMATED PARAMETERS:
  AR coefficients: [-0.598, -0.206]
  MA coefficients: [ 0.413,  0.097]

COEFFICIENT ERRORS:
  AR[0] error: 0.33% ✓ PASS (< 5% threshold)
  AR[1] error: 3.00% ✓ PASS (< 5% threshold)
  MA[0] error: 3.25% ✓ PASS (< 10% threshold)
  MA[1] error: 3.00% ✓ PASS (< 10% threshold)

PREDICTION ACCURACY:
  RMSE: 0.0987
  Normalized RMSE: 8.23% ✓ PASS (< 15% threshold)
  Fit percentage: 92.3% ✓ PASS (> 80% threshold)

CONVERGENCE:
  Iterations: 12
  Final variance: 0.00974

RESIDUAL ANALYSIS:
  Ljung-Box p-value: 0.342 ✓ PASS (> 0.05, residuals are white)
  Residual variance: 0.00974 (expected: 0.01000, error: 2.6%)

OVERALL: ✅ ALL TESTS PASSED

================================================================================
```

---

## Acceptance Criteria

### Tier 1: Coefficient Accuracy (Synthetic Data)

**Must Pass**:
- AR coefficient relative error < 5% for all test cases
- MA coefficient relative error < 10% for all test cases
- High SNR case: AR < 1%, MA < 2%

**Rationale**: Direct validation of estimation accuracy

### Tier 2: Prediction Accuracy

**Must Pass**:
- Normalized prediction RMSE < 15%
- Prediction fit percentage > 80%
- Residuals pass Ljung-Box test (p > 0.05)

**Rationale**: Validates model utility for forecasting

### Tier 3: Convergence and Stability

**Must Pass**:
- Algorithm converges within 100 iterations
- No numerical errors (NaN, Inf)
- Estimated poles inside unit circle (stability)

**Rationale**: Ensures robustness

### Tier 4: Cross-Branch Comparison (Optional)

**Should Pass**:
- Transfer function coefficient error < 1e-2
- If error > 1e-2, investigate but may accept up to 0.1

**Rationale**: Different algorithms, tolerance expected

### Overall Assessment

**Production Ready**: All Tier 1, Tier 2, Tier 3 tests pass
**Acceptable**: Tier 1 and Tier 2 pass, Tier 3 may have minor issues
**Needs Work**: Any Tier 1 or Tier 2 failures

---

## Interpretation Guide

### When Results Are Good (< 5% Error)

**Conclusion**: Implementation is highly accurate
**Action**: Document and proceed to production
**Confidence**: High - algorithm works as intended

### When Results Are Acceptable (5-10% Error)

**Conclusion**: Implementation is reasonably accurate
**Action**:
1. Check if errors are consistent across test cases
2. Verify convergence (more iterations may help)
3. Consider if tolerance is acceptable for use case

**Confidence**: Medium - algorithm works but not optimal

### When Results Are Marginal (10-20% Error)

**Conclusion**: Implementation may have issues
**Action**:
1. Investigate convergence (is it reaching local minimum?)
2. Check for numerical stability issues
3. Compare with master branch (if available)
4. Consider if data is pathological (too short, too noisy)

**Confidence**: Low - needs investigation

### When Results Are Poor (> 20% Error)

**Conclusion**: Implementation likely has bugs
**Action**:
1. Check coefficient sign conventions (AR negation?)
2. Verify regression matrix construction
3. Check noise estimate update logic
4. Compare line-by-line with master branch ARMAX

**Confidence**: Very Low - likely bug present

### When Master Comparison Fails

**If harold vs. master error > 0.1**:

1. **Check data preprocessing**: Master branch may rescale data (see ARMAX_ERROR_INVESTIGATION_REPORT.md)
2. **Verify convergence**: Both branches may have converged to different local minima
3. **Test with simpler case**: Start with ARMA(1,1) for debugging
4. **Compare intermediate steps**: Check regression matrices, residuals at each iteration

**If harold vs. master error 0.01 - 0.1**:

- This is **acceptable** - different optimization paths
- Both solutions are likely mathematically valid
- Verify both models have similar prediction accuracy

**If harold vs. master error < 0.01**:

- **Excellent** - algorithms converged to same solution
- High confidence in implementation correctness

### Troubleshooting Common Issues

**Issue 1: MA coefficients are all zero**

**Diagnosis**: MA estimation not working
**Check**:
- Noise estimate update loop (lines 265-286 in arma.py)
- Regression matrix MA filling (lines 209-220 in arma.py)
- Coefficient extraction (line 299 in arma.py)

**Issue 2: AR coefficients have wrong sign**

**Diagnosis**: Transfer function convention error
**Check**:
- Line 298 in arma.py: `AR_coeffs[i, :] = -theta[:na]` (negation required)
- One-step prediction (line 320): Should use `-AR_coeffs[i, lag]`

**Issue 3: Algorithm doesn't converge**

**Diagnosis**: Numerical instability or poor initialization
**Check**:
- Is lstsq failing? (try-except block, line 224)
- Is variance increasing? (binary search, lines 242-255)
- Is rcond too strict? (currently 1e-10, line 225)

**Issue 4: Results vary across runs**

**Diagnosis**: Random initialization or numerical precision
**Check**:
- Set `np.random.seed()` for reproducibility
- Check if noise estimates are initialized consistently
- Verify no stochastic components in algorithm

---

## Summary

This validation strategy provides a comprehensive framework for assessing ARMA algorithm correctness:

1. **Primary validation**: Synthetic data with known coefficients (most reliable)
2. **Secondary validation**: Cross-branch comparison with ARMAX proxy (contextual)
3. **Tertiary validation**: Prediction accuracy and residual analysis (practical)

**Recommended Workflow**:
1. Run synthetic data tests (Test Cases 1-6) - must pass Tier 1 and Tier 2
2. If all pass, run master comparison (optional) - document tolerance
3. Test on real data (Test Case 4) - verify robustness
4. Generate validation report - document results

**Expected Outcome**:
- Harold ARMA should pass all synthetic data tests (current: 3/3 accuracy tests passing)
- Master comparison should show tolerance < 1e-2 (different algorithms)
- Real data should converge without errors

**Reference Documents**:
- Implementation: `/Users/josephj/Workspace/SIPPY/src/sippy/identification/algorithms/arma.py`
- Accuracy tests: `/Users/josephj/Workspace/SIPPY/test_arma_accuracy.py`
- Unit tests: `/Users/josephj/Workspace/SIPPY/src/sippy/identification/tests/test_arma_algorithm.py`
- Master comparison: `/Users/josephj/Workspace/SIPPY/src/sippy/identification/tests/test_master_comparison.py`
- Previous reports: `ARMA_FIX_REPORT.md`, `ARMA_ACCURACY_IMPROVEMENT_REPORT.md`

---

**Document Author**: Claude Code (Anthropic)
**Document Location**: `/Users/josephj/Workspace/SIPPY/ARMA_VALIDATION_STRATEGY.md`
**Last Updated**: 2025-10-13
