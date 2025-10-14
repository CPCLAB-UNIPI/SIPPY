# CRITICAL: PARSIM Family Migration Issues

## Overview
The PARSIM family algorithms (PARSIM-K, PARSIM-S, PARSIM-P) in the harold branch have **critical algorithmic deviations** from the reference implementation in the master branch. These are NOT minor implementation differences - they fundamentally change the algorithm behavior.

---

## Issue #1: Wrong SVD Method (CRITICAL)

### What Master Does:
```python
def SVD_weighted_K(Uf, Zp, Gamma_L):
    W2 = sc.linalg.sqrtm(np.dot(Z_dot_PIort(Zp, Uf), Zp.T)).real
    U_n, S_n, V_n = np.linalg.svd(np.dot(Gamma_L, W2), full_matrices=False)
    return U_n, S_n, V_n
```

Custom weighting matrix W2 that incorporates:
- Orthogonal projection of Zp onto the complement of Uf
- Matrix square root for proper weighting
- Applied to Gamma_L before SVD

### What Harold Does:
```python
U_n, S_n, V_n, W1, O_i = SubspaceCoreAlgorithm.svd_weighted(
    y, u, f, l_, "N4SID"
)
```

Uses N4SID's SVD which computes:
```python
O_i = (Yf - Yf*Uf^T*pinv(Uf^T)) * pinv(Zp - Zp*Uf^T*pinv(Uf^T)) * Zp
U_n, S_n, V_n = np.linalg.svd(O_i)
```

### Why This Matters:
- **Different matrices**: PARSIM uses `Gamma_L * W2`, N4SID uses `O_i`
- **Different construction**: Gamma_L is built iteratively with output feedback, O_i is computed directly
- **Different weighting**: PARSIM's W2 includes sqrt of covariance, N4SID has no extra weighting
- **Result**: Completely different state-space realization

---

## Issue #2: Wrong Matrix Square Root (HIGH SEVERITY)

### Master:
```python
Ob_K = np.dot(U_n, sc.linalg.sqrtm(S_n))
```

Uses `scipy.linalg.sqrtm()` - **MATRIX** square root
- For diagonal S_n, this is fine, but the pattern matters
- Mathematically correct for subspace identification

### Harold:
```python
S_n_diag = np.diag(S_n)
Ob_K = np.dot(U_n, np.sqrt(S_n_diag))
```

Uses `np.sqrt()` - **ELEMENT-WISE** square root
- Takes sqrt of diagonal elements independently
- For diagonal matrices, numerically similar BUT...
- Different API and potential numerical differences in edge cases

### Why This Matters:
- Consistency with reference implementation
- Future-proofing if S_n handling changes
- Potential numerical precision differences

---

## Issue #3: Missing Gamma_L Construction (CRITICAL for PARSIM-K)

### Master's PARSIM-K:
```python
# Initial projection
M = np.dot(Yf[0:l_, :], np.linalg.pinv(impile(Zp, Uf[0:m, :])))
Gamma_L = M[:, 0 : (m + l_) * f]
H_K = M[:, (m + l_) * f : :]
G_K = np.zeros((l_, l_))

# Iterative construction with OUTPUT FEEDBACK
for i in range(1, f):
    # Estimate y_tilde using BOTH H_K and G_K
    y_tilde = np.dot(H_K[0:l_, :], Uf[m*i : m*(i+1), :])
    for j in range(1, i):
        y_tilde = y_tilde + \
                  np.dot(H_K[l_*j : l_*(j+1), :], Uf[m*(i-j) : m*(i-j+1), :]) + \
                  np.dot(G_K[l_*j : l_*(j+1), :], Yf[l_*(i-j) : l_*(i-j+1), :])

    # Project residual
    M = np.dot((Yf[l_*i : l_*(i+1)] - y_tilde), Matrix_pinv)
    H_K = impile(H_K, M[:, (m + l_)*f : (m + l_)*f + m])
    G_K = impile(G_K, M[:, (m + l_)*f + m : :])
    Gamma_L = impile(Gamma_L, M[:, 0 : (m + l_)*f])
```

**Key points:**
1. `y_tilde` includes BOTH input (H_K) and output feedback (G_K)
2. G_K grows with each iteration
3. Gamma_L is built row-block by row-block with corrected projections

### Harold's PARSIM-K:
```python
# Loop exists but doesn't properly construct Gamma_L
for i in range(1, f):
    if NUMBA_AVAILABLE and parsim_y_tilde_estimation_compiled is not None:
        y_tilde = parsim_y_tilde_estimation_compiled(...)
    else:
        y_tilde = estimating_y(H_K, Uf, G_K, Yf, i, m, l_)

    M = np.dot((Yf[l_*i : l_*(i+1)] - y_tilde), Matrix_pinv)
    H_K = impile(H_K, M[:, (m + l_)*f : (m + l_)*f + m])
    G_K = impile(G_K, M[:, (m + l_)*f + m :])
    Gamma_L = impile(Gamma_L, M[:, 0 : (m + l_)*f])
```

**Problem:** The Gamma_L built here is NEVER USED because:
- Line 171-174 calls N4SID's `svd_weighted()` which computes its own O_i
- The carefully constructed Gamma_L with output feedback is discarded
- This defeats the entire purpose of PARSIM-K

---

## Issue #4: Wrong Simulation Method (HIGH SEVERITY)

### Master's Approach:
```python
def simulations_sequence(A_K, C, L, y, u, l_, m, n, D_required):
    """Systematically simulate with unit vectors for ALL parameters."""
    if D_required:
        n_simulations = n*m + l_*m + n*l_ + n  # B_K, D, K, x0
    else:
        n_simulations = n*m + n*l_ + n  # B_K, K, x0

    vect = np.zeros((n_simulations, 1))
    for i in range(n_simulations):
        vect[i, 0] = 1.0
        # Extract B_K, D, K, x0 from vect
        y_sim.append(SS_lsim_predictor_form(A_K, B_K, C, D, K, y, u, x0))
        vect[i, 0] = 0.0
```

**Key:** Uses predictor form: `x[i+1] = A_K*x[i] + B_K*u[i] + K*y[i]`

### Harold's Approach:
```python
# Create parameter vectors directly
if D_required:
    n_simulations = n*m + l_*m + n*l_ + n
    vect = np.zeros((n_simulations, 1))
    for i in range(n_simulations):
        vect[i, 0] = 1.0
        B_K = vect[0:n*m, :].reshape((n, m))
        # ... but doesn't actually simulate!

# Then uses process form
X_states, Y_estimate = simulate_ss_system(A_K, B_K, C, D, u, x0=x0)
Y_corrected = Y_estimate + np.dot(K, y - Y_estimate)
```

**Problems:**
1. Doesn't use predictor form simulation
2. Process form: `x[i+1] = A_K*x[i] + B_K*u[i]` (no K*y term)
3. Tries to correct afterwards but this isn't equivalent
4. Parameter estimation is ad-hoc, not systematic

---

## Issue #5: PARSIM-S K Estimation (CRITICAL)

### Master's Approach:
```python
def AK_C_estimating_S_P(U_n, S_n, V_n, l_, f, m, Zp, Uf, Yf):
    # ... construct Ob_f, A, C ...

    # QR-based K estimation
    Q, R = np.linalg.qr(impile(impile(Zp, Uf), Yf).T)
    Q = Q.T
    R = R.T
    G_f = R[(2*m + l_)*f : :, (2*m + l_)*f : :]
    F = G_f[0:l_, 0:l_]
    K = np.dot(
        np.dot(np.linalg.pinv(Ob_f[0:l_*(f-1), :]), G_f[l_::, 0:l_]),
        np.linalg.inv(F)
    )
    A_K = A - np.dot(K, C)
    return A, C, A_K, K, n
```

**Method:** Rigorous QR decomposition approach from subspace identification theory

### Harold's Approach:
```python
try:
    H_est = np.dot(np.linalg.pinv(Zp), Yf[0:l_, :])
    residuals = Yf - np.dot(H_est, Zp)
    K = np.dot(residuals, np.linalg.pinv(Yf))
    K = K[:, 0:l_] * 0.1  # Scale down ???
except Exception:
    K = np.random.randn(n, l_) * 0.01  # Random fallback ???
```

**Problems:**
1. Heuristic approach not based on subspace theory
2. Arbitrary scaling factor (0.1)
3. Random fallback if it fails
4. Doesn't match reference algorithm AT ALL

---

## Issue #6: PARSIM-P Implementation (CRITICAL)

### Master's PARSIM-P:
```python
# Initial setup
Matrix_pinv = np.linalg.pinv(impile(Zp, Uf[0:m, :]))
M = np.dot(Yf[0:l_, :], Matrix_pinv)
Gamma_L = M[:, 0 : (m + l_)*f]

# EXPANDING WINDOW approach
for i in range(1, f):
    # Recompute pinv with EXPANDING Uf
    Matrix_pinv = np.linalg.pinv(impile(Zp, Uf[0 : m*(i+1), :]))
    M = np.dot(Yf[l_*i : l_*(i+1)], Matrix_pinv)
    Gamma_L = impile(Gamma_L, M[:, 0 : (m + l_)*f])
```

**Key difference:** `Uf[0 : m*(i+1), :]` GROWS from `m` to `m*f` rows

### Harold's PARSIM-P:
```python
def parsim_p(...):
    # PARSIM-P is very similar to PARSIM-S, using the same implementation
    return ParsimCoreAlgorithm.parsim_s(...)
```

**WRONG:** Just calls PARSIM-S! The expanding window approach is COMPLETELY MISSING.

---

## Issue #7: Missing Helper Functions

### Not Implemented in Harold Branch:

1. **SVD_weighted_K**: Custom PARSIM weighting
2. **AK_C_estimating_S_P**: QR-based state estimation
3. **simulations_sequence**: Systematic parameter simulation
4. **simulations_sequence_S**: PARSIM-S simulation
5. **SS_lsim_predictor_form**: Predictor form dynamics
6. **recalc_K**: B matrix recalculation

---

## Numerical Impact Estimates

Based on the algorithmic differences:

| Component | Expected Error Magnitude | Reason |
|-----------|--------------------------|--------|
| State matrix A | 10-50% | Wrong SVD, wrong observability |
| Input matrix B | 20-100% | Wrong simulation method |
| Output matrix C | 10-30% | Wrong observability matrix |
| Kalman gain K | 50-200% | Wrong estimation (S) or missing output feedback (K) |
| Noise variance Vn | 50-300% | Wrong identified model |

**These are NOT small numerical errors - these are algorithmic errors.**

---

## Test Case Example

### Simple System:
```python
A_true = [[0.9, 0.1], [0.0, 0.8]]
B_true = [[1.0], [0.5]]
C_true = [[1.0, 0.5]]
D_true = [[0.0]]
```

### Expected Results:
- Master branch should identify something close to true system
- Harold branch will identify a DIFFERENT system (wrong algorithm)

### NOT Just Numerical Precision:
- This isn't about 1e-6 vs 1e-8 precision
- This is about getting fundamentally different A, B, C, D matrices
- The harold branch implementations are solving a DIFFERENT optimization problem

---

## Recommendations

### Immediate Actions:
1. ❌ **DO NOT USE** harold branch PARSIM algorithms
2. 🔴 **MARK AS BROKEN** in documentation
3. 🚫 **DISABLE** in production code

### Required Fixes (in order):
1. Implement `SVD_weighted_K()` function
2. Implement `AK_C_estimating_S_P()` function
3. Implement `SS_lsim_predictor_form()` function
4. Implement `simulations_sequence()` function
5. Fix PARSIM-K to use correct SVD and simulation
6. Fix PARSIM-S to use correct K estimation
7. Implement PARSIM-P properly (not as wrapper to PARSIM-S)

### Testing Strategy:
1. Port one simple test case from master
2. Compare numerical results (should match within 1e-6)
3. Test on multiple system types (SISO, MIMO, stable, marginally stable)
4. Validate eigenvalues of identified A matrices
5. Validate model fit (Vn should be similar)

---

## Code Locations

### Master Branch (Reference):
- `/Users/josephj/Workspace/SIPPY-master/sippy_unipi/Parsim_methods.py`
  - PARSIM_K: lines 179-272
  - PARSIM_S: lines 410-485
  - PARSIM_P: lines 597-670
  - SVD_weighted_K: lines 76-79
  - AK_C_estimating_S_P: lines 160-176

### Harold Branch (Broken):
- `/Users/josephj/Workspace/SIPPY/src/sippy/identification/algorithms/parsim_core.py`
  - parsim_k: lines 43-278 (WRONG SVD, WRONG simulation)
  - parsim_s: lines 281-452 (WRONG K estimation)
  - parsim_p: lines 455-496 (WRONG - just calls parsim_s)

---

## Conclusion

The PARSIM family migration is **NOT COMPLETE and NOT CORRECT**. The implementations in the harold branch are not faithful to the reference algorithms and will produce incorrect results.

**Severity: CRITICAL**
**Status: BROKEN**
**Action Required: Complete reimplementation**
