# PARSIM Family Algorithm Migration Analysis
## Comparison: Master Branch (Reference) vs Harold Branch

**Date:** 2025-10-12
**Branch:** harold
**Algorithms Analyzed:** PARSIM-K, PARSIM-S, PARSIM-P

---

## Executive Summary

This report analyzes the numerical and algorithmic accuracy of the PARSIM family algorithm migration from the master branch (reference implementation) to the harold branch (new OOP architecture).

### Key Findings:

1. **PARSIM-K**: SIGNIFICANT ALGORITHMIC DEVIATIONS DETECTED
2. **PARSIM-S**: MAJOR IMPLEMENTATION DIFFERENCES
3. **PARSIM-P**: INCORRECT IMPLEMENTATION (delegates to PARSIM-S)

---

## 1. PARSIM-K Analysis

### Reference Implementation (Master Branch)
**File:** `/Users/josephj/Workspace/SIPPY-master/sippy_unipi/Parsim_methods.py`
**Lines:** 179-272

#### Algorithm Structure:

1. **Data Preprocessing** (Lines 211-216):
   - Rescales inputs and outputs using `rescale()` function
   - Stores scaling factors: `Ustd`, `Ystd`

2. **Ordinate Sequence Construction** (Lines 217-219):
   ```python
   Yf, Yp = ordinate_sequence(y, f, p)
   Uf, Up = ordinate_sequence(u, f, p)
   Zp = impile(Up, Yp)
   ```

3. **Initial Projection** (Lines 220-226):
   ```python
   M = np.dot(Yf[0:l_, :], np.linalg.pinv(impile(Zp, Uf[0:m, :])))
   Matrix_pinv = np.linalg.pinv(impile(Zp, impile(Uf[0:m, :], Yf[0:l_, :])))
   Gamma_L = M[:, 0 : (m + l_) * f]
   H_K = M[:, (m + l_) * f : :]
   G_K = np.zeros((l_, l_))
   ```
   - **Key Point**: Initial `M` projects `Yf[0:l_, :]` onto `impile(Zp, Uf[0:m, :])`

4. **Iterative Gamma_L Construction** (Lines 227-232):
   ```python
   for i in range(1, f):
       y_tilde = estimating_y(H_K, Uf, G_K, Yf, i, m, l_)
       M = np.dot((Yf[l_ * i : l_ * (i + 1)] - y_tilde), Matrix_pinv)
       H_K = impile(H_K, M[:, (m + l_) * f : (m + l_) * f + m])
       G_K = impile(G_K, M[:, (m + l_) * f + m : :])
       Gamma_L = impile(Gamma_L, (M[:, 0 : (m + l_) * f]))
   ```
   - **Key Point**: Uses `estimating_y()` with BOTH `H_K` and `G_K` matrices (Line 228)
   - `y_tilde` includes feedback from past outputs via `G_K`

5. **Custom Weighted SVD** (Lines 233-234):
   ```python
   U_n, S_n, V_n = SVD_weighted_K(Uf, Zp, Gamma_L)
   U_n, S_n, V_n = reducingOrder(U_n, S_n, V_n, threshold, max_order)
   ```
   - **Critical**: Uses `SVD_weighted_K()` function (Lines 76-79):
   ```python
   def SVD_weighted_K(Uf, Zp, Gamma_L):
       W2 = sc.linalg.sqrtm(np.dot(Z_dot_PIort(Zp, Uf), Zp.T)).real
       U_n, S_n, V_n = np.linalg.svd(np.dot(Gamma_L, W2), full_matrices=False)
       return U_n, S_n, V_n
   ```
   - Custom weighting: `W2 = sqrt((Zp - Zp*Uf^T*pinv(Uf^T)) * Zp^T)`

6. **Observability Matrix** (Lines 235-238):
   ```python
   S_n = np.diag(S_n)
   Ob_K = np.dot(U_n, sc.linalg.sqrtm(S_n))
   A_K = np.dot(np.linalg.pinv(Ob_K[0 : l_ * (f - 1), :]), Ob_K[l_::, :])
   C = Ob_K[0:l_, :]
   ```

7. **Predictor Form Simulation** (Lines 240-254):
   - Uses `simulations_sequence()` function (Lines 82-120)
   - Systematically simulates with unit vectors for ALL parameters:
     - If `D_required=True`: `n*m + l_*m + n*l_ + n` simulations (B_K, D, K, x0)
     - If `D_required=False`: `n*m + n*l_ + n` simulations (B_K, K, x0)
   - Uses `SS_lsim_predictor_form()` for each simulation

8. **Parameter Extraction** (Lines 241-254):
   ```python
   vect = np.dot(np.linalg.pinv(y_sim), y.reshape((L * l_, 1)))
   Y_estimate = np.dot(y_sim, vect)
   Vn = Vn_mat(y.reshape((L * l_, 1)), Y_estimate)
   B_K = vect[0 : n * m, :].reshape((n, m))
   # Extract D, K, x0 based on D_required
   ```

9. **Process Form Recovery** (Line 255):
   ```python
   A = A_K + np.dot(K, C)
   ```

10. **Optional B Recalculation** (Lines 256-263):
    - If `B_recalc=True`, recalculates B using `recalc_K()` function
    - Uses process form simulation instead of predictor form

11. **Rescaling** (Lines 264-271):
    ```python
    for j in range(m):
        B_K[:, j] = B_K[:, j] / Ustd[j]
        D[:, j] = D[:, j] / Ustd[j]
    for j in range(l_):
        K[:, j] = K[:, j] / Ystd[j]
        C[j, :] = C[j, :] * Ystd[j]
        D[j, :] = D[j, :] * Ystd[j]
    B = B_K + np.dot(K, D)
    ```

### Harold Branch Implementation
**File:** `/Users/josephj/Workspace/SIPPY/src/sippy/identification/algorithms/parsim_core.py`
**Lines:** 43-278

#### Critical Deviations:

1. **WRONG SVD METHOD** (Lines 171-174):
   ```python
   # SVD for order estimation
   U_n, S_n, V_n, W1, O_i = SubspaceCoreAlgorithm.svd_weighted(
       y, u, f, l_, "N4SID"
   )
   ```
   - **ERROR**: Uses N4SID's `svd_weighted()` instead of PARSIM-K's `SVD_weighted_K()`
   - **Impact**: Completely different weighting scheme
   - N4SID uses: `SVD(O_i)` where `O_i = (Yf - Yf*Uf^T*pinv(Uf^T)) * pinv(Zp - Zp*Uf^T*pinv(Uf^T)) * Zp`
   - PARSIM-K uses: `SVD(Gamma_L * W2)` where `W2 = sqrt((Zp - Zp*Uf^T*pinv(Uf^T)) * Zp^T)`

2. **INCORRECT OBSERVABILITY MATRIX** (Lines 177-178):
   ```python
   S_n_diag = np.diag(S_n)
   Ob_K = np.dot(U_n, np.sqrt(S_n_diag))
   ```
   - **ERROR**: Uses `np.sqrt()` instead of `scipy.linalg.sqrtm()`
   - `np.sqrt()` does element-wise square root
   - Master uses `sc.linalg.sqrtm()` for matrix square root
   - **Impact**: Different numerical behavior

3. **MISSING GAMMA_L CONSTRUCTION** (Lines 111-168):
   - Harold branch never constructs the `Gamma_L` matrix properly
   - The iterative loop (lines 153-168) exists but doesn't build the correct structure
   - Master's `Gamma_L` is (l*f, cols) where each row block comes from iterative estimation
   - Harold's implementation doesn't use this matrix for SVD

4. **SIMPLIFIED SIMULATION** (Lines 192-240):
   - Uses generic `simulate_ss_system()` instead of predictor form simulation
   - Line 214-216:
   ```python
   X_states, Y_estimate = simulate_ss_system(A_K, B_K, C, D, u, x0=x0)
   Y_corrected = Y_estimate + np.dot(K, y - Y_estimate)
   ```
   - **ERROR**: This is NOT the same as simulating the predictor form
   - Master uses `SS_lsim_predictor_form()` which has different dynamics:
   ```python
   x[:, i + 1] = np.dot(A_K, x[:, i]) + np.dot(B_K, u[:, i]) + np.dot(K, y[:, i])
   y_hat[:, i] = np.dot(C, x[:, i]) + np.dot(D, u[:, i])
   ```

5. **MISSING SYSTEMATIC PARAMETER ESTIMATION** (Lines 192-240):
   - Master uses systematic unit vector simulations
   - Harold uses ad-hoc parameter estimation
   - Harold's approach (lines 206-211) just creates parameter vectors directly without proper simulation

### Numerical Impact Assessment:

| Aspect | Master Branch | Harold Branch | Impact |
|--------|---------------|---------------|--------|
| SVD Weighting | Custom PARSIM-K | N4SID (wrong) | **CRITICAL** |
| Matrix Square Root | scipy.linalg.sqrtm | np.sqrt (element-wise) | **HIGH** |
| Gamma_L Construction | Iterative with G_K | Not properly used | **CRITICAL** |
| Simulation Method | Predictor form | Process form + correction | **HIGH** |
| Parameter Estimation | Systematic | Ad-hoc | **MODERATE** |

**Overall Assessment for PARSIM-K: INCORRECT IMPLEMENTATION**

---

## 2. PARSIM-S Analysis

### Reference Implementation (Master Branch)
**File:** `/Users/josephj/Workspace/SIPPY-master/sippy_unipi/Parsim_methods.py`
**Lines:** 410-485

#### Algorithm Structure:

1. **Data Preprocessing** (Lines 441-446): Same as PARSIM-K

2. **Ordinate Sequence Construction** (Lines 447-450):
   ```python
   Yf, Yp = ordinate_sequence(y, f, p)
   Uf, Up = ordinate_sequence(u, f, p)
   Zp = impile(Up, Yp)
   Matrix_pinv = np.linalg.pinv(impile(Zp, Uf[0:m, :]))
   ```

3. **Initial Projection** (Lines 451-453):
   ```python
   M = np.dot(Yf[0:l_, :], Matrix_pinv)
   Gamma_L = M[:, 0 : (m + l_) * f]
   H = M[:, (m + l_) * f : :]
   ```
   - **Key Difference from PARSIM-K**: Only uses `H` matrix, no `G_K`

4. **Iterative Gamma_L Construction** (Lines 454-458):
   ```python
   for i in range(1, f):
       y_tilde = estimating_y_S(H, Uf, Yf, i, m, l_)
       M = np.dot((Yf[l_ * i : l_ * (i + 1)] - y_tilde), Matrix_pinv)
       Gamma_L = impile(Gamma_L, (M[:, 0 : (m + l_) * f]))
       H = impile(H, M[:, (m + l_) * f : :])
   ```
   - **Key Point**: Uses `estimating_y_S()` with ONLY `H` matrix (Line 67-73)
   - No output feedback via `G_K`

5. **Same Weighted SVD as PARSIM-K** (Lines 459-460):
   ```python
   U_n, S_n, V_n = SVD_weighted_K(Uf, Zp, Gamma_L)
   ```

6. **K and A Estimation** (Lines 461-462):
   ```python
   A, C, A_K, K, n = AK_C_estimating_S_P(U_n, S_n, V_n, l_, f, m, Zp, Uf, Yf)
   ```
   - Uses shared function `AK_C_estimating_S_P()` (Lines 160-176)

7. **Different Simulation** (Lines 464-476):
   - Uses `simulations_sequence_S()` (Lines 123-157)
   - K is FIXED (from step 6), only estimates B_K, D, x0

### Harold Branch Implementation
**File:** `/Users/josephj/Workspace/SIPPY/src/sippy/identification/algorithms/parsim_core.py`
**Lines:** 281-452

#### Critical Deviations:

1. **WRONG SVD METHOD** (Lines 374-377):
   ```python
   U_n, S_n, V_n, W1, O_i = SubspaceCoreAlgorithm.svd_weighted(
       y, u, f, l_, "N4SID"
   )
   ```
   - **Same error as PARSIM-K**: Uses N4SID instead of `SVD_weighted_K()`

2. **INCORRECT K ESTIMATION** (Lines 394-401):
   ```python
   try:
       H_est = np.dot(np.linalg.pinv(Zp), Yf[0:l_, :])
       residuals = Yf - np.dot(H_est, Zp)
       K = np.dot(residuals, np.linalg.pinv(Yf))
       K = K[:, 0:l_] * 0.1  # Scale down
   except Exception:
       K = np.random.randn(n, l_) * 0.01
   ```
   - **ERROR**: Completely different K estimation method
   - Master uses QR decomposition approach in `AK_C_estimating_S_P()`:
   ```python
   Q, R = np.linalg.qr(impile(impile(Zp, Uf), Yf).T)
   Q = Q.T
   R = R.T
   G_f = R[(2 * m + l_) * f : :, (2 * m + l_) * f : :]
   F = G_f[0:l_, 0:l_]
   K = np.dot(
       np.dot(np.linalg.pinv(Ob_f[0 : l_ * (f - 1), :]), G_f[l_::, 0:l_]),
       np.linalg.inv(F),
   )
   ```

3. **SIMPLIFIED PARAMETER ESTIMATION** (Lines 407-437):
   - Uses generic simulation instead of predictor form
   - Doesn't match master's `simulations_sequence_S()` approach

### Numerical Impact Assessment:

| Aspect | Master Branch | Harold Branch | Impact |
|--------|---------------|---------------|--------|
| SVD Weighting | Custom PARSIM | N4SID (wrong) | **CRITICAL** |
| K Estimation | QR-based | Heuristic | **CRITICAL** |
| Simulation Method | Predictor form | Process form | **HIGH** |
| Algorithm Logic | Follows theory | Simplified | **HIGH** |

**Overall Assessment for PARSIM-S: SIGNIFICANT DEVIATIONS**

---

## 3. PARSIM-P Analysis

### Reference Implementation (Master Branch)
**File:** `/Users/josephj/Workspace/SIPPY-master/sippy_unipi/Parsim_methods.py`
**Lines:** 597-670

#### Algorithm Structure:

1. **Key Difference from PARSIM-S** (Lines 640-643):
   ```python
   for i in range(1, f):
       Matrix_pinv = np.linalg.pinv(impile(Zp, Uf[0 : m * (i + 1), :]))
       M = np.dot((Yf[l_ * i : l_ * (i + 1)]), Matrix_pinv)
       Gamma_L = impile(Gamma_L, (M[:, 0 : (m + l_) * f]))
   ```
   - **CRITICAL**: `Matrix_pinv` is recomputed at EACH iteration with expanding `Uf`
   - `Uf[0 : m * (i + 1), :]` grows from `Uf[0:m, :]` to `Uf[0:m*f, :]`
   - No `y_tilde` estimation - direct projection

2. Rest is identical to PARSIM-S (uses same `AK_C_estimating_S_P()` function)

### Harold Branch Implementation
**File:** `/Users/josephj/Workspace/SIPPY/src/sippy/identification/algorithms/parsim_core.py`
**Lines:** 455-496

#### Critical Error:

```python
def parsim_p(y, u, f=20, p=20, threshold=0.1, max_order=np.nan,
             fixed_order=np.nan, D_required=False):
    """PARSIM-P algorithm implementation..."""
    # PARSIM-P is very similar to PARSIM-S, using the same implementation
    # but with slight parameter variations that are implementation-specific
    return ParsimCoreAlgorithm.parsim_s(
        y, u, f, p, threshold, max_order, fixed_order, D_required
    )
```

**WRONG**: PARSIM-P simply calls PARSIM-S!

- Comment claims "very similar" but this is incorrect
- Master branch has DISTINCT algorithm for PARSIM-P
- The expanding `Uf` window in Gamma_L construction is completely missing

### Numerical Impact Assessment:

| Aspect | Master Branch | Harold Branch | Impact |
|--------|---------------|---------------|--------|
| Algorithm | Distinct PARSIM-P | Calls PARSIM-S | **CRITICAL** |
| Gamma_L Construction | Expanding window | Fixed window | **CRITICAL** |
| Matrix_pinv | Recomputed iteratively | Fixed | **CRITICAL** |

**Overall Assessment for PARSIM-P: COMPLETELY WRONG IMPLEMENTATION**

---

## 4. Common Helper Functions Analysis

### SVD_weighted_K (Master Branch)
**File:** `/Users/josephj/Workspace/SIPPY-master/sippy_unipi/Parsim_methods.py`
**Lines:** 76-79

```python
def SVD_weighted_K(Uf, Zp, Gamma_L):
    W2 = sc.linalg.sqrtm(np.dot(Z_dot_PIort(Zp, Uf), Zp.T)).real
    U_n, S_n, V_n = np.linalg.svd(np.dot(Gamma_L, W2), full_matrices=False)
    return U_n, S_n, V_n
```

**NOT PRESENT in Harold Branch** - This is a CRITICAL omission.

### AK_C_estimating_S_P (Master Branch)
**Lines:** 160-176

```python
def AK_C_estimating_S_P(U_n, S_n, V_n, l_, f, m, Zp, Uf, Yf):
    n = S_n.size
    S_n = np.diag(S_n)
    Ob_f = np.dot(U_n, sc.linalg.sqrtm(S_n))
    A = np.dot(np.linalg.pinv(Ob_f[0 : l_ * (f - 1), :]), Ob_f[l_::, :])
    C = Ob_f[0:l_, :]
    Q, R = np.linalg.qr(impile(impile(Zp, Uf), Yf).T)
    Q = Q.T
    R = R.T
    G_f = R[(2 * m + l_) * f : :, (2 * m + l_) * f : :]
    F = G_f[0:l_, 0:l_]
    K = np.dot(
        np.dot(np.linalg.pinv(Ob_f[0 : l_ * (f - 1), :]), G_f[l_::, 0:l_]),
        np.linalg.inv(F),
    )
    A_K = A - np.dot(K, C)
    return A, C, A_K, K, n
```

**NOT PROPERLY IMPLEMENTED in Harold Branch** - K estimation is different.

### estimating_y (PARSIM-K specific, Master Branch)
**Lines:** 50-64

Includes BOTH `H_K` and `G_K` terms - feedback from past outputs.

### estimating_y_S (PARSIM-S/P specific, Master Branch)
**Lines:** 67-73

Only includes `H` term - no output feedback.

---

## 5. Detailed Line-by-Line Mapping

### PARSIM-K

| Master Branch | Harold Branch | Match? | Notes |
|---------------|---------------|--------|-------|
| 211-216 (rescale) | 104-109 | ✓ | Correct |
| 217-219 (ordinate) | 111-129 | ✓ | Correct |
| 220 (M initial) | 129 | ✓ | Correct |
| 227-232 (Gamma_L loop) | 153-168 | ✗ | Loop exists but doesn't build correct Gamma_L |
| 233 (SVD_weighted_K) | 171-174 | ✗ | Uses wrong SVD (N4SID) |
| 235-238 (Ob_K, A_K, C) | 177-189 | ✗ | Wrong matrix sqrt, wrong Ob_K |
| 240-254 (simulation) | 192-240 | ✗ | Wrong simulation method |
| 255 (A recovery) | 243 | ✓ | Correct formula |
| 256-263 (B_recalc) | 246-266 | ~ | Different approach |
| 264-271 (rescale) | 269-276 | ✓ | Correct |

### PARSIM-S

| Master Branch | Harold Branch | Match? | Notes |
|---------------|---------------|--------|-------|
| 441-446 (rescale) | 339-344 | ✓ | Correct |
| 447-450 (ordinate) | 347-352 | ✓ | Correct |
| 454-458 (Gamma_L loop) | 367-371 | ~ | Logic similar but missing details |
| 459 (SVD_weighted_K) | 374-377 | ✗ | Uses wrong SVD (N4SID) |
| 461-462 (AK_C_estimating_S_P) | 394-404 | ✗ | K estimation completely different |
| 464-476 (simulation) | 407-437 | ✗ | Wrong simulation method |

### PARSIM-P

| Master Branch | Harold Branch | Match? | Notes |
|---------------|---------------|--------|-------|
| 640-643 (Gamma_L with expanding Uf) | 494 | ✗ | MISSING - just calls parsim_s |
| All other logic | 494 | ✗ | MISSING - just calls parsim_s |

---

## 6. Testing Recommendations

Due to the severity of deviations, I recommend:

1. **DO NOT USE** the harold branch PARSIM implementations for production
2. Create proper migration by:
   - Implementing `SVD_weighted_K()` correctly
   - Implementing `AK_C_estimating_S_P()` correctly
   - Implementing predictor form simulation
   - Implementing PARSIM-P as distinct from PARSIM-S

3. **Test Cases Needed:**
   - Simple SISO system (n=2, m=1, l=1)
   - MIMO system (n=3, m=2, l=2)
   - Comparison with known good results from master branch
   - Stability analysis of identified models

---

## 7. Critical Code Sections Requiring Reimplementation

### 7.1 SVD_weighted_K Function (MISSING)

**Required implementation:**
```python
def SVD_weighted_K(Uf, Zp, Gamma_L):
    """Custom PARSIM weighted SVD."""
    import scipy as sc
    W2 = sc.linalg.sqrtm(np.dot(Z_dot_PIort(Zp, Uf), Zp.T)).real
    U_n, S_n, V_n = np.linalg.svd(np.dot(Gamma_L, W2), full_matrices=False)
    return U_n, S_n, V_n
```

### 7.2 AK_C_estimating_S_P Function (MISSING)

**Required implementation:**
```python
def AK_C_estimating_S_P(U_n, S_n, V_n, l_, f, m, Zp, Uf, Yf):
    """Estimate A, C, A_K, K for PARSIM-S and PARSIM-P."""
    import scipy as sc
    n = S_n.size
    S_n = np.diag(S_n)
    Ob_f = np.dot(U_n, sc.linalg.sqrtm(S_n))
    A = np.dot(np.linalg.pinv(Ob_f[0 : l_ * (f - 1), :]), Ob_f[l_::, :])
    C = Ob_f[0:l_, :]
    Q, R = np.linalg.qr(impile(impile(Zp, Uf), Yf).T)
    Q = Q.T
    R = R.T
    G_f = R[(2 * m + l_) * f : :, (2 * m + l_) * f : :]
    F = G_f[0:l_, 0:l_]
    K = np.dot(
        np.dot(np.linalg.pinv(Ob_f[0 : l_ * (f - 1), :]), G_f[l_::, 0:l_]),
        np.linalg.inv(F),
    )
    A_K = A - np.dot(K, C)
    return A, C, A_K, K, n
```

### 7.3 Predictor Form Simulation (SIMPLIFIED)

**Required implementation:**
```python
def SS_lsim_predictor_form(A_K, B_K, C, D, K, y, u, x0=None):
    """Simulate predictor form state-space model."""
    m, L = u.shape
    l_, n = C.shape
    y_hat = np.zeros((l_, L))
    x = np.zeros((n, L + 1))

    if x0 is not None:
        x[:, 0] = x0[:, 0]

    for i in range(0, L):
        x[:, i + 1] = (
            np.dot(A_K, x[:, i]) +
            np.dot(B_K, u[:, i]) +
            np.dot(K, y[:, i])
        )
        y_hat[:, i] = np.dot(C, x[:, i]) + np.dot(D, u[:, i])

    return x, y_hat
```

### 7.4 PARSIM-P Distinct Implementation (MISSING)

**Required:**
- Separate implementation that doesn't call parsim_s
- Expanding window in Gamma_L construction
- Different Matrix_pinv calculation per iteration

---

## 8. Conclusions

### Migration Status:

| Algorithm | Status | Severity | Action Required |
|-----------|--------|----------|-----------------|
| PARSIM-K | FAILED | CRITICAL | Complete reimplementation |
| PARSIM-S | FAILED | HIGH | Major corrections needed |
| PARSIM-P | FAILED | CRITICAL | Implement from scratch |

### Root Causes:

1. **Incorrect abstraction**: Tried to reuse N4SID's SVD instead of implementing PARSIM-specific SVD
2. **Missing functions**: Key helper functions not ported
3. **Simplified approach**: Replaced systematic simulation with ad-hoc methods
4. **Wrong delegation**: PARSIM-P incorrectly delegates to PARSIM-S

### Recommendations:

1. **IMMEDIATE**: Mark PARSIM family as "experimental" or disable in harold branch
2. **SHORT-TERM**: Port missing helper functions exactly from master
3. **MEDIUM-TERM**: Implement proper predictor form simulation
4. **LONG-TERM**: Add comprehensive test suite comparing with master branch results

---

## 9. Algorithm-Specific Details

### PARSIM-K Theory:
- Predictor-based subspace identification
- Uses Kalman gain in state update
- Custom weighting for SVD based on innovation properties
- Requires both H_K (input influence) and G_K (output feedback) matrices

### PARSIM-S Theory:
- Stable realization focus
- Simpler than PARSIM-K (no G_K matrix)
- Uses QR-based K estimation for numerical stability

### PARSIM-P Theory:
- Predictor form optimization
- Expanding window approach for better parameter estimation
- Different Gamma_L construction from PARSIM-S

---

## 10. References

**Master Branch Files:**
- `/Users/josephj/Workspace/SIPPY-master/sippy_unipi/Parsim_methods.py`
- `/Users/josephj/Workspace/SIPPY-master/sippy_unipi/functionsetSIM.py`

**Harold Branch Files:**
- `/Users/josephj/Workspace/SIPPY/src/sippy/identification/algorithms/parsim_core.py`
- `/Users/josephj/Workspace/SIPPY/src/sippy/identification/algorithms/parsim_k.py`
- `/Users/josephj/Workspace/SIPPY/src/sippy/identification/algorithms/parsim_s.py`
- `/Users/josephj/Workspace/SIPPY/src/sippy/identification/algorithms/parsim_p.py`

---

**Report Generated:** 2025-10-12
**Investigator:** Claude Code
**Status:** READ-ONLY INVESTIGATION - NO CHANGES MADE
