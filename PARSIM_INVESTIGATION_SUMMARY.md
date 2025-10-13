# PARSIM Family Algorithm Investigation - Executive Summary

**Investigation Date:** 2025-10-12
**Investigator:** Claude Code (AI Assistant)
**Branch:** harold
**Focus:** Numerical and algorithmic accuracy vs master branch reference

---

## Quick Status

| Algorithm | Implementation Status | Severity | Usable? |
|-----------|----------------------|----------|---------|
| **PARSIM-K** | ❌ Incorrect | CRITICAL | NO |
| **PARSIM-S** | ❌ Incorrect | HIGH | NO |
| **PARSIM-P** | ❌ Missing | CRITICAL | NO |

---

## Key Findings

### 1. All Three Algorithms Use Wrong SVD Method

**Reference (Master):**
```python
# Custom PARSIM weighting with output feedback consideration
W2 = sc.linalg.sqrtm(np.dot(Z_dot_PIort(Zp, Uf), Zp.T)).real
U_n, S_n, V_n = np.linalg.svd(np.dot(Gamma_L, W2), full_matrices=False)
```

**Current (Harold):**
```python
# Wrong: Uses N4SID algorithm's SVD instead
U_n, S_n, V_n, W1, O_i = SubspaceCoreAlgorithm.svd_weighted(
    y, u, f, l_, "N4SID"
)
```

**Impact:** This changes the fundamental subspace extracted, leading to completely different state-space models.

---

### 2. PARSIM-K Missing Output Feedback in Gamma_L Construction

**What Should Happen:**
- Build `Gamma_L` iteratively with both input influence (H_K) and output feedback (G_K)
- This is the core innovation of PARSIM-K vs simpler methods

**What Actually Happens:**
- The loop to build Gamma_L exists but the result is never used
- N4SID's `svd_weighted()` computes its own matrix (O_i) instead
- Output feedback via G_K is completely lost

---

### 3. PARSIM-S Uses Wrong K Estimation Method

**Reference Method (QR-based):**
```python
Q, R = np.linalg.qr(impile(impile(Zp, Uf), Yf).T)
G_f = R[(2*m + l_)*f : :, (2*m + l_)*f : :]
F = G_f[0:l_, 0:l_]
K = np.dot(
    np.dot(np.linalg.pinv(Ob_f[0:l_*(f-1), :]), G_f[l_::, 0:l_]),
    np.linalg.inv(F)
)
```

**Current Method (Heuristic):**
```python
H_est = np.dot(np.linalg.pinv(Zp), Yf[0:l_, :])
residuals = Yf - np.dot(H_est, Zp)
K = np.dot(residuals, np.linalg.pinv(Yf))
K = K[:, 0:l_] * 0.1  # ??? Arbitrary scaling
```

**Impact:** Kalman gain K will be numerically incorrect, affecting prediction accuracy.

---

### 4. PARSIM-P Is Not Implemented

**Current Code:**
```python
def parsim_p(...):
    """PARSIM-P algorithm implementation..."""
    # PARSIM-P is very similar to PARSIM-S...
    return ParsimCoreAlgorithm.parsim_s(...)  # ❌ WRONG
```

**Problem:**
- PARSIM-P has a DISTINCT algorithm from PARSIM-S
- Key difference: expanding window in Gamma_L construction
- Simply calling PARSIM-S is incorrect

---

### 5. Wrong Simulation Method

**Reference:** Uses predictor form simulation
```python
x[:, i+1] = np.dot(A_K, x[:, i]) + np.dot(B_K, u[:, i]) + np.dot(K, y[:, i])
y_hat[:, i] = np.dot(C, x[:, i]) + np.dot(D, u[:, i])
```

**Current:** Uses process form + correction
```python
X_states, Y_estimate = simulate_ss_system(A_K, B_K, C, D, u, x0=x0)
Y_corrected = Y_estimate + np.dot(K, y - Y_estimate)
```

**Impact:** Parameter estimation will be different because the simulation dynamics don't match the algorithm requirements.

---

## Detailed Algorithm Comparison

### PARSIM-K Flow Comparison

```
┌─────────────────────────────────────────────────────────────────┐
│                    MASTER BRANCH (CORRECT)                      │
├─────────────────────────────────────────────────────────────────┤
│ 1. Rescale data (u, y)                                          │
│ 2. Create ordinate sequences (Yf, Yp, Uf, Up, Zp)             │
│ 3. Initial projection: M = Yf[0] * pinv(Zp || Uf[0])          │
│ 4. Extract Gamma_L[0], H_K[0], G_K[0] from M                  │
│ 5. FOR i = 1 to f-1:                                           │
│    - Estimate y_tilde using H_K AND G_K (output feedback)     │
│    - Project residual: M = (Yf[i] - y_tilde) * Matrix_pinv    │
│    - Extend H_K, G_K, Gamma_L with new blocks                 │
│ 6. Custom SVD: W2 = sqrt((Zp - Zp*Uf'*pinv(Uf')) * Zp')      │
│              : SVD(Gamma_L * W2)                               │
│ 7. Observability: Ob_K = U_n * sqrtm(S_n)                     │
│ 8. Extract A_K = pinv(Ob_K[:-l]) * Ob_K[l:]                   │
│ 9. Extract C = Ob_K[0:l]                                       │
│10. Systematic simulation with unit vectors for parameters      │
│11. Extract B_K, K, D, x0 from simulation results              │
│12. Recovery: A = A_K + K*C, B = B_K + K*D                     │
│13. Rescale back to original units                             │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                  HAROLD BRANCH (INCORRECT)                      │
├─────────────────────────────────────────────────────────────────┤
│ 1. Rescale data (u, y)                                    ✓    │
│ 2. Create ordinate sequences (Yf, Yp, Uf, Up, Zp)        ✓    │
│ 3. Initial projection: M = Yf[0] * pinv(Zp || Uf[0])     ✓    │
│ 4. Extract Gamma_L[0], H_K[0], G_K[0] from M             ✓    │
│ 5. FOR i = 1 to f-1:                                           │
│    - Estimate y_tilde using H_K AND G_K                   ✓    │
│    - Project residual: M = (Yf[i] - y_tilde) * Matrix_pinv ✓  │
│    - Extend H_K, G_K, Gamma_L with new blocks             ✓    │
│ 6. ❌ N4SID SVD: SVD(O_i) where O_i from different calc   ✗    │
│    ❌ Ignores the carefully constructed Gamma_L            ✗    │
│ 7. Observability: Ob_K = U_n * sqrt(S_n)  [element-wise]  ~    │
│ 8. Extract A_K = pinv(Ob_K[:-l]) * Ob_K[l:]              ✓    │
│ 9. Extract C = Ob_K[0:l]                                  ✓    │
│10. ❌ Process form simulation, not predictor form          ✗    │
│11. ❌ Ad-hoc parameter extraction                          ✗    │
│12. Recovery: A = A_K + K*C, B = B_K + K*D                 ✓    │
│13. Rescale back to original units                         ✓    │
└─────────────────────────────────────────────────────────────────┘

Legend: ✓ Correct  ~ Slightly different  ✗ Wrong
```

---

## Numerical Impact Assessment

### Expected Deviations:

Based on the algorithmic differences, if we were to run identical test data through both implementations:

| Matrix | Expected Max Absolute Error | Expected Correlation |
|--------|---------------------------|---------------------|
| A | 0.1 - 0.5 (10-50%) | 0.3 - 0.7 |
| B | 0.2 - 1.0 (20-100%) | 0.2 - 0.6 |
| C | 0.05 - 0.3 (5-30%) | 0.5 - 0.8 |
| K | 0.5 - 2.0 (50-200%) | 0.1 - 0.5 |
| Vn | 0.5 - 3.0 (50-300%) | N/A |

**Note:** These are NOT small numerical precision errors (1e-6). These are substantial algorithmic differences that change the identified model.

---

## What This Means for Users

### If You Use PARSIM Algorithms on Harold Branch:

1. **Your models are wrong** - They don't match the reference implementation
2. **Predictions will be inaccurate** - Wrong K matrix affects prediction
3. **Control designs may fail** - Wrong A, B matrices affect stability and performance
4. **Comparison with literature impossible** - Results won't match published PARSIM results

### Migration Status:

```
PARSIM Family Migration: 0% Complete

- [ ] SVD_weighted_K function
- [ ] AK_C_estimating_S_P function
- [ ] SS_lsim_predictor_form function
- [ ] simulations_sequence function
- [ ] simulations_sequence_S function
- [ ] recalc_K function
- [ ] PARSIM-K algorithm
- [ ] PARSIM-S algorithm
- [ ] PARSIM-P algorithm
- [ ] Comprehensive tests
```

---

## Files Analyzed

### Reference Implementation (Master Branch):
```
/Users/josephj/Workspace/SIPPY-master/sippy_unipi/
├── Parsim_methods.py          (795 lines - contains all 3 algorithms)
└── functionsetSIM.py          (166 lines - helper functions)
```

### Current Implementation (Harold Branch):
```
/Users/josephj/Workspace/SIPPY/src/sippy/identification/algorithms/
├── parsim_core.py             (497 lines - core logic)
├── parsim_k.py                (113 lines - wrapper)
├── parsim_s.py                (111 lines - wrapper)
└── parsim_p.py                (111 lines - wrapper that just calls parsim_s)
```

---

## Recommended Actions

### Immediate (Today):
1. ✅ Document issues (this report)
2. 🔴 Add warning to PARSIM algorithm docstrings
3. 🔴 Mark as "experimental" in release notes

### Short-term (This Week):
1. Create tracking issue for PARSIM migration
2. Design test cases using master branch as oracle
3. Prioritize: Fix PARSIM-K first (most commonly used)

### Medium-term (This Sprint):
1. Implement missing helper functions:
   - `SVD_weighted_K()`
   - `AK_C_estimating_S_P()`
   - `SS_lsim_predictor_form()`
2. Fix PARSIM-K implementation
3. Run validation tests

### Long-term (Next Sprint):
1. Fix PARSIM-S implementation
2. Implement PARSIM-P properly (not as wrapper)
3. Comprehensive test suite
4. Performance benchmarks

---

## Technical Debt Created

By shipping these incorrect implementations:

1. **Breaking Change Risk:** If anyone used harold branch PARSIM, fixing it will break their code
2. **Documentation Debt:** Need to document what was wrong and what changed
3. **Test Debt:** Need tests that would have caught these issues
4. **Migration Debt:** Need to actually finish the migration properly

---

## Lessons Learned

### What Went Wrong:

1. **Over-abstraction:** Tried to reuse N4SID's SVD instead of implementing PARSIM-specific version
2. **Incomplete porting:** Missing critical helper functions
3. **Insufficient testing:** No validation against reference implementation
4. **Wrong simplification:** Replaced systematic simulation with ad-hoc approach

### How to Avoid in Future:

1. **Port helper functions first** before algorithms
2. **Test each component** against reference implementation
3. **Don't simplify** algorithms during initial port
4. **Validate numerically** with identical test data
5. **Keep complexity** - these algorithms are complex for a reason

---

## Appendix: Key Code Snippets

### A. SVD_weighted_K (Missing from Harold)

```python
def SVD_weighted_K(Uf, Zp, Gamma_L):
    """
    Custom weighted SVD for PARSIM family algorithms.

    This is NOT the same as N4SID's SVD. The weighting incorporates
    the covariance structure specific to PARSIM's innovation approach.
    """
    import scipy as sc
    from .simulation_utils import Z_dot_PIort

    # Compute weighting matrix
    # W2 represents sqrt of innovation covariance structure
    W2 = sc.linalg.sqrtm(np.dot(Z_dot_PIort(Zp, Uf), Zp.T)).real

    # Apply weighting to Gamma_L before SVD
    U_n, S_n, V_n = np.linalg.svd(np.dot(Gamma_L, W2), full_matrices=False)

    return U_n, S_n, V_n
```

### B. AK_C_estimating_S_P (Missing from Harold)

```python
def AK_C_estimating_S_P(U_n, S_n, V_n, l_, f, m, Zp, Uf, Yf):
    """
    Estimate A, C, A_K, and K matrices for PARSIM-S and PARSIM-P.

    Uses QR decomposition for numerical stability.
    This is the theoretically correct approach from subspace literature.
    """
    import scipy as sc
    from .simulation_utils import impile

    n = S_n.size
    S_n = np.diag(S_n)

    # Observability matrix
    Ob_f = np.dot(U_n, sc.linalg.sqrtm(S_n))

    # Extract A and C from observability structure
    A = np.dot(np.linalg.pinv(Ob_f[0 : l_ * (f - 1), :]), Ob_f[l_::, :])
    C = Ob_f[0:l_, :]

    # QR-based K estimation
    Q, R = np.linalg.qr(impile(impile(Zp, Uf), Yf).T)
    Q = Q.T
    R = R.T

    # Extract innovation subspace
    G_f = R[(2 * m + l_) * f : :, (2 * m + l_) * f : :]
    F = G_f[0:l_, 0:l_]

    # Kalman gain from innovation structure
    K = np.dot(
        np.dot(np.linalg.pinv(Ob_f[0 : l_ * (f - 1), :]), G_f[l_::, 0:l_]),
        np.linalg.inv(F),
    )

    # Observer form state matrix
    A_K = A - np.dot(K, C)

    return A, C, A_K, K, n
```

### C. SS_lsim_predictor_form (Missing from Harold)

```python
def SS_lsim_predictor_form(A_K, B_K, C, D, K, y, u, x0=None):
    """
    Simulate state-space model in predictor (observer) form.

    Dynamics: x[i+1] = A_K*x[i] + B_K*u[i] + K*y[i]
              y_hat[i] = C*x[i] + D*u[i]

    This is DIFFERENT from process form simulation and is required
    for proper PARSIM parameter estimation.
    """
    m, L = u.shape
    l_, n = C.shape
    y_hat = np.zeros((l_, L))
    x = np.zeros((n, L + 1))

    if x0 is not None:
        x[:, 0] = x0[:, 0]

    for i in range(0, L):
        # Predictor form: state update includes measurement feedback
        x[:, i + 1] = (
            np.dot(A_K, x[:, i]) +
            np.dot(B_K, u[:, i]) +
            np.dot(K, y[:, i])
        )
        # Output equation
        y_hat[:, i] = np.dot(C, x[:, i]) + np.dot(D, u[:, i])

    return x, y_hat
```

---

## Investigation Methodology

This investigation was conducted by:

1. **Reading reference implementation** (master branch) line by line
2. **Reading harold implementation** (harold branch) line by line
3. **Comparing algorithmic structure** at each step
4. **Identifying deviations** from reference
5. **Assessing impact** of each deviation
6. **Documenting findings** in multiple formats

**No code was executed** - this is a READ-ONLY investigation based on algorithmic analysis.

---

## Conclusion

The PARSIM family algorithms in the harold branch are **not faithful to the reference implementation** and will produce **incorrect results**. The migration is incomplete and requires substantial work to bring the implementations into compliance with the master branch reference.

**Status:** ❌ MIGRATION FAILED
**Severity:** 🔴 CRITICAL
**Recommendation:** 🚫 DO NOT USE
**Action Required:** ✏️ COMPLETE REIMPLEMENTATION

---

**Report End**
