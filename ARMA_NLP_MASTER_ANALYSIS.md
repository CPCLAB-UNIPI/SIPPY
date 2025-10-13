# ARMA NLP Master Branch Analysis

**Date:** 2025-10-13
**Objective:** Comprehensive analysis of the master branch ARMA NLP implementation using CasADi/IPOPT for accurate reimplementation in harold branch

---

## Executive Summary

The master branch ARMA implementation uses **CasADi-based nonlinear programming (NLP)** with the **IPOPT solver** to estimate model parameters through constrained optimization. ARMA is a **time-series-only** identification method (no input dynamics) that estimates autoregressive (AR) and moving average (MA) components.

**Key Findings:**
- Optimization library: **CasADi + IPOPT** (Interior Point Optimizer)
- Cost function: Minimizes squared prediction error between Y and Yid
- Decision variables: Model coefficients (a, c) + auxiliary time series (Yid, noise estimates)
- Constraints: Equality constraints for consistency + optional stability constraints
- ARMA differs from ARMAX: **NO input dynamics** (G=1), only noise modeling (H=C/A)
- **nb forced to 0** internally, ARMA API has only 3 parameters: [na, nc, theta]

**Expected Accuracy Target:** < 10% error vs master branch (MA estimation is inherently challenging)

---

## 1. Algorithm Overview

### 1.1 ARMA Model Structure

The ARMA model is defined as:
```
A(q) y(k) = C(q) e(k)
```

Where:
- **A(q)** = 1 + a₁q⁻¹ + ... + aₙₐq⁻ⁿᵃ (autoregressive polynomial, na coefficients)
- **C(q)** = 1 + c₁q⁻¹ + ... + cₙcq⁻ⁿᶜ (moving average polynomial, nc coefficients)
- **θ** = delay parameter (usually 0 for ARMA, used for indexing but no input delay)
- **e(k)** = white noise process

Transfer functions:
- **G(q) = 1** (NO input transfer function - ARMA is time-series only)
- **H(q) = C(q) / A(q)** (noise transfer function)

**Mathematical form (difference equation):**
```
y(k) = -a₁·y(k-1) - a₂·y(k-2) - ... - aₙₐ·y(k-na)
       + e(k) + c₁·e(k-1) + c₂·e(k-2) + ... + cₙc·e(k-nc)
```

### 1.2 ARMA vs ARMAX vs ARARX Comparison

| Aspect | ARMA | ARMAX | ARARX |
|--------|------|-------|-------|
| **API** | [na, nc, theta] (3 params) | [na, nb, nc, theta] (4 params) | [na, nb, nd, theta] (4 params) |
| **Input Dynamics** | None (G=1) | Yes (B/A) | Yes (B/(A·D)) |
| **Regressor** | [-y, e] | [-y, u, e] | [-y, u, -V] |
| **Auxiliary Variables** | Yid, Epsi (2·N) | Yid, Epsi (2·N) | Yid, W, V (3·N) |
| **Use Case** | Time-series only | Input-output systems | Input-output with denominator dynamics |
| **Coefficients** | [a, c] | [a, b, c] | [a, b, d] |
| **n_opt** | na + nc + 2·N | na + nb + nc + 2·N | na + nb + nd + 3·N |

**Key Difference from ARARX:**
- ARMA has **NO input terms** (nb=0 forced internally)
- ARMA has **NO auxiliary variables W and V** (only Yid and Epsi)
- ARMA uses **nc (MA terms)** instead of nd (denominator terms)

### 1.3 Key Differences from Harold Branch (Current)

| Aspect | Master Branch (NLP) | Harold Branch (Current) |
|--------|-------------------|------------------------|
| **Method** | Single-shot NLP with IPOPT | Iterative Least Squares (ILLS) |
| **Solver** | CasADi symbolic + IPOPT | NumPy least squares |
| **Variables** | Coefficients + full time series | Coefficients only |
| **Constraints** | Explicit equality constraints | Implicit through iterations |
| **Stability** | Optional companion matrix norm constraints | No stability enforcement |
| **Convergence** | IPOPT interior point method | Manual iteration with tolerance check |
| **Speed** | ~1-2 sec for N=1000 | ~10 ms for N=1000 |
| **Accuracy** | Reference implementation | 71-2600% error vs master |

---

## 2. Optimization Problem Formulation

### 2.1 Decision Variables

The optimization vector `w_opt` has **n_opt** elements:

```python
n_opt = n_coeff + N  # For ARMA (Nc != 0, Nd = 0)

where:
  n_coeff = na + nc  # Number of polynomial coefficients (NO nb for ARMA)
  N = len(y)         # Number of time samples
```

**Structure of w_opt:**
```
w_opt = [a₁, a₂, ..., aₙₐ,           # A polynomial coefficients (na elements)
         c₁, c₂, ..., cₙc,           # C polynomial coefficients (nc elements)
         Yid[0], Yid[1], ..., Yid[N-1]]  # Predicted output (N elements)
```

**Coefficient extraction** (from `functionset_OPT.py`, lines 52-96):
```python
# Symbolic variables
a = w_opt[0:na]                              # A coefficients
c = w_opt[na : na + nc]                      # C coefficients (NO b for ARMA)

# Auxiliary time series
Yidw = w_opt[-N:]                             # Predicted output (only 1 aux variable)
```

**Key difference from ARARX:**
- ARMA has **na + nc** coefficients (not na + nb + nd)
- ARMA has **only Yidw** auxiliary variable in optimization (not Ww, Vw, Yidw)
- Total variables: `n_opt = na + nc + N` (much smaller than ARARX's `na + nb + nd + 3*N`)

### 2.2 Cost Function (Objective)

The objective function minimizes the **mean squared prediction error** (line 189):

```python
DY = Y - Yidw
f_obj = (1.0 / N) * mtimes(DY.T, DY)
```

**Mathematical form:**
```
J = (1/N) * Σ[k=0 to N-1] (y[k] - Yid[k])²
```

This is the **standard prediction error method (PEM)** objective - **identical to ARARX**.

### 2.3 Constraints

The NLP includes **two types of constraints**:

#### 2.3.1 Equality Constraints (Lines 197-202)

For ARMA, only **one equality constraint** is used (not three like ARARX):

```python
g = []
g.append(Yid - Yidw)    # Predicted output must match optimization variable
# NO W - Ww constraint (W not used in ARMA)
# NO V - Vw constraint (V not used in ARMA)
```

**Constraint bounds:**
```python
g_lb = -1e-7 * ones(ng, 1)
g_ub = 1e-7 * ones(ng, 1)
```

These are **near-zero tolerances** (±1e-7), effectively equality constraints.

#### 2.3.2 Stability Constraints (Lines 204-260, Optional)

If `stab_cons=True`, the optimizer adds **companion matrix infinity-norm constraints**:

For each polynomial P(q) = 1 + p₁q⁻¹ + ... + pₙq⁻ⁿ, construct the **companion matrix**:

```
       [  0   1   0  ...  0 ]
       [  0   0   1  ...  0 ]
CompP = [ ...............  ]
       [  0   0   0  ...  1 ]
       [-pₙ -pₙ₋₁ ... -p₁ ]
```

**Constraint:** `||CompP||∞ ≤ stab_marg` (default stab_marg = 1.0)

This is applied to:
- **A(q) companion matrix** (if na > 0, lines 207-219)
- **C(q) companion matrix** (if nc > 0, would need to add - NOT present in master code!)

**Important Note:** Master code does **NOT** add stability constraint for C(q) polynomial. Only A(q) is constrained.

**Mathematical property:** `||CompP||∞ ≥ spectral_radius(CompP)`, so this constraint ensures poles inside the unit circle scaled by `stab_marg`.

### 2.4 Variable Bounds

Coefficients and auxiliary variables are **bounded** (lines 74-78):

```python
w_lb = -100 * ones(n_opt)
w_ub = +100 * ones(n_opt)
```

This prevents unbounded growth during optimization.

---

## 3. Auxiliary Variable Definitions

### 3.1 Noise Estimates (Epsi) - Internal Symbolic Variable

**Purpose:** Represents the prediction error (noise) at each time step

**Definition (lines 115-116, 144-145, 168-169):**
```python
if Nc != 0:
    Epsi = SX.zeros(N)   # Preallocate symbolic noise vector

# ... in loop ...
if Nc != 0:
    vecE = Epsi[k - Nc : k][::-1]   # Past prediction errors

# Update prediction error
if Nc != 0:
    Epsi[k] = Y[k] - Yidw[k]
```

**Mathematical form:**
```
Epsi[k] = y[k] - Yid[k]
```

**Important:** Epsi is **NOT an optimization variable** - it's a symbolic expression derived from Y and Yidw.

### 3.2 Predicted Output Yid (Lines 118-165)

**Purpose:** One-step-ahead prediction using the ARMA model structure

**Regressor construction** (lines 121-162):

For ARMA (line 156):
```python
phi = vertcat(-vecY, vecE)
```

Where:
- `vecY = Y[k-na : k][::-1]` - Output lags (na elements, reversed)
- `vecE = Epsi[k-nc : k][::-1]` - Prediction error lags (nc elements, reversed)

**Prediction** (line 165):
```python
coeff = vertcat(a, c)  # For ARMA (line 95)
Yid[k] = mtimes(phi.T, coeff)
```

**Mathematical form:**
```
Yid[k] = -a₁·y[k-1] - a₂·y[k-2] - ... - aₙₐ·y[k-na]
         + c₁·e[k-1] + c₂·e[k-2] + ... + cₙc·e[k-nc]

where e[k] = y[k] - Yid[k]
```

**Key difference from ARARX:**
- ARMA regressor has **NO input terms** (no vecU)
- ARMA regressor has **NO V auxiliary terms** (no vecV)
- ARMA regressor uses **prediction errors Epsi** instead of auxiliary variables

### 3.3 No W and V Auxiliary Variables

**Critical difference:** ARMA does **NOT** use W and V auxiliary variables because:
- W represents `B/D * u` - but ARMA has no input (nb=0)
- V represents `A*y - W` - but without W, this is not needed
- The condition `if Nd != 0` (lines 40, 104, 136) is **FALSE** for ARMA (nd=0 for ARMA)

**Code path for ARMA:**
```python
if Nd != 0:  # FALSE for ARMA
    n_aus = 3 * N
else:        # ARMA takes this branch
    n_aus = N
```

So ARMA has:
- `n_aus = N` (only Yid as auxiliary variable)
- `n_opt = n_coeff + N` (not n_coeff + 3*N like ARARX)

---

## 4. Algorithm Step-by-Step

### 4.1 Initialization (Lines 15-48, io_opt.py)

**Data preprocessing:**
```python
ystd, y = rescale(y)       # Normalize output to zero mean, unit std
# NO input rescaling for ARMA (no inputs!)
```

**Determine problem dimensions:**
```python
val = max(na, nc)          # Max lag (no nb, nf, nd for ARMA)
m = 1                      # Number of inputs (irrelevant for ARMA)
p = 1                      # Number of outputs
n_coeff = na + nc          # For ARMA (NO nb)
n_opt = n_coeff + N        # Optimization variables (only Yid, no W or V)
```

**Initial guess** (lines 51-60, io_opt.py):
```python
w_0 = zeros(1, n_coeff)    # Start with zero coefficients
w_0 = hstack([w_0, y])     # Initialize Yid = y (measured output)
# NO additional hstack for W and V (ARMA does not use them)
```

**Key difference:** ARMA initial guess is **[0, ..., 0, y[0], y[1], ..., y[N-1]]** with length `na + nc + N`.

### 4.2 Build Regressor Loop (Lines 118-185)

**For each time step k ≥ val:**

1. **Build output regressor** (lines 132-133):
   ```python
   if Na != 0:
       vecY = Y[k-na : k][::-1]
   ```

2. **Build prediction error regressor** (lines 144-145):
   ```python
   if Nc != 0:
       vecE = Epsi[k-nc : k][::-1]
   ```

3. **Construct ARMA regressor** (line 156):
   ```python
   phi = vertcat(-vecY, vecE)
   ```

4. **Update prediction** (line 165):
   ```python
   Yid[k] = mtimes(phi.T, coeff)
   ```

5. **Update prediction error** (lines 168-169):
   ```python
   if Nc != 0:
       Epsi[k] = Y[k] - Yidw[k]
   ```

**Important differences from ARARX:**
- **NO input regressor** (no vecU construction, lines 122-129 skipped)
- **NO W auxiliary update** (lines 172-177 skipped for ARMA)
- **NO V auxiliary update** (lines 179-184 skipped for ARMA)
- Only **Yid and Epsi** are updated

**Loop pseudocode:**
```python
for k in range(N):
    if k >= val:  # val = max(na, nc)
        # Build regressors
        vecY = Y[k-na:k][::-1]         # Past outputs
        vecE = Epsi[k-nc:k][::-1]      # Past prediction errors

        # ARMA regressor
        phi = vertcat(-vecY, vecE)

        # One-step prediction
        Yid[k] = phi^T · [a, c]

        # Prediction error
        Epsi[k] = Y[k] - Yidw[k]
```

### 4.3 Solve NLP (Lines 63, io_opt.py)

**IPOPT solver call:**
```python
sol = solver(lbx=w_lb, ubx=w_ub, x0=w_0, lbg=g_lb, ubg=g_ub)
```

**Solver configuration** (lines 269-277):
```python
sol_opts = {
    'ipopt.max_iter': max_iterations,      # Default 200
    'ipopt.print_level': 0,                # Suppress output
    'ipopt.sb': 'yes',                     # Suppress banner
    'print_time': 0                        # Suppress timing
}
solver = nlpsol('solver', 'ipopt', nlp, sol_opts)
```

IPOPT uses:
- **Interior point method** with barrier functions
- **Quasi-Newton approximation** of Hessian (default L-BFGS)
- **Line search** for global convergence
- **Constraint satisfaction** through penalty/barrier terms

**Identical to ARARX** - only the problem structure differs.

### 4.4 Extract Solution (Lines 65-117, io_opt.py)

**Optimization output:**
```python
x_opt = sol['x']                      # Optimal variables
y_id = x_opt[-N:].full()[:, 0]        # Extract Yid
THETA = array(x_opt[:n_coeff])[:, 0]  # Extract coefficients

# Estimated variance
Vn = (||y_id - y||₂)² / (2*N)
```

**Coefficient extraction:**
```python
# For ARMA:
A_coeffs = THETA[0:na]
C_coeffs = THETA[na : na+nc]
# NO B_coeffs (nb=0 for ARMA)
```

**Build transfer functions** (lines 75-115):
```python
# G(z) for ARMA
if id_method == "ARMA":
    NUM = 1.0  # G(z) = 1 (no input dynamics)
else:
    NUM = np.zeros(valG)
    NUM[theta : nb + theta] = THETA[na : nb + na]

# Build A polynomial
A = tf([1, zeros(na)], [1, A_coeffs], 1)
F = tf([1, zeros(nf)], [1, F_coeffs], 1)  # F = 1 for ARMA (nf=0)

# Denominator for G
_, deng = tfdata(A * F)  # For ARMA: just A since F=1
DEN_G = zeros(valG + 1)
DEN_G[0 : na + 1] = deng

# H(z) for ARMA: C(z) / A(z)
if id_method == "OE":
    NUMH = 1.0
else:  # ARMA, ARMAX, etc.
    NUMH = zeros(valH + 1)
    NUMH[0] = 1.0
    NUMH[1 : nc + 1] = THETA[na : na + nc]  # C coefficients

# Build D polynomial
D = tf([1, zeros(nd)], [1, D_coeffs], 1)  # D = 1 for ARMA (nd=0)

# Denominator for H
_, denh = tfdata(A * D)  # For ARMA: just A since D=1
DENH = zeros(valH + 1)
DENH[0 : na + 1] = denh

# Transfer functions
G = tf(NUM, DEN_G, tsample)  # G = 1 for ARMA
H = tf(NUMH, DENH, tsample)  # H = C(z)/A(z) for ARMA
```

**Rescale coefficients** (lines 262-268, io_opt.py):
```python
# For ARMA, skip B rescaling (no input coefficients)
if id_method != "ARMA":
    NUM[theta_min : nb_min + theta_min] = (
        NUM[theta_min : nb_min + theta_min] * ystd / Ustd
    )

# Rescale output
Y_id = y_id * ystd
```

**Key difference:** ARMA has **no B coefficient rescaling** since there are no input coefficients.

---

## 5. Key Implementation Details

### 5.1 CasADi Symbolic Construction

**Purpose:** CasADi constructs a **symbolic computation graph** for automatic differentiation.

**Variable declaration** (line 49):
```python
w_opt = SX.sym('w', n_opt)  # Symbolic optimization variables
```

For ARMA:
```python
n_opt = na + nc + N  # Much smaller than ARARX
```

**Matrix operations use CasADi functions:**
- `vertcat()` - Vertical concatenation
- `mtimes()` - Matrix multiplication
- `norm_inf()` - Infinity norm

**Constraint construction** (line 250):
```python
g_ = vertcat(*g)  # Stack all constraints
```

**NLP dictionary** (line 265):
```python
nlp = {'x': w_opt, 'f': f_obj, 'g': g_}
```

This symbolic formulation allows IPOPT to compute **exact gradients** and **Jacobians**.

**Identical to ARARX** - CasADi usage is the same.

### 5.2 Data Preprocessing

**Rescaling function** (from `functionset.py`, lines 123-126):
```python
def rescale(data):
    ystd = std(data)
    data_scaled = data / ystd
    return ystd, data_scaled
```

**Why rescale?**
- Improves **numerical conditioning** (all variables ~O(1))
- Prevents ill-conditioning when outputs have large magnitude
- Standard practice in NLP

**Rescaling applied:**
- Output: `ystd, y = rescale(y)`
- **NO input rescaling** (ARMA has no inputs)

**Coefficient rescaling at end:**
```python
# NO B coefficient rescaling for ARMA (no B coefficients)
Y_id = y_id * ystd  # Restore output scaling
```

**Key difference:** ARMA only rescales output, not inputs (since there are no inputs).

### 5.3 Initial Conditions Handling

**Warm start strategy** (lines 51-60, io_opt.py):

```python
w_0 = zeros(1, n_coeff)       # Coefficients start at zero
w_0 = hstack([w_0, y])        # Yid starts as measured output
# NO additional W and V initialization for ARMA
```

**Why this works:**
- If all coefficients are zero, prediction Yid ≈ y is a reasonable initial guess
- IPOPT can quickly adjust from this starting point
- Avoids local minima from random initialization

**Time-stepping in regressor:**
- Lags are handled naturally by indexing `Y[k-na:k]`, `Epsi[k-nc:k]`
- Initial samples `k < val` are **not identifiable** (insufficient lag data)
- Loop starts at `k = val` where `val = max(na, nc)` (line 118)

### 5.4 Stability Constraints (Optional)

**Companion matrix construction** (lines 209-213 for A(q)):

```python
compA = zeros(na, na)
diagA = eye(na - 1)
compA[:-1, 1:] = diagA        # Super-diagonal identity
compA[-1, :] = -a[::-1]        # Last row: negative reversed coefficients
```

**Example for A(q) = 1 + a₁q⁻¹ + a₂q⁻² (na=2):**
```
compA = [  0    1  ]
        [ -a₂  -a₁ ]
```

**Eigenvalues of compA = roots of A(q)**

**Stability check:**
```python
norm_CompA = norm_inf(compA)
g.append(norm_CompA)
g_ub[-ng_norm:] = stab_marg * ones(ng_norm, 1)
```

**Default stab_marg = 1.0** → poles strictly inside unit circle
**For robustness, use stab_marg < 1.0** (e.g., 0.95)

**Important Note:** Master code does **NOT** add stability constraint for C(q) polynomial. Only A(q) is constrained. To add C(q) constraint, would need:

```python
if Nc != 0:
    ng_norm += 1
    compC = SX.zeros(Nc, Nc)
    diagC = SX.eye(Nc - 1)
    compC[:-1, 1:] = diagC
    compC[-1, :] = -c[::-1]
    norm_CompC = norm_inf(compC)
    g.append(norm_CompC)
```

### 5.5 Convergence Criteria

**IPOPT internal criteria:**
- Dual feasibility: `||∇L|| < tol` (gradient of Lagrangian)
- Primal feasibility: `||g(x)|| < constr_tol`
- Complementarity: `|μᵢcᵢ| < tol` (KKT conditions)

**Default tolerances:**
- `tol = 1e-8`
- `constr_viol_tol = 1e-4`
- `max_iter = 200` (from master code)

**Solution quality:**
```python
iterations = solver.stats()['iter_count']
if iterations >= max_iterations:
    print('Warning! Reached maximum iterations')
```

**Identical to ARARX** - IPOPT convergence criteria are solver-specific.

---

## 6. Implementation Specification for Harold Branch

### 6.1 Required Dependencies

```python
import numpy as np
from casadi import DM, SX, vertcat, mtimes, norm_inf, nlpsol
import harold  # For transfer function creation
```

**CasADi installation:**
```bash
pip install casadi
```

**Check IPOPT availability:**
```python
try:
    nlpsol('test', 'ipopt', {'x': SX.sym('x'), 'f': 0}, {})
    IPOPT_AVAILABLE = True
except:
    IPOPT_AVAILABLE = False
```

### 6.2 Pseudocode for ARMA NLP

```python
def identify_arma_nlp(y, na, nc, theta, tsample, max_iterations=200,
                      stab_marg=1.0, stab_cons=False):
    """
    ARMA identification using NLP (CasADi + IPOPT).

    Args:
        y: Output array (1 x N) - time series data
        na: A polynomial order (autoregressive order)
        nc: C polynomial order (moving average order)
        theta: Delay parameter (usually 0 for ARMA, used for indexing)
        tsample: Sampling time
        max_iterations: IPOPT max iterations
        stab_marg: Stability margin (< 1.0)
        stab_cons: Enable stability constraints

    Returns:
        A_coeffs, C_coeffs, Yid, Vn, G_tf, H_tf
    """

    # 1. Preprocess data (rescale output only, no inputs)
    ystd, y_scaled = rescale(y)
    N = y.shape[0] if y.ndim == 1 else y.shape[1]

    # 2. Problem dimensions
    n_coeff = na + nc           # Only A and C coefficients
    n_opt = n_coeff + N         # Coefficients + Yid (NO W, V for ARMA)
    val = max(na, nc)           # Maximum lag

    # 3. Define symbolic optimization variables
    w_opt = SX.sym('w', n_opt)
    a = w_opt[0:na]                # A coefficients
    c = w_opt[na : na+nc]          # C coefficients
    Yidw = w_opt[-N:]              # Predicted output (only aux variable)

    # 4. Initialize auxiliary variables
    Yid = y_scaled * SX.ones(1)    # Symbolic copy
    Epsi = SX.zeros(N)             # Prediction errors (symbolic)

    # 5. Build regressor loop
    coeff = vertcat(a, c)          # ARMA coefficient vector

    for k in range(N):
        if k >= val:
            # Output lags
            vecY = y_scaled[k-na : k][::-1]

            # Prediction error lags
            vecE = Epsi[k-nc : k][::-1]

            # ARMA regressor: [-y lags, e lags]
            phi = vertcat(-vecY, vecE)

            # One-step prediction
            Yid[k] = mtimes(phi.T, coeff)

            # Prediction error
            Epsi[k] = y_scaled[k] - Yidw[k]

    # 6. Objective function
    DY = y_scaled - Yidw
    f_obj = (1.0 / N) * mtimes(DY.T, DY)

    # 7. Constraints (only Yid consistency for ARMA)
    g = []
    g.append(Yid - Yidw)    # Prediction consistency

    # 8. Stability constraints (optional, only for A polynomial)
    ng_norm = 0
    if stab_cons:
        if na > 0:
            compA = SX.zeros(na, na)
            compA[:-1, 1:] = SX.eye(na-1)
            compA[-1, :] = -a[::-1]
            g.append(norm_inf(compA))
            ng_norm += 1

        # Optional: Add C polynomial stability constraint
        # (NOT in master code, but could be useful)
        # if nc > 0:
        #     compC = SX.zeros(nc, nc)
        #     compC[:-1, 1:] = SX.eye(nc-1)
        #     compC[-1, :] = -c[::-1]
        #     g.append(norm_inf(compC))
        #     ng_norm += 1

    g_ = vertcat(*g)

    # 9. Bounds
    w_lb = -100 * DM.ones(n_opt)
    w_ub = +100 * DM.ones(n_opt)

    ng = g_.size1()
    g_lb = -1e-7 * DM.ones(ng, 1)
    g_ub = +1e-7 * DM.ones(ng, 1)

    if ng_norm > 0:
        g_ub[-ng_norm:] = stab_marg * DM.ones(ng_norm, 1)

    # 10. NLP formulation
    nlp = {'x': w_opt, 'f': f_obj, 'g': g_}

    # 11. Solver options
    sol_opts = {
        'ipopt.max_iter': max_iterations,
        'ipopt.print_level': 0,
        'ipopt.sb': 'yes',
        'print_time': 0
    }

    # 12. Create solver
    solver = nlpsol('solver', 'ipopt', nlp, sol_opts)

    # 13. Initial guess (coefficients=0, Yid=y)
    w_0 = DM.zeros(1, n_coeff)
    w_0 = hstack([w_0, y_scaled.reshape(1, -1)])

    # 14. Solve NLP
    sol = solver(lbx=w_lb, ubx=w_ub, x0=w_0, lbg=g_lb, ubg=g_ub)

    # 15. Extract solution
    x_opt = sol['x']
    y_id = x_opt[-N:].full()[:, 0]
    THETA = np.array(x_opt[:n_coeff])[:, 0]

    A_coeffs = THETA[0:na]
    C_coeffs = THETA[na : na+nc]

    # 16. Rescale output (no coefficient rescaling for ARMA)
    y_id = y_id * ystd

    # 17. Compute variance
    Vn = np.linalg.norm(y_id - y.flatten(), 2)**2 / (2*N)

    # 18. Build transfer functions
    G_tf, H_tf = create_transfer_functions_arma(
        A_coeffs, C_coeffs, na, nc, tsample
    )

    return A_coeffs, C_coeffs, y_id, Vn, G_tf, H_tf
```

### 6.3 Helper Function: Rescale

```python
def rescale(data):
    """
    Normalize data to unit standard deviation (zero mean not needed for ARMA).

    Args:
        data: Input array (shape agnostic)

    Returns:
        std: Standard deviation (for rescaling back)
        data_scaled: Normalized data
    """
    data_std = np.std(data)

    if data_std < 1e-10:
        # Constant signal, avoid division by zero
        return 1.0, data

    data_scaled = data / data_std
    return data_std, data_scaled
```

**Note:** Master branch rescale divides by std only, not by mean. Different from typical z-score normalization.

### 6.4 Transfer Function Creation

```python
def create_transfer_functions_arma(A_coeffs, C_coeffs, na, nc, tsample):
    """
    Create G(q) and H(q) transfer functions for ARMA.

    G(q) = 1 (no input dynamics)
    H(q) = C(q) / A(q)

    Args:
        A_coeffs, C_coeffs: Polynomial coefficients (arrays)
        na, nc: Polynomial orders
        tsample: Sampling time

    Returns:
        G_tf, H_tf: Harold Transfer objects
    """
    import harold

    # Build A polynomial
    A_poly = np.concatenate(([1.0], A_coeffs.flatten()))

    # Build C polynomial
    C_poly = np.concatenate(([1.0], C_coeffs.flatten()))

    # Create transfer functions
    G_tf = harold.Transfer([1.0], [1.0], dt=tsample)  # G = 1 for ARMA
    H_tf = harold.Transfer(C_poly, A_poly, dt=tsample)  # H = C(z)/A(z)

    return G_tf, H_tf
```

---

## 7. Comparison with ARARX Implementation

### 7.1 Structural Similarities

Both ARMA and ARARX use:
1. **CasADi symbolic construction** - Identical symbolic variable setup
2. **IPOPT solver** - Same NLP solver with same options
3. **Equality constraints** - Both enforce auxiliary variable consistency
4. **Data rescaling** - Both normalize data before optimization
5. **Companion matrix stability constraints** - Optional for both

### 7.2 Key Differences

| Aspect | ARMA | ARARX |
|--------|------|-------|
| **Coefficients** | a, c (na + nc) | a, b, d (na + nb + nd) |
| **Auxiliary Variables** | Yid only (N) | Yid, W, V (3·N) |
| **Regressor** | [-y, e] | [-y, u, -V] |
| **n_opt** | na + nc + N | na + nb + nd + 3·N |
| **Input handling** | None (time-series) | Input lags with delay |
| **Prediction errors** | Used in regressor (MA terms) | Not used (replaced by V) |
| **G(z)** | 1 | B(z) / (A(z)·D(z)) |
| **H(z)** | C(z) / A(z) | 1 / (A(z)·D(z)) |
| **Stability constraints** | Only A(z) | A(z) and D(z) |

### 7.3 Code Reuse Strategy

**Shared components (can reuse from ARARX):**
- `rescale()` function
- CasADi initialization boilerplate
- IPOPT solver configuration
- Companion matrix construction (for A polynomial)
- Solution extraction and transfer function creation

**ARMA-specific components (new code needed):**
- Regressor construction with prediction errors (Epsi)
- Symbolic Epsi update loop
- ARMA coefficient vector `[a, c]` instead of `[a, b, d]`
- G=1 transfer function (trivial)
- H=C/A transfer function (simpler than ARARX's H=1/(A·D))

**Estimated code reuse:** ~60% from ARARX, ~40% ARMA-specific

---

## 8. Implementation Challenges and Solutions

### 8.1 Challenge: Circular Dependency in Epsi

**Problem:** Prediction error `Epsi[k] = Y[k] - Yidw[k]` depends on Yidw, but Yid prediction uses past Epsi values.

**Solution:** CasADi's symbolic framework handles this automatically:
1. Epsi is a **symbolic expression**, not an optimization variable
2. The symbolic loop builds a computation graph: Yid[k] → Epsi[k] → Yid[k+1] → ...
3. IPOPT solves the entire system simultaneously (not sequentially)
4. The equality constraint `Yid = Yidw` ensures consistency

**Key insight:** This is why NLP is needed - iterative methods struggle with this circular dependency.

### 8.2 Challenge: MA Coefficient Identifiability

**Problem:** Moving average (MA) terms are notoriously difficult to estimate because:
- Prediction errors `e[k]` are latent (unobserved) variables
- Many parameter combinations can produce similar output
- MA models are non-convex (multiple local minima)

**Solution in master:**
1. **Good initial guess** (w_0 = [zeros, y]) provides reasonable starting point
2. **Tight constraints** (±1e-7) enforce consistency
3. **IPOPT's interior point method** handles non-convexity better than gradient descent
4. **Data rescaling** improves conditioning

**Expected accuracy:** 10-20% error on MA coefficients is acceptable (vs <5% for AR)

### 8.3 Challenge: Stability of C Polynomial

**Problem:** Master code only constrains A(z) stability, not C(z). Unstable C(z) can cause numerical issues.

**Solution options:**
1. **Follow master exactly** - Do not add C(z) constraint (for exact replication)
2. **Add C(z) constraint** - Improve robustness (recommended for production)

**Recommendation for harold branch:**
- Add C(z) stability constraint as **optional** (default OFF to match master)
- Document this enhancement in algorithm docstring
- Enable in high-noise scenarios

### 8.4 Challenge: Convergence with Low SNR

**Problem:** ARMA struggles when noise dominates signal (low SNR).

**Solution:**
1. **Increase max_iterations** (e.g., 500 instead of 200)
2. **Use stability constraints** (helps IPOPT avoid divergent solutions)
3. **Try multiple random initializations** (not in master, but could improve)

**Validation strategy:**
- Test with synthetic data at various SNR levels
- Document minimum SNR for reliable convergence
- Warn users when Vn is close to signal variance

---

## 9. Test Cases and Validation

### 9.1 Synthetic Data Tests

**Test Case 1: Pure AR(2)**
```python
# True model: A(z) = 1 + 0.5z⁻¹ - 0.3z⁻²
#             C(z) = 1 (no MA terms)
na = 2
nc = 0  # Pure AR
A_true = [0.5, -0.3]
C_true = []

# Generate data
N = 500
e = np.random.randn(N) * 0.1
y = np.zeros(N)
for k in range(2, N):
    y[k] = -0.5*y[k-1] + 0.3*y[k-2] + e[k]

# Identify
A_est, C_est, Yid, Vn, G_tf, H_tf = identify_arma_nlp(y, na, nc, 0, 1.0)

# Validate
assert np.allclose(A_est, A_true, rtol=0.05)  # 5% tolerance
```

**Test Case 2: Pure MA(1)**
```python
# True model: A(z) = 1
#             C(z) = 1 + 0.4z⁻¹
na = 0  # No AR terms
nc = 1
A_true = []
C_true = [0.4]

# Generate data
N = 500
e = np.random.randn(N) * 0.1
y = np.zeros(N)
for k in range(1, N):
    y[k] = e[k] + 0.4*e[k-1]

# Identify
A_est, C_est, Yid, Vn, G_tf, H_tf = identify_arma_nlp(y, na, nc, 0, 1.0)

# Validate
assert np.allclose(C_est, C_true, rtol=0.10)  # 10% tolerance (MA harder)
```

**Test Case 3: ARMA(2,2)**
```python
# True model: A(z) = 1 - 0.6z⁻¹ - 0.2z⁻²
#             C(z) = 1 + 0.4z⁻¹ + 0.1z⁻²
na = 2
nc = 2
A_true = [-0.6, -0.2]
C_true = [0.4, 0.1]

# Generate data
N = 1000  # Longer for ARMA(2,2)
e = np.random.randn(N) * 0.05
y = np.zeros(N)
for k in range(2, N):
    y[k] = 0.6*y[k-1] + 0.2*y[k-2] + e[k] + 0.4*e[k-1] + 0.1*e[k-2]

# Identify
A_est, C_est, Yid, Vn, G_tf, H_tf = identify_arma_nlp(y, na, nc, 0, 1.0)

# Validate
assert np.allclose(A_est, A_true, rtol=0.10)
assert np.allclose(C_est, C_true, rtol=0.15)  # MA terms harder
```

### 9.2 Cross-Validation with Master Branch

**Validation script:**
```python
import sys
sys.path.insert(0, '/Users/josephj/Workspace/SIPPY-master')
from sippy_unipi import system_identification

# Generate test data
np.random.seed(42)
N = 500
e = np.random.randn(N) * 0.1
y = np.zeros(N)
for k in range(2, N):
    y[k] = 0.5*y[k-1] - 0.3*y[k-2] + e[k] + 0.4*e[k-1]

# Master branch
u_dummy = np.zeros_like(y)
model_master = system_identification(
    y, u_dummy, 'ARMA',
    ARMA_orders=[2, 1, 0],
    tsample=1.0
)

# Harold branch
A_harold, C_harold, Yid_harold, Vn_harold, _, _ = identify_arma_nlp(
    y, na=2, nc=1, theta=0, tsample=1.0
)

# Compare
# Extract master coefficients (need to parse from model_master)
# ...
# assert np.allclose(A_harold, A_master, rtol=0.01)
# assert np.allclose(C_harold, C_master, rtol=0.01)
```

### 9.3 Accuracy Targets

**Coefficient estimation:**
- AR coefficients: < 5% relative error
- MA coefficients: < 15% relative error (inherently harder)

**Prediction accuracy:**
- NRMSE < 10% for well-conditioned data
- Correlation(Yid, y) > 0.95

**Transfer function:**
- Poles within 1% of master branch
- Zeros within 5% of master branch (MA terms)

**Stability:**
- If stab_cons=True, all poles inside unit circle scaled by stab_marg

---

## 10. Implementation Roadmap

### Phase 1: Core NLP Implementation (3-4 days)

**Tasks:**
1. Implement `rescale()` utility function (reuse from ARARX)
2. Set up CasADi symbolic variables (a, c, Yidw)
3. Implement regressor loop with Epsi symbolic updates
4. Add objective function and equality constraint (Yid = Yidw)
5. Configure IPOPT solver
6. Extract and rescale solution

**Deliverable:** Working ARMA NLP algorithm without stability constraints

### Phase 2: Stability Constraints (1 day)

**Tasks:**
1. Add companion matrix constraint for A(q) (reuse from ARARX)
2. (Optional) Add companion matrix constraint for C(q)
3. Test with unstable systems
4. Tune stab_marg parameter

**Deliverable:** ARMA with optional stability enforcement

### Phase 3: Integration and Testing (2 days)

**Tasks:**
1. Integrate into `ARMAAlgorithm.identify()` method
2. Add fallback to ILLS if CasADi unavailable
3. Write unit tests (AR, MA, ARMA test cases)
4. Cross-validate with master branch
5. Test on Examples data

**Deliverable:** Production-ready ARMA NLP implementation

### Phase 4: Documentation (1 day)

**Tasks:**
1. Update algorithm docstrings
2. Add usage examples
3. Document limitations (MA estimation challenges)
4. Update CLAUDE.md status

**Deliverable:** Complete, documented, tested ARMA NLP

**Total Estimated Effort:** 5-6 days

---

## 11. Code Location Reference

### 11.1 Master Branch Files

**Core optimization framework:**
- `/Users/josephj/Workspace/SIPPY-master/sippy_unipi/functionset_OPT.py`
  - Lines 10-280: `opt_id()` - NLP problem construction
  - Lines 94-95: ARMA coefficient vector `vertcat(a, c)`
  - Lines 155-156: ARMA regressor `vertcat(-vecY, vecE)`

**SISO implementation:**
- `/Users/josephj/Workspace/SIPPY-master/sippy_unipi/io_opt.py`
  - Lines 15-117: `GEN_id()` - SISO identification
  - Lines 81-82: ARMA transfer function G=1
  - Lines 100-105: ARMA transfer function H=C/A

**MIMO implementation:**
- `/Users/josephj/Workspace/SIPPY-master/sippy_unipi/io_optMIMO.py`
  - Lines 15-313: `GEN_MIMO_id()` - MIMO identification
  - Lines 146-147: ARMA transfer function G=1 for MIMO

**Entry point:**
- `/Users/josephj/Workspace/SIPPY-master/sippy_unipi/__init__.py`
  - Lines 639-685: ARMA order parsing (nb=0 forced)
  - Lines 893-941: ARMA identification path

**Utilities:**
- `/Users/josephj/Workspace/SIPPY-master/sippy_unipi/functionset.py`
  - Lines 123-126: `rescale()` function

### 11.2 Example Usage

**Example file:**
- `/Users/josephj/Workspace/SIPPY-master/Examples/Ex_OPT_GEN-INOUT.py`
  - Lines 205-207: ARMA identification example
  ```python
  Id_ARMA = system_identification(
      Ytot, Usim, 'ARMA',
      ARMA_orders=[na_ord, nc_ord, theta]
  )
  ```

---

## 12. Summary and Key Takeaways

### 12.1 Critical Insights

1. **ARMA uses NLP with CasADi/IPOPT** - NOT iterative least squares
2. **Only 1 auxiliary variable** - Yid (not 3 like ARARX: Yid, W, V)
3. **Prediction errors (Epsi) are symbolic** - Not optimization variables
4. **No input handling** - ARMA is time-series only (G=1)
5. **Circular dependency handled by CasADi** - Epsi depends on Yid, Yid depends on past Epsi
6. **Companion matrix constraint only for A(z)** - Not C(z) in master code
7. **MA estimation is challenging** - Expect 10-15% error even with NLP
8. **Data rescaling critical** - Only output, not input (since no inputs)

### 12.2 Expected Improvements

Reimplementing with NLP will:
- **Reduce error from 71-2600% to <10%** vs harold's current ILLS
- **Match master accuracy** (within 1-5% for AR, 10-15% for MA)
- **Enable stability guarantees** through optional constraints
- **Handle circular dependencies** automatically via symbolic framework

### 12.3 Differences from ARARX Implementation

- **Simpler problem structure** (na + nc + N vs na + nb + nd + 3N variables)
- **No W and V auxiliary variables** (condition `Nd != 0` is FALSE)
- **Prediction errors in regressor** (not used in ARARX)
- **G=1 transfer function** (trivial, no input dynamics)
- **H=C/A transfer function** (simpler than ARARX's H=1/(A·D))

### 12.4 Success Criteria

**Phase completion gates:**
- ✅ NLP solver runs without errors
- ✅ AR coefficients within 5% of master
- ✅ MA coefficients within 15% of master
- ✅ Yid correlation > 0.95 with measured output
- ✅ Transfer function poles match (< 1% difference)
- ✅ All unit tests pass

---

## 13. References

### 13.1 Mathematical Background

1. **ARMA Models:**
   - Box, G. E. P., & Jenkins, G. M. (1976). *Time Series Analysis: Forecasting and Control*. Holden-Day.
   - Chapter 3: Autoregressive-Moving-Average Processes

2. **Prediction Error Methods (PEM):**
   - Ljung, L. (1999). *System Identification: Theory for the User*. Prentice Hall.
   - Chapter 7: Prediction error identification methods

3. **MA Parameter Estimation:**
   - Söderström, T., & Stoica, P. (1989). *System Identification*. Prentice Hall.
   - Section 4.5: Moving average model estimation

### 13.2 Software Documentation

1. **CasADi:**
   - Documentation: https://web.casadi.org/docs/
   - User guide: https://web.casadi.org/docs/#user-guide
   - Python API: https://web.casadi.org/python-api/

2. **IPOPT:**
   - Documentation: https://coin-or.github.io/Ipopt/
   - Options reference: https://coin-or.github.io/Ipopt/OPTIONS.html

3. **Harold:**
   - Documentation: https://harold.readthedocs.io/
   - Function reference: https://harold.readthedocs.io/function_reference.html

---

**END OF DOCUMENT**

This comprehensive analysis provides all the information needed to implement an accurate NLP-based ARMA algorithm matching the master branch implementation. The key insight is that ARMA is a simplified version of the general NLP framework with no input dynamics (G=1) and prediction errors in the regressor instead of auxiliary variables.
