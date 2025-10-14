# ARARX NLP Master Branch Analysis

**Date:** 2025-10-13
**Objective:** Comprehensive analysis of the master branch ARARX NLP implementation using CasADi/IPOPT for accurate reimplementation in harold branch

---

## Executive Summary

The master branch ARARX implementation uses **CasADi-based nonlinear programming (NLP)** with the **IPOPT solver** to estimate model parameters through constrained optimization. This approach differs fundamentally from the harold branch's iterative auxiliary variable method.

**Key Findings:**
- Optimization library: **CasADi + IPOPT** (Interior Point Optimizer)
- Cost function: Minimizes squared prediction error between Y and Yid
- Decision variables: Model coefficients (a, b, d) + auxiliary time series (Yid, W, V)
- Constraints: Equality constraints for auxiliary variables + optional stability constraints
- ARARX uses 3 auxiliary variable sequences: **Yid** (predicted output), **W** (B/D filtered input), **V** (A*y - W residual)

**Expected Accuracy Target:** < 1e-4 relative error vs master branch

---

## 1. Algorithm Overview

### 1.1 ARARX Model Structure

The ARARX model is defined as:
```
A(q) y(k) = B(q)/D(q) * u(k-θ) + e(k)
```

Where:
- **A(q)** = 1 + a₁q⁻¹ + ... + aₙₐq⁻ⁿᵃ (output auto-regressive polynomial, na coefficients)
- **B(q)** = b₀ + b₁q⁻¹ + ... + bₙᵦq⁻ⁿᵇ (input numerator polynomial, nb coefficients)
- **D(q)** = 1 + d₁q⁻¹ + ... + dₙ_dq⁻ⁿᵈ (denominator polynomial for input path, nd coefficients)
- **θ** = input delay (time steps)
- **e(k)** = white noise

Transfer functions:
- **G(q) = B(q) / (A(q) * D(q))** (deterministic input-output)
- **H(q) = 1 / A(q)** (noise transfer function)

### 1.2 Key Differences from Harold Branch

| Aspect | Master Branch (NLP) | Harold Branch (Current) |
|--------|-------------------|------------------------|
| **Method** | Single-shot NLP with IPOPT | Iterative auxiliary variable method |
| **Solver** | CasADi symbolic + IPOPT | NumPy least squares |
| **Variables** | Coefficients + full time series | Coefficients only |
| **Constraints** | Explicit equality constraints | Implicit through iterations |
| **Stability** | Optional companion matrix norm constraints | No stability enforcement |
| **Convergence** | IPOPT interior point method | Manual iteration with tolerance check |
| **Accuracy** | Reference implementation | ~100% error vs master |

---

## 2. Optimization Problem Formulation

### 2.1 Decision Variables

The optimization vector `w_opt` has **n_opt** elements:

```python
n_opt = n_coeff + 3 * N  # For ARARX (nd != 0)

where:
  n_coeff = na + nb + nd  # Number of polynomial coefficients
  N = len(y)              # Number of time samples
```

**Structure of w_opt:**
```
w_opt = [a₁, a₂, ..., aₙₐ,           # A polynomial coefficients (na elements)
         b₀, b₁, ..., bₙᵦ,           # B polynomial coefficients (nb elements)
         d₁, d₂, ..., dₙ_d,          # D polynomial coefficients (nd elements)
         W[0], W[1], ..., W[N-1],    # Auxiliary variable W (N elements)
         V[0], V[1], ..., V[N-1],    # Auxiliary variable V (N elements)
         Yid[0], Yid[1], ..., Yid[N-1]]  # Predicted output (N elements)
```

**Coefficient extraction** (from `functionset_OPT.py`, lines 52-65):
```python
# Symbolic variables
a = w_opt[0:na]                              # A coefficients
b = w_opt[na : na + nb]                      # B coefficients
d = w_opt[na + nb : na + nb + nd]            # D coefficients

# Auxiliary time series
Ww = w_opt[-3*N : -2*N]                      # W auxiliary variable
Vw = w_opt[-2*N : -N]                        # V auxiliary variable
Yidw = w_opt[-N:]                             # Predicted output
```

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

This is the **standard prediction error method (PEM)** objective.

### 2.3 Constraints

The NLP includes **two types of constraints**:

#### 2.3.1 Equality Constraints (Lines 197-202)

These enforce consistency between the auxiliary variables and the model structure:

```python
g = []
g.append(Yid - Yidw)    # Predicted output must match optimization variable
g.append(W - Ww)         # W auxiliary variable consistency
g.append(V - Vw)         # V auxiliary variable consistency
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
- **D(q) companion matrix** (if nd > 0, lines 235-247)

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

### 3.1 Auxiliary Variable W (Lines 104-107, 172-178)

**Purpose:** Represents the filtered input `W = B/D * u`

**Recursive definition:**
```python
if nf != 0:
    # For BJ/GEN models (not ARARX)
    W[k] = mtimes(vertcat(vecU, -vecW).T, vertcat(b, f))
else:
    # For ARARX (no F polynomial)
    W[k] = mtimes(vecU.T, b)
```

For ARARX (nf=0), line 176:
```python
phiw = vertcat(vecU)     # Only input lags
W[k] = mtimes(phiw.T, coeff_w)
```

**Mathematical form:**
```
W[k] = Σ[i=0 to nb-1] b[i] * u[k - θ - i]
```

This is the **numerator response** to the input.

### 3.2 Auxiliary Variable V (Lines 104-107, 179-184)

**Purpose:** Represents the corrected output `V = A*y - W`

**Recursive definition:**
```python
if na == 0:  # For BJ models (A(z) = 1)
    V[k] = Y[k] - Ww[k]
else:  # For ARARX (A(z) ≠ 1)
    phiv = vertcat(vecY)
    V[k] = Y[k] + mtimes(phiv.T, coeff_v) - Ww[k]
```

For ARARX (na > 0), line 184:
```python
V[k] = Y[k] + mtimes(vecY.T, a) - Ww[k]
```

**Mathematical form:**
```
V[k] = y[k] + Σ[i=1 to na] a[i] * y[k-i] - W[k]
```

This is the **AR-corrected residual** after removing the input effect.

### 3.3 Predicted Output Yid (Lines 118-165)

**Purpose:** One-step-ahead prediction using the model structure

**Regressor construction** (lines 121-162):

For ARARX (line 158):
```python
phi = vertcat(-vecY, vecU, -vecV)
```

Where:
- `vecY = Y[k-na : k][::-1]` - Output lags (na elements)
- `vecU = U[k-nb-θ : k-θ][::-1]` - Input lags (nb elements)
- `vecV = Vw[k-nd : k][::-1]` - V auxiliary variable lags (nd elements)

**Prediction** (line 165):
```python
coeff = vertcat(a, b, d)  # For ARARX
Yid[k] = mtimes(phi.T, coeff)
```

**Mathematical form:**
```
Yid[k] = - Σ[i=1 to na] a[i] * y[k-i]
         + Σ[i=0 to nb-1] b[i] * u[k-θ-i]
         - Σ[i=1 to nd] d[i] * V[k-i]
```

---

## 4. Algorithm Step-by-Step

### 4.1 Initialization (Lines 31-63, io_opt.py and io_optMIMO.py)

**Data preprocessing:**
```python
ystd, y = rescale(y)       # Normalize output to zero mean, unit std
Ustd, u = rescale(u)       # Normalize each input
```

**Determine problem dimensions:**
```python
val = max(nb + theta, na, nc, nd, nf)  # Max lag
m = number of inputs (udim)
p = number of outputs (1 for ARARX SISO)
n_coeff = na + nb + nd     # For ARARX
n_opt = n_coeff + 3*N      # Optimization variables
```

**Initial guess** (lines 51-60, io_opt.py):
```python
w_0 = zeros(1, n_coeff)    # Start with zero coefficients
w_0 = hstack([w_0, y])     # Initialize Yid = y (measured output)
w_0 = hstack([w_0, y, y])  # Initialize W = y, V = y (for ARARX)
```

This provides a **warm start** for IPOPT.

### 4.2 Build Regressor Loop (Lines 118-185)

**For each time step k ≥ val:**

1. **Build input regressor** (lines 122-129):
   ```python
   vecU = U[k-nb-theta : k-theta][::-1]
   ```

2. **Build output regressor** (lines 132-133):
   ```python
   vecY = Y[k-na : k][::-1]
   ```

3. **Build V auxiliary regressor** (lines 136-138):
   ```python
   vecV = Vw[k-nd : k][::-1]
   ```

4. **Construct ARARX regressor** (line 158):
   ```python
   phi = vertcat(-vecY, vecU, -vecV)
   ```

5. **Update prediction** (line 165):
   ```python
   Yid[k] = mtimes(phi.T, coeff)
   ```

6. **Update W auxiliary variable** (line 177):
   ```python
   W[k] = mtimes(vecU.T, b)
   ```

7. **Update V auxiliary variable** (line 184):
   ```python
   V[k] = Y[k] + mtimes(vecY.T, a) - Ww[k]
   ```

**Important:** The predictions use:
- **Measured outputs y** (not predicted Yid) for output lags
- **Auxiliary variables Ww, Vw** (optimization variables) for consistency
- The constraint `Yid = Yidw` enforces that the symbolically computed Yid matches the optimization variable

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

### 4.4 Extract Solution (Lines 65-116, io_opt.py)

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
# For ARARX:
A_coeffs = THETA[0:na]
B_coeffs = THETA[na : na+nb]
D_coeffs = THETA[na+nb : na+nb+nd]
```

**Build transfer functions** (lines 75-115):
```python
# Build polynomials
A_poly = tf([1, zeros(na)], [1, A_coeffs], 1)
F_poly = tf([1, zeros(nf)], [1, F_coeffs], 1)  # F = 1 for ARARX (nf=0)
D_poly = tf([1, zeros(nd)], [1, D_coeffs], 1)

# Multiply polynomials for denominators
DEN_G = A_poly * F_poly  # For ARARX: A(q) only (since F=1)
DEN_H = A_poly * D_poly

# Build numerator with delay
NUM_G = zeros(valG)
NUM_G[theta : theta+nb] = B_coeffs

# Transfer functions
G = tf(NUM_G, DEN_G, tsample)
H = tf(NUM_H, DEN_H, tsample)
```

**Rescale coefficients** (lines 152-160, io_optMIMO.py):
```python
B_coeffs = B_coeffs * ystd / Ustd  # Restore original scaling
y_id = y_id * ystd                # Restore output scaling
```

---

## 5. Key Implementation Details

### 5.1 CasADi Symbolic Construction

**Purpose:** CasADi constructs a **symbolic computation graph** for automatic differentiation.

**Variable declaration** (line 49):
```python
w_opt = SX.sym('w', n_opt)  # Symbolic optimization variables
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

### 5.2 Data Preprocessing

**Rescaling function** (from `functionset.py`, imported at line 11):
```python
def rescale(data):
    data_mean = mean(data)
    data_std = std(data)
    data_scaled = (data - data_mean) / data_std
    return data_std, data_scaled
```

**Why rescale?**
- Improves **numerical conditioning** (all variables ~O(1))
- Prevents ill-conditioning when inputs/outputs have different scales
- Standard practice in NLP

**Rescaling applied:**
- Output: `ystd, y = rescale(y)`
- Each input: `Ustd[i], u[i] = rescale(u[i])`

**Coefficient rescaling at end:**
```python
B_coeffs = B_coeffs * ystd / Ustd
```

This restores the original units.

### 5.3 Initial Conditions Handling

**Warm start strategy** (lines 51-60, io_opt.py):

```python
w_0 = zeros(1, n_coeff)       # Coefficients start at zero
w_0 = hstack([w_0, y])        # Yid starts as measured output
w_0 = hstack([w_0, y, y])     # W and V start as measured output
```

**Why this works:**
- If all coefficients are zero, prediction Yid ≈ y is a reasonable initial guess
- IPOPT can quickly adjust from this starting point
- Avoids local minima from random initialization

**Time-stepping in regressor:**
- Lags are handled naturally by indexing `Y[k-na:k]`, `U[k-nb-theta:k-theta]`
- Initial samples `k < val` are **not identifiable** (insufficient lag data)
- Loop starts at `k = val` (line 118)

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

---

## 6. Implementation Specification for Harold Branch

### 6.1 Required Dependencies

```python
import numpy as np
from casadi import DM, SX, vertcat, mtimes, norm_inf, nlpsol
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

### 6.2 Pseudocode for ARARX NLP

```python
def identify_ararx_nlp(y, u, na, nb, nd, theta, tsample, max_iterations=200,
                       stab_marg=1.0, stab_cons=False):
    """
    ARARX identification using NLP (CasADi + IPOPT).

    Args:
        y: Output array (ny x N)
        u: Input array (nu x N)
        na: A polynomial order
        nb: B polynomial order
        nd: D polynomial order
        theta: Input delay
        tsample: Sampling time
        max_iterations: IPOPT max iterations
        stab_marg: Stability margin (< 1.0)
        stab_cons: Enable stability constraints

    Returns:
        A_coeffs, B_coeffs, D_coeffs, Yid, Vn, G_tf, H_tf
    """

    # 1. Preprocess data
    ystd, y_scaled = rescale(y)
    ustd, u_scaled = rescale(u)
    N = y.shape[1]

    # 2. Problem dimensions
    n_coeff = na + nb + nd
    n_opt = n_coeff + 3 * N
    val = max(nb + theta, na, nd)

    # 3. Define symbolic optimization variables
    w_opt = SX.sym('w', n_opt)
    a = w_opt[0:na]
    b = w_opt[na : na+nb]
    d = w_opt[na+nb : na+nb+nd]
    Ww = w_opt[-3*N : -2*N]
    Vw = w_opt[-2*N : -N]
    Yidw = w_opt[-N:]

    # 4. Initialize auxiliary variables
    Yid = y_scaled * SX.ones(1)  # Symbolic copy
    W = y_scaled * SX.ones(1)
    V = y_scaled * SX.ones(1)

    # 5. Build regressor loop
    coeff = vertcat(a, b, d)
    coeff_w = b
    coeff_v = a

    for k in range(N):
        if k >= val:
            # Input lags
            vecU = u_scaled[:, k-nb-theta : k-theta][::-1]

            # Output lags
            vecY = y_scaled[k-na : k][::-1]

            # V auxiliary lags
            vecV = Vw[k-nd : k][::-1]

            # W auxiliary lags (if needed)
            vecW = Ww[k-1 : max(k-1, 0)][::-1] if k > 0 else []

            # ARARX regressor: [-y lags, u lags, -V lags]
            phi = vertcat(-vecY, vecU, -vecV)

            # Prediction
            Yid[k] = mtimes(phi.T, coeff)

            # W auxiliary (B * u)
            W[k] = mtimes(vecU.T, coeff_w)

            # V auxiliary (A*y - W)
            V[k] = y_scaled[k] + mtimes(vecY.T, coeff_v) - Ww[k]

    # 6. Objective function
    DY = y_scaled - Yidw
    f_obj = (1.0 / N) * mtimes(DY.T, DY)

    # 7. Constraints
    g = []
    g.append(Yid - Yidw)    # Prediction consistency
    g.append(W - Ww)         # W consistency
    g.append(V - Vw)         # V consistency

    # 8. Stability constraints (optional)
    ng_norm = 0
    if stab_cons:
        if na > 0:
            compA = SX.zeros(na, na)
            compA[:-1, 1:] = SX.eye(na-1)
            compA[-1, :] = -a[::-1]
            g.append(norm_inf(compA))
            ng_norm += 1

        if nd > 0:
            compD = SX.zeros(nd, nd)
            compD[:-1, 1:] = SX.eye(nd-1)
            compD[-1, :] = -d[::-1]
            g.append(norm_inf(compD))
            ng_norm += 1

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

    # 13. Initial guess
    w_0 = DM.zeros(1, n_coeff)
    w_0 = hstack([w_0, y_scaled.reshape(1, -1)])
    w_0 = hstack([w_0, y_scaled.reshape(1, -1), y_scaled.reshape(1, -1)])

    # 14. Solve NLP
    sol = solver(lbx=w_lb, ubx=w_ub, x0=w_0, lbg=g_lb, ubg=g_ub)

    # 15. Extract solution
    x_opt = sol['x']
    y_id = x_opt[-N:].full()[:, 0]
    THETA = np.array(x_opt[:n_coeff])[:, 0]

    A_coeffs = THETA[0:na]
    B_coeffs = THETA[na : na+nb]
    D_coeffs = THETA[na+nb : na+nb+nd]

    # 16. Rescale coefficients
    B_coeffs = B_coeffs * ystd / ustd
    y_id = y_id * ystd

    # 17. Compute variance
    Vn = np.linalg.norm(y_id - y.flatten(), 2)**2 / (2*N)

    # 18. Build transfer functions
    G_tf, H_tf = create_transfer_functions_ararx(
        A_coeffs, B_coeffs, D_coeffs, na, nb, nd, theta, tsample
    )

    return A_coeffs, B_coeffs, D_coeffs, y_id, Vn, G_tf, H_tf
```

### 6.3 Helper Function: Rescale

```python
def rescale(data):
    """
    Normalize data to zero mean and unit standard deviation.

    Args:
        data: Input array (shape agnostic)

    Returns:
        std: Standard deviation (for rescaling back)
        data_scaled: Normalized data
    """
    data_mean = np.mean(data)
    data_std = np.std(data)

    if data_std < 1e-10:
        # Constant signal, avoid division by zero
        return 1.0, data - data_mean

    data_scaled = (data - data_mean) / data_std
    return data_std, data_scaled
```

### 6.4 Transfer Function Creation

```python
def create_transfer_functions_ararx(A_coeffs, B_coeffs, D_coeffs,
                                     na, nb, nd, theta, tsample):
    """
    Create G(q) and H(q) transfer functions for ARARX.

    G(q) = B(q) / (A(q) * D(q))
    H(q) = 1 / (A(q) * D(q))

    Args:
        A_coeffs, B_coeffs, D_coeffs: Polynomial coefficients (arrays)
        na, nb, nd: Polynomial orders
        theta: Input delay
        tsample: Sampling time

    Returns:
        G_tf, H_tf: Harold Transfer objects
    """
    import harold

    # Build A polynomial
    A_poly = np.concatenate(([1.0], A_coeffs.flatten()))

    # Build D polynomial
    D_poly = np.concatenate(([1.0], D_coeffs.flatten()))

    # Build B polynomial with delay
    B_poly = np.concatenate((B_coeffs.flatten(), np.zeros(theta)))

    # Multiply A * D for denominator
    DEN_G = harold.haroldpolymul(A_poly, D_poly)

    # Create transfer functions
    G_tf = harold.Transfer(B_poly, DEN_G, dt=tsample)
    H_tf = harold.Transfer([1.0], DEN_G, dt=tsample)

    return G_tf, H_tf
```

---

## 7. Test Cases from Master Branch

### 7.1 Example: Ex_OPT_GEN-INOUT.py (Lines 209-215)

**System:**
```python
na_ord = [2]
nb_ord = [[3]]
nd_ord = [3]
theta = [[11]]
max_iterations = 300
```

**Usage:**
```python
Id_ARARX = system_identification(
    Ytot, Usim, 'ARARX',
    ARARX_orders=[na_ord, nb_ord, nd_ord, theta],
    max_iterations=300
)
```

**Expected outputs:**
- `Id_ARARX.G` - Transfer function G(q)
- `Id_ARARX.H` - Transfer function H(q)
- `Id_ARARX.Yid` - Predicted output
- `Id_ARARX.Vn` - Variance of prediction error

### 7.2 Accuracy Targets

From master branch, typical performance:
- Prediction error: `Vn < 0.01` for well-conditioned data
- Coefficient convergence: Relative change < 1e-8
- Transfer function poles: Inside unit circle (if `stab_cons=True`)

**For harold branch reimplementation:**
- Target relative error: **< 1e-4** vs master for same inputs
- Yid correlation: **> 0.9999** with master Yid
- Coefficient error: **< 0.01** absolute for normalized coefficients

---

## 8. Implementation Roadmap

### Phase 1: Core NLP Implementation (Week 1)

**Tasks:**
1. Implement `rescale()` utility function
2. Set up CasADi symbolic variables and indexing
3. Implement regressor loop with ARARX structure
4. Add objective function and equality constraints
5. Configure IPOPT solver with correct options
6. Extract and rescale solution

**Deliverable:** Working ARARX NLP algorithm without stability constraints

### Phase 2: Stability Constraints (Week 2)

**Tasks:**
1. Implement companion matrix construction for A(q)
2. Implement companion matrix construction for D(q)
3. Add infinity-norm constraints
4. Test with unstable systems (verify poles stay inside unit circle)
5. Tune `stab_marg` parameter for robustness

**Deliverable:** ARARX with optional stability enforcement

### Phase 3: Integration and Testing (Week 3)

**Tasks:**
1. Integrate NLP solver into `ARARXAlgorithm.identify()` method
2. Add fallback to auxiliary variable method if CasADi unavailable
3. Write unit tests comparing with master branch
4. Test on Examples/Ex_OPT_GEN-INOUT.py data
5. Validate numerical accuracy (< 1e-4 error target)

**Deliverable:** Production-ready ARARX NLP implementation

### Phase 4: Documentation and Validation (Week 4)

**Tasks:**
1. Document NLP approach in algorithm docstrings
2. Add usage examples to Examples/ folder
3. Update MIGRATION_ACCURACY_TODO.md (mark TASK 12 complete)
4. Run full test suite and ensure 100% pass rate
5. Performance profiling (NLP vs auxiliary variable method)

**Deliverable:** Complete, documented, tested ARARX NLP

---

## 9. Summary and Next Steps

### 9.1 Key Takeaways

1. **Master uses CasADi/IPOPT NLP** - Not iterative least squares
2. **3 auxiliary variables** - Yid, W, V (not just V and W)
3. **Equality constraints** enforce auxiliary variable consistency
4. **Stability constraints** are optional (companion matrix norm bounds)
5. **Data rescaling** is critical for numerical conditioning
6. **Warm start** from measured output improves convergence

### 9.2 Expected Improvements

Reimplementing with NLP will:
- **Eliminate 100% error** vs master branch
- **Improve stability** through optional constraints
- **Match master accuracy** (< 1e-4 relative error)
- **Enable stability guarantees** for control applications

### 9.3 Success Criteria

**Phase completion gates:**
- ✅ NLP solver runs without errors
- ✅ Coefficients within 1% of master branch
- ✅ Yid correlation > 0.9999 with master
- ✅ Transfer function poles match (< 0.01 difference)
- ✅ All unit tests pass (100% pass rate)

---

## 10. References

### 10.1 Mathematical Background

1. **Prediction Error Methods (PEM):**
   - Ljung, L. (1999). *System Identification: Theory for the User*. Prentice Hall.
   - Chapter 7: Prediction error identification methods

2. **ARARX Models:**
   - Söderström, T., & Stoica, P. (1989). *System Identification*. Prentice Hall.
   - Section 6.3: ARARX and ARARMAX structures

3. **Nonlinear Programming:**
   - Nocedal, J., & Wright, S. J. (2006). *Numerical Optimization*. Springer.
   - Chapter 19: Interior point methods

### 10.2 Software Documentation

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

### 10.3 Master Branch Files

Key files for reference:
- `/Users/josephj/Workspace/SIPPY-master/sippy_unipi/functionset_OPT.py` (lines 10-280)
- `/Users/josephj/Workspace/SIPPY-master/sippy_unipi/io_opt.py` (lines 15-117)
- `/Users/josephj/Workspace/SIPPY-master/sippy_unipi/io_optMIMO.py` (lines 15-173)

---

**END OF DOCUMENT**

This comprehensive analysis provides all the information needed to implement an accurate NLP-based ARARX algorithm matching the master branch implementation.
