# ARMA Master Branch Investigation Report

**Date:** 2025-10-13
**Investigator:** Claude Code
**Branch:** master (via git worktree at /Users/josephj/Workspace/SIPPY-master)
**Purpose:** Determine ARMA implementation in master branch and validation strategy for harold branch

---

## Executive Summary

**ARMA Support Status:** ✅ **FULLY SUPPORTED** in master branch

The master branch implements ARMA (AutoRegressive Moving Average) as a **special case of ARMAX** with `nb=0` (no input coefficients). ARMA is implemented as a pure time-series identification method (output-only, no inputs) using optimization-based techniques.

**Key Finding:** ARMA in master branch is **NOT** validated by calling ARMAX with `nu=0`. Instead, it's a **distinct identification pathway** that sets `nb=0` internally within the generalized optimization framework.

---

## 1. ARMA Implementation Architecture

### 1.1 Entry Point: `system_identification()`

**File:** `/Users/josephj/Workspace/SIPPY-master/sippy_unipi/__init__.py`

**Lines 639-685:** ARMA order parsing logic

```python
# ARMA
if id_method == "ARMA":
    # not 3 inputs
    if len(ARMA_orders) != 3:
        sys.exit(
            "Error! ARMA identification takes three arguments in ARMA_orders"
        )

    # assigned orders
    if (
        isinstance(ARMA_orders[0], list)
        and isinstance(ARMA_orders[1], list)
        and isinstance(ARMA_orders[2], list)
    ):
        na = ARMA_orders[0]
        nb = np.zeros((ydim, udim), dtype=int).tolist()  # ← KEY: nb forced to 0
        nc = ARMA_orders[1]
        theta = ARMA_orders[2]

    # not assigned orders (read default)
    elif (
        isinstance(ARMA_orders[0], int)
        and isinstance(ARMA_orders[1], int)
        and isinstance(ARMA_orders[2], int)
    ):
        na = (
            ARMA_orders[0] * np.ones((ydim,), dtype=int)
        ).tolist()
        nb = np.zeros((ydim, udim), dtype=int).tolist()  # ← KEY: nb forced to 0
        nc = (
            ARMA_orders[1] * np.ones((ydim,), dtype=int)
        ).tolist()
        theta = (
            ARMA_orders[2] * np.ones((ydim, udim), dtype=int)
        ).tolist()
```

**Key Insight:**
- ARMA_orders takes **3 parameters**: `[na, nc, theta]` (NOT 4 like ARMAX)
- `nb` is **internally set to zero-filled matrix** of shape `(ydim, udim)`
- ARMA flows through the same optimization path as ARMAX, ARARX, BJ, and GEN

### 1.2 Identification Path

**Lines 893-941:** ARMA uses the generalized optimization framework

```python
elif (
    id_method == "ARMA"
    or id_method == "ARARX"
    or id_method == "ARARMAX"
    or id_method == "GEN"
    or id_method == "BJ"
    or id_method == "OE"
):
    # ... order setup ...

    # id MODEL: ARMA, ARARX, ARARMAX, BJ, GEN
    from . import io_optMIMO

    (
        DENOMINATOR,
        NUMERATOR,
        DENOMINATOR_H,
        NUMERATOR_H,
        G,
        H,
        Vn_tot,
        Yid,
    ) = io_optMIMO.GEN_MIMO_id(
        id_method,
        y,
        u,
        na,
        nb,  # ← nb=0 for ARMA
        nc,
        nd,
        nf,
        theta,
        tsample,
        max_iterations,
        stab_marg,
        stab_cons,
    )
```

---

## 2. Core ARMA Algorithm Implementation

### 2.1 Optimization Module: `io_optMIMO.py`

**File:** `/Users/josephj/Workspace/SIPPY-master/sippy_unipi/io_optMIMO.py`

**Lines 146-163:** ARMA-specific transfer function logic

```python
if id_method == "ARMA":
    NUM = np.ones((udim, 1))  # ← G(z) = 1 (no input dynamics)
else:
    NUM = np.zeros((udim, valG))
#
for k in range(udim):
    if id_method != "ARMA":
        THETA[na + np.sum(nb[0:k]) : na + np.sum(nb[0 : k + 1])] = (
            THETA[na + np.sum(nb[0:k]) : na + np.sum(nb[0 : k + 1])]
            * ystd
            / Ustd[k]
        )
        NUM[k, theta[k] : theta[k] + nb[k]] = THETA[
            na + np.sum(nb[0:k]) : na + np.sum(nb[0 : k + 1])
        ]
    DEN[k, 0 : na + nf + 1] = denG
```

**SISO Version:** `/Users/josephj/Workspace/SIPPY-master/sippy_unipi/io_opt.py`

**Lines 81-82:**

```python
if id_method == "ARMA":
    NUM = 1.0  # ← G(z) = 1
else:
    NUM = np.zeros(valG)
    NUM[theta : nb + theta] = THETA[na : nb + na]
```

### 2.2 Optimization Problem Setup: `functionset_OPT.py`

**File:** `/Users/josephj/Workspace/SIPPY-master/sippy_unipi/functionset_OPT.py`

**Lines 87-94:** ARMA coefficient vector

```python
# Building coefficient vector
if FLAG == "OE":
    coeff = vertcat(b, f)
elif FLAG == "BJ":
    coeff = vertcat(b, f, c, d)
elif FLAG == "ARMAX":
    coeff = vertcat(a, b, c)
elif FLAG == "ARARX":
    coeff = vertcat(a, b, d)
elif FLAG == "ARARMAX":
    coeff = vertcat(a, b, c, d)
elif FLAG == "ARMA":
    coeff = vertcat(a, c)  # ← Only AR (a) and MA (c) terms
else:  # GEN
    coeff = vertcat(a, b, f, c, d)
```

**Lines 154-155:** ARMA regressor

```python
elif FLAG == "ARMAX":
    phi = vertcat(-vecY, vecU, vecE)
elif FLAG == "ARMA":
    phi = vertcat(-vecY, vecE)  # ← Output-only regressor
elif FLAG == "ARARX":
    phi = vertcat(-vecY, vecU, -vecV)
```

---

## 3. ARMA Mathematical Model

### 3.1 Transfer Function Structure

```
ARMA Model:
    y(t) = C(z)/A(z) * e(t)

Where:
    A(z) = 1 + a₁z⁻¹ + a₂z⁻² + ... + aₙₐz⁻ⁿᵃ  (AR polynomial)
    C(z) = 1 + c₁z⁻¹ + c₂z⁻² + ... + cₙcz⁻ⁿᶜ  (MA polynomial)

    G(z) = 1  (no input dynamics)
    H(z) = C(z)/A(z)  (noise dynamics)
```

### 3.2 Difference Equation

```
A(z)·y(t) = C(z)·e(t)

Expanded:
y(t) = -a₁·y(t-1) - a₂·y(t-2) - ... - aₙₐ·y(t-na)
       + e(t) + c₁·e(t-1) + c₂·e(t-2) + ... + cₙc·e(t-nc)
```

### 3.3 Regressor Construction

```python
# At time k >= max(na, nc):
vecY = Y[k - na : k][::-1]           # Past outputs (reversed)
vecE = Epsi[k - nc : k][::-1]        # Past prediction errors (reversed)
phi = vertcat(-vecY, vecE)            # Regressor
coeff = vertcat(a, c)                 # Coefficients [a₁,...,aₙₐ, c₁,...,cₙc]

# One-step ahead prediction:
Yid[k] = mtimes(phi.T, coeff)
       = -a₁·y(k-1) - ... - aₙₐ·y(k-na) + c₁·e(k-1) + ... + cₙc·e(k-nc)
```

---

## 4. ARMA vs ARMAX Comparison

| Aspect | ARMA | ARMAX |
|--------|------|-------|
| **Parameters** | `ARMA_orders=[na, nc, theta]` | `ARMAX_orders=[na, nb, nc, theta]` |
| **Input Dynamics** | None (G=1) | Yes (B/A) |
| **nb Order** | Forced to 0 | User-specified |
| **Regressor** | `phi = [-y(t-1:t-na), e(t-1:t-nc)]` | `phi = [-y(t-1:t-na), u(t-θ:t-θ-nb), e(t-1:t-nc)]` |
| **Coefficients** | `[a, c]` | `[a, b, c]` |
| **Use Case** | Time-series (output-only) | Input-output systems |
| **Transfer Functions** | G=1, H=C/A | G=B/A, H=C/A |
| **Validation** | Uses `validation()` with G=1 | Uses `validation()` with G=B/A |

**Critical Distinction:** ARMA is **NOT** "ARMAX with nb=0" from a user perspective. The API is different:
- ARMA: 3 parameters (na, nc, theta)
- ARMAX: 4 parameters (na, nb, nc, theta)

However, internally, both use `GEN_MIMO_id()` with ARMA having `nb=[[0]]`.

---

## 5. Information Criterion (IC) Mode

**Lines 1458-1461:** IC mode for ARMA

```python
# ARMA
if id_method == "ARMA":
    nb_ord = [1, 1]      # ← nb range forced to [1,1] (effectively nb=1)
    nd_ord = [0, 0]
    nf_ord = [0, 0]
```

**Note:** In IC mode, ARMA uses `nb_ord=[1,1]` which seems contradictory. This likely means the algorithm iterates with `nb=1` but during identification, the input contribution is ignored (G=1).

---

## 6. Example Usage

**File:** `/Users/josephj/Workspace/SIPPY-master/Examples/Ex_OPT_GEN-INOUT.py`

**Lines 205-207:** ARMA identification example

```python
# ARMA - ARARX - ARARMAX
Id_ARMA = system_identification(
    Ytot, Usim, "ARMA", ARMA_orders=[na_ord, nc_ord, theta]
)
```

**Lines 57-63:** Using ARMA model

```python
Y_arma = Id_ARMA.Yid.T
Y_ararx = Id_ARARX.Yid.T
Y_ararmax = Id_ARARMAX.Yid.T
Y_oe = Id_OE.Yid.T
Y_bj = Id_BJ.Yid.T
Y_gen = Id_GEN.Yid.T
```

**Lines 111-112:** ARMA validation (commented out in example)

```python
# ARMA - ARARX - ARARMAX
# Yv_arma = fset.validation(Id_ARMA,U_valid,Ytotvalid,Time)
```

**Note:** The example comments out ARMA validation, likely because ARMA models output-only dynamics and may not behave well with validation on new input sequences.

---

## 7. Model Structure

**File:** `/Users/josephj/Workspace/SIPPY-master/sippy_unipi/io_optMIMO.py`

**Lines 316-350:** `GEN_MIMO_model` class

```python
class GEN_MIMO_model:
    def __init__(
        self,
        na,
        nb,
        nc,
        nd,
        nf,
        theta,
        ts,
        NUMERATOR,
        DENOMINATOR,
        NUMERATOR_H,
        DENOMINATOR_H,
        G,
        H,
        Vn,
        Yid,
    ):
        self.na = na
        self.nb = nb      # ← Will be [[0]] for ARMA
        self.nc = nc
        self.nd = nd
        self.nf = nf
        self.theta = theta
        self.ts = ts
        self.NUMERATOR = NUMERATOR
        self.DENOMINATOR = DENOMINATOR
        self.NUMERATOR_H = NUMERATOR_H
        self.DENOMINATOR_H = DENOMINATOR_H
        self.G = G        # ← G=1 for ARMA
        self.H = H        # ← H=C/A for ARMA
        self.Vn = Vn
        self.Yid = Yid
```

---

## 8. Validation Strategy

### 8.1 Master Branch Validation Function

**File:** `/Users/josephj/Workspace/SIPPY-master/sippy_unipi/functionset.py`

**Lines 155-223:** `validation()` function

```python
def validation(SYS, u, y, Time, k=1, centering="None"):
    # ...
    for i in range(ydim):
        # one-step ahead predictor
        if k == 1:
            T, Y_u = forced_response((1 / SYS.H[i, 0]) * SYS.G[i, :], Time, u)
            T, Y_y = forced_response(
                1 - (1 / SYS.H[i, 0]), Time, y[i, :] - y_rif[i]
            )
            Yval[i, :] = Y_u + np.atleast_2d(Y_y) + y_rif[i]
        else:
            # k-step ahead predictor
            # ...
    return Yval
```

**For ARMA:**
- `SYS.G = 1` (no input contribution)
- `Y_u = forced_response(1/H * G, Time, u) = forced_response(1/H, Time, u)`
- Since G=1, the input contribution is minimal/identity
- The model is primarily driven by past outputs via `Y_y`

### 8.2 ARMA-Specific Validation Behavior

For ARMA models where `G(z) = 1`:

```
Yval = Y_u + Y_y + y_rif
     = forced_response(1/H, Time, u) + forced_response(1 - 1/H, Time, y) + y_rif
```

Since `1/H + (1 - 1/H) = 1`, this reduces to:
```
Yval ≈ forced_response(1/H, Time, u) + forced_response(1 - 1/H, Time, y) + y_rif
```

However, with `G=1`, the input contribution is an identity passthrough, so ARMA validation primarily uses past output information.

---

## 9. Recommended Validation Strategy for Harold Branch

### 9.1 Direct ARMA Implementation (Recommended)

**Approach:** Implement ARMA as a **separate algorithm class** in harold branch.

**Rationale:**
1. ARMA has distinct API: `ARMA_orders=[na, nc, theta]` (3 params) vs ARMAX (4 params)
2. ARMA requires special handling: `G=1`, no input dynamics
3. Cleaner separation of concerns in factory pattern

**Implementation Path:**

```python
# File: src/sippy/identification/algorithms/arma.py

class ARMAAlgorithm(IdentificationAlgorithm):
    """
    ARMA (AutoRegressive Moving Average) identification.

    Time-series-only model (no inputs):
        y(t) = C(z)/A(z) * e(t)

    Where:
        A(z) = 1 + a1*z^-1 + ... + ana*z^-na  (AR polynomial)
        C(z) = 1 + c1*z^-1 + ... + cnc*z^-nc  (MA polynomial)
    """

    def identify(
        self,
        y: Optional[np.ndarray] = None,
        u: Optional[np.ndarray] = None,
        iddata: Optional["IDData"] = None,
        **kwargs,
    ) -> StateSpaceModel:
        # Extract parameters
        na = kwargs.get("na", 1)
        nc = kwargs.get("nc", 1)
        theta = kwargs.get("theta", 0)
        max_iterations = kwargs.get("max_iterations", 100)
        Ts = kwargs.get("Ts", 1.0)

        # Get data
        if iddata is not None:
            y_data = iddata.y
            u_data = iddata.u
            Ts = iddata.Ts
        else:
            y_data = y
            u_data = u if u is not None else np.zeros_like(y)

        # Call optimization-based ARMA identification
        # (implement similar to ARMAX but with nb=0 internally)
        coeffs = self._estimate_arma(y_data, u_data, na, nc, theta, max_iterations)

        # Create transfer functions with G=1, H=C/A
        G_tf = harold.Transfer([1.0], [1.0], dt=Ts) if HAROLD_AVAILABLE else None
        H_tf = self._create_h_transfer_function(coeffs, Ts)

        # Build state-space representation
        A, B, C, D = self._build_state_space(coeffs, Ts)

        return StateSpaceModel(
            A=A, B=B, C=C, D=D,
            Ts=Ts,
            G_tf=G_tf,
            H_tf=H_tf,
            # ... other fields
        )
```

### 9.2 ARMAX Wrapper Approach (Alternative)

**Approach:** Validate ARMA by calling ARMAX with `nb=0`.

**Pros:**
- Code reuse (no new ARMA implementation needed)
- Ensures consistency with ARMAX

**Cons:**
- API mismatch: ARMA expects 3 params, ARMAX expects 4 params
- Requires special handling in ARMAX to detect `nb=0` case
- G(z)=1 must be enforced explicitly

**Implementation:**

```python
# In ARMAX algorithm:
def identify(self, y, u, iddata=None, **kwargs):
    na = kwargs.get("na", 1)
    nb = kwargs.get("nb", 1)
    nc = kwargs.get("nc", 1)
    theta = kwargs.get("theta", 0)

    # Detect ARMA case
    if nb == 0 or (isinstance(nb, list) and all(n == 0 for n in nb)):
        # ARMA mode: no input dynamics
        # ... special handling for G=1
        pass

    # Regular ARMAX identification
    # ...
```

### 9.3 Cross-Branch Validation Test

To validate harold branch ARMA against master branch:

```python
# File: src/sippy/identification/tests/test_arma_cross_validation.py

def test_arma_master_validation():
    """Cross-validate ARMA against master branch implementation."""

    # Generate test data
    np.random.seed(42)
    N = 500
    e = np.random.randn(N) * 0.1
    y = np.zeros(N)

    # True ARMA(2,1) system: y(t) = 0.5*y(t-1) - 0.3*y(t-2) + e(t) + 0.4*e(t-1)
    a_true = [0.5, -0.3]
    c_true = [0.4]

    for t in range(2, N):
        y[t] = 0.5*y[t-1] - 0.3*y[t-2] + e[t] + 0.4*e[t-1]

    # Identify with harold branch ARMA
    from sippy.identification import SystemIdentification

    sys_harold = SystemIdentification(
        y=y,
        u=None,  # or zeros
        method="ARMA",
        na=2,
        nc=1,
        theta=0,
        Ts=1.0,
    )

    model_harold = sys_harold.identify()

    # Identify with master branch (via subprocess or import)
    import sys
    sys.path.insert(0, '/Users/josephj/Workspace/SIPPY-master')
    from sippy_unipi import system_identification

    u_dummy = np.zeros_like(y)
    model_master = system_identification(
        y, u_dummy, "ARMA",
        ARMA_orders=[2, 1, 0],
        tsample=1.0
    )

    # Compare coefficients
    # Extract A and C coefficients from both models
    # ...

    # Compare predictions
    np.testing.assert_allclose(
        model_harold.Yid,
        model_master.Yid,
        rtol=1e-3,
        atol=1e-3
    )
```

---

## 10. Summary and Recommendations

### 10.1 Key Findings

1. **ARMA exists in master branch** as a fully-functional algorithm
2. **ARMA is NOT a wrapper to ARMAX** - it's a distinct method with different API
3. **ARMA uses optimization-based identification** (CasADi/IPOPT) via `io_optMIMO.GEN_MIMO_id()`
4. **ARMA mathematical model:**
   - G(z) = 1 (no input transfer function)
   - H(z) = C(z)/A(z) (noise transfer function)
   - Regressor: `phi = [-y(t-1), ..., -y(t-na), e(t-1), ..., e(t-nc)]`
   - Coefficients: `[a1, ..., ana, c1, ..., cnc]`

### 10.2 Validation Strategy Recommendation

**Recommended Approach:** **Direct ARMA Implementation**

1. Create `src/sippy/identification/algorithms/arma.py`
2. Implement `ARMAAlgorithm` class extending `IdentificationAlgorithm`
3. Use optimization-based estimation (similar to ARMAX/OE/BJ)
4. Set `nb=0` internally, expose 3-parameter API: `[na, nc, theta]`
5. Build transfer functions: `G=1`, `H=C/A`
6. Register in factory: `AlgorithmFactory.register("ARMA", ARMAAlgorithm)`

**Cross-Validation Test Plan:**
1. Generate synthetic ARMA data with known coefficients
2. Identify with harold branch ARMA
3. Identify with master branch ARMA
4. Compare:
   - Coefficient estimates (A, C polynomials)
   - Predicted outputs (Yid)
   - Transfer function H(z)
   - Fit metrics (Vn, explained variance)

### 10.3 Implementation Checklist

- [ ] Read master branch ARMA implementation details (DONE)
- [ ] Create ARMA algorithm class in harold branch
- [ ] Implement optimization-based ARMA estimation
- [ ] Handle transfer function creation (G=1, H=C/A)
- [ ] Write unit tests for ARMA
- [ ] Write cross-validation test against master branch
- [ ] Update factory registration
- [ ] Update documentation

---

## 11. Code Locations Reference

| Component | File | Lines |
|-----------|------|-------|
| ARMA Entry Point | `/Users/josephj/Workspace/SIPPY-master/sippy_unipi/__init__.py` | 639-685 |
| ARMA Identification Path | `/Users/josephj/Workspace/SIPPY-master/sippy_unipi/__init__.py` | 893-941 |
| ARMA MIMO Implementation | `/Users/josephj/Workspace/SIPPY-master/sippy_unipi/io_optMIMO.py` | 146-163 |
| ARMA SISO Implementation | `/Users/josephj/Workspace/SIPPY-master/sippy_unipi/io_opt.py` | 81-82 |
| ARMA Optimization Setup | `/Users/josephj/Workspace/SIPPY-master/sippy_unipi/functionset_OPT.py` | 87-94, 154-155 |
| ARMA Example Usage | `/Users/josephj/Workspace/SIPPY-master/Examples/Ex_OPT_GEN-INOUT.py` | 205-207 |
| Validation Function | `/Users/josephj/Workspace/SIPPY-master/sippy_unipi/functionset.py` | 155-223 |
| Model Class | `/Users/josephj/Workspace/SIPPY-master/sippy_unipi/io_optMIMO.py` | 316-350 |

---

## 12. Pseudocode: ARMA Identification Algorithm

```
ALGORITHM: ARMA Identification (Optimization-Based)
INPUT:
    y: output time series [N x 1]
    na: AR order (number of past outputs)
    nc: MA order (number of past prediction errors)
    theta: delay (usually 0 for ARMA)
    max_iterations: maximum optimization iterations
OUTPUT:
    A: AR polynomial coefficients [na x 1]
    C: MA polynomial coefficients [nc x 1]
    G_tf: Input transfer function (always 1 for ARMA)
    H_tf: Noise transfer function C(z)/A(z)
    Yid: Predicted outputs [N x 1]

PROCEDURE:
1. Initialize:
   - nb ← 0 (no input terms)
   - val ← max(na, nc) (maximum lag)
   - n_coeff ← na + nc (total parameters)
   - coefficients ← [a1, ..., ana, c1, ..., cnc]

2. Build Optimization Problem:
   FOR k = val to N-1:
       # Build regressor
       vecY ← [y(k-1), y(k-2), ..., y(k-na)]  (reversed)
       vecE ← [e(k-1), e(k-2), ..., e(k-nc)]  (reversed, prediction errors)
       phi ← [-vecY; vecE]

       # One-step prediction
       Yid(k) ← phi^T · coefficients

       # Prediction error
       e(k) ← y(k) - Yid(k)

   # Objective function
   J ← (1/N) · sum((y - Yid)^2)

   # Constraints
   g ← [Yid_predicted - Yid_optimvar]  (multiple shooting)

   IF stability_constraint:
       # Companion matrix for A(z)
       CompA ← companion_matrix([1, a1, ..., ana])
       g ← [g; norm_inf(CompA)]  (ensure ||CompA|| < stability_margin)

3. Solve Optimization Problem:
   - Solver: IPOPT (Interior Point OPTimizer)
   - Variables: [a1, ..., ana, c1, ..., cnc, Yid(1), ..., Yid(N)]
   - Minimize: J
   - Subject to: g = 0 (equality constraints)
   - Bounds: -100 ≤ coefficients ≤ 100

4. Extract Results:
   - A_coeffs ← optimized [a1, ..., ana]
   - C_coeffs ← optimized [c1, ..., cnc]

5. Build Transfer Functions:
   - A_poly ← [1, a1, a2, ..., ana]
   - C_poly ← [1, c1, c2, ..., cnc]

   - G(z) ← 1  (no input dynamics)
   - H(z) ← C(z) / A(z)

6. Return Model:
   RETURN (A_coeffs, C_coeffs, G_tf=1, H_tf, Yid)
```

---

## Appendix A: ARMA Model Validation Example

```python
# Example: Validate ARMA model on new data

# Training
model = system_identification(
    y_train, u_dummy, "ARMA",
    ARMA_orders=[2, 1, 0],
    tsample=1.0
)

# Validation
y_pred = validation(model, u_valid, y_valid, Time, k=1)

# Metrics
explained_variance = 1 - np.var(y_valid - y_pred) / np.var(y_valid)
print(f"Explained Variance: {explained_variance * 100:.2f}%")
```

---

## Appendix B: ARMA vs ARMAX API Comparison

```python
# ARMA (3 parameters)
model_arma = system_identification(
    y, u, "ARMA",
    ARMA_orders=[na, nc, theta]  # 3 params: AR order, MA order, delay
)

# ARMAX (4 parameters)
model_armax = system_identification(
    y, u, "ARMAX",
    ARMAX_orders=[na, nb, nc, theta]  # 4 params: AR, input, MA, delay
)
```

**Key Difference:** ARMA omits `nb` parameter entirely. It's not just `nb=0`, it's **not present in the API**.

---

**End of Report**
