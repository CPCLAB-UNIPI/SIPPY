"""
Continuous/discrete conversion under ZOH assumptions.

This module provides ZOH-based conversion routines for state-space models.

Available functions
-------------------
- c2d_zoh:
  continuous-to-discrete conversion under zero-order hold assumptions.
- d2c_zoh:
  discrete-to-continuous conversion under zero-order hold assumptions,
  including support for simple negative real discrete poles.

Scope
-----
Supported:
- continuous-time state-space models for standard ZOH c2d conversion;
- minimal discrete-time state-space models for ZOH d2c conversion;
- standard ZOH d2c when no negative real discrete poles are present;
- simple negative real discrete poles in the discrete-time model.

Not supported
-------------
- poles at z = 0 in d2c conversion;
- repeated negative real poles in d2c conversion;
- Jordan blocks associated with negative real poles;
- non-minimal discrete-time realizations in d2c conversion.

Main functions
--------------
- c2d_zoh(sys_c, Ts, ...)
- d2c_zoh(sys_d, ...)

Each function returns a ``control.StateSpace`` model and, optionally,
a compact diagnostic dictionary.

Dependencies
------------
- numpy
- scipy
- python-control

@author: Matteo Frascarelli
"""

import math
import warnings

import numpy as np
import scipy.linalg as la
import control as cnt


# =============================================================================
# Continuous to discrete
# =============================================================================

def c2d_zoh(
    sys_c,
    Ts,
    check_roundtrip=False,
    tol_roundtrip=1e-7,
    return_info=False,
):
    """
    Convert a continuous-time state-space model to discrete time under ZOH.
    """
    try:
        Ts = float(Ts)
    except Exception as exc:
        raise ValueError("Sampling time Ts must be convertible to float.") from exc

    if Ts <= 0.0:
        raise ValueError("Sampling time Ts must be > 0.")

    Ac = np.array(sys_c.A, copy=True, dtype=float)
    Bc = np.array(sys_c.B, copy=True, dtype=float)
    Cc = np.array(sys_c.C, copy=True, dtype=float)
    Dc = np.array(sys_c.D, copy=True, dtype=float)

    if Ac.ndim != 2 or Ac.shape[0] != Ac.shape[1]:
        raise ValueError("A must be a square 2D matrix.")

    n = Ac.shape[0]

    if Bc.ndim != 2 or Bc.shape[0] != n:
        raise ValueError("B must be a 2D matrix with the same number of rows as A.")

    if Cc.ndim != 2 or Cc.shape[1] != n:
        raise ValueError("C must be a 2D matrix with the same number of columns as A.")

    if Dc.ndim != 2 or Dc.shape != (Cc.shape[0], Bc.shape[1]):
        raise ValueError("D must have shape (n_outputs, n_inputs).")

    m = Bc.shape[1]

    Mc = np.block([
        [Ac, Bc],
        [np.zeros((m, n)), np.zeros((m, m))]
    ])

    try:
        Md = la.expm(Mc * Ts)
    except Exception as exc:
        raise RuntimeError("Matrix exponential failed during ZOH c2d conversion.") from exc

    Ad = np.real_if_close(Md[:n, :n], tol=1000)
    Bd = np.real_if_close(Md[:n, n:n + m], tol=1000)
    Cd = Cc.copy()
    Dd = Dc.copy()

    sys_d = cnt.ss(np.real(Ad), np.real(Bd), Cd, Dd, Ts)

    info = {"Ts": Ts}

    if check_roundtrip:
        sys_c_back = d2c_zoh(sys_d, check_roundtrip=False, return_info=False)

        theta = np.linspace(0.0, np.pi, 1000, endpoint=False)
        z_points = np.exp(1j * theta)

        sys_d_back = cnt.sample_system(sys_c_back, Ts, method="zoh")
        roundtrip_err = _relerr_freq(sys_d, sys_d_back, z_points)

        info["roundtrip_tf_error"] = roundtrip_err

        if roundtrip_err > tol_roundtrip:
            warnings.warn(
                "c2d_zoh round-trip transfer-function check: "
                f"relative error = {roundtrip_err:.3e} > "
                f"tol_roundtrip = {tol_roundtrip:.3e}.",
                RuntimeWarning,
            )

    if return_info:
        return sys_d, info

    return sys_d


# =============================================================================
# Discrete to continuous utilities
# =============================================================================

def _d2c_standard_no_neg_ss(Ad, Bd, Cd, Dd, Ts):
    """
    Standard ZOH d2c via augmented matrix logarithm.
    Assumes no negative real discrete poles are present.
    """
    n, m = Bd.shape

    if n == 0:
        return cnt.ss([], [], [], Dd)

    Md = np.block([
        [Ad, Bd],
        [np.zeros((m, n)), np.eye(m)]
    ])

    try:
        Mc = la.logm(Md) / Ts
    except Exception as exc:
        raise RuntimeError("Matrix logarithm failed during ZOH d2c conversion.") from exc

    Ac = np.real_if_close(Mc[:n, :n], tol=1000)
    Bc = np.real_if_close(Mc[:n, n:n + m], tol=1000)

    return cnt.ss(np.real(Ac), np.real(Bc), Cd.copy(), Dd.copy())


def _eval_ss_freq(sys, z_points):
    """
    Evaluate a discrete-time state-space transfer matrix on arbitrary z-points.
    """
    A = np.array(sys.A, copy=False)
    B = np.array(sys.B, copy=False)
    C = np.array(sys.C, copy=False)
    D = np.array(sys.D, copy=False)

    p, m = D.shape
    vals = np.zeros((len(z_points), p, m), dtype=complex)

    if A.size == 0:
        for k in range(len(z_points)):
            vals[k] = D
        return vals

    I = np.eye(A.shape[0])

    for k, z in enumerate(z_points):
        vals[k] = C @ la.solve(z * I - A, B) + D

    return vals


def _relerr_freq(sys1, sys2, z_points):
    """
    Relative frequency-domain error between two discrete systems.
    """
    V1 = _eval_ss_freq(sys1, z_points)
    V2 = _eval_ss_freq(sys2, z_points)

    num = np.max(np.linalg.norm((V1 - V2).reshape(len(z_points), -1), axis=1))
    den = max(
        np.max(np.linalg.norm(V1.reshape(len(z_points), -1), axis=1)),
        1e-15,
    )

    return num / den


# =============================================================================
# Discrete to continuous
# =============================================================================

def d2c_zoh(
    sys_d,
    tol_zero=1e-12,
    tol_imag=1e-10,
    tol_rank=1e-10,
    tol_roundtrip=1e-7,
    check_roundtrip=True,
    return_info=False,
):
    """
    Discrete-to-continuous conversion under ZOH assumptions.

    Supported cases
    ---------------
    - Standard discrete-time systems with no negative real poles:
      conversion via augmented matrix logarithm.
    - Simple negative real discrete poles:
      conversion via Schur reordering + Sylvester decoupling + transfer-level
      reconstruction with minimal continuous second-order blocks.

    Assumptions / choices
    ---------------------
    - sys_d must be a discrete-time state-space model with valid dt > 0.
    - sys_d must be a minimal realization.
    - Poles at z = 0 are rejected.
    - Simple negative real discrete poles are supported.
    - Repeated negative real poles / Jordan blocks are not supported.
    - Round-trip check, when enabled, is performed at transfer-function level.
    """
    # ------------------------------------------------------------------
    # Validate dt
    # ------------------------------------------------------------------
    dt = sys_d.dt

    if dt is None or dt is True:
        raise ValueError(
            "System must be a discrete-time system with a specified positive dt."
        )

    try:
        Ts = float(dt)
    except Exception as exc:
        raise ValueError("System dt must be convertible to float.") from exc

    if Ts <= 0.0:
        raise ValueError("System dt must be > 0.")

    # ------------------------------------------------------------------
    # Extract matrices
    # ------------------------------------------------------------------
    Ad = np.array(sys_d.A, copy=True, dtype=float)
    Bd = np.array(sys_d.B, copy=True, dtype=float)
    Cd = np.array(sys_d.C, copy=True, dtype=float)
    Dd = np.array(sys_d.D, copy=True, dtype=float)

    # ------------------------------------------------------------------
    # Dimension checks
    # ------------------------------------------------------------------
    if Ad.ndim != 2 or Ad.shape[0] != Ad.shape[1]:
        raise ValueError("A must be a square 2D matrix.")

    n = Ad.shape[0]

    if Bd.ndim != 2 or Bd.shape[0] != n:
        raise ValueError("B must be a 2D matrix with the same number of rows as A.")

    if Cd.ndim != 2 or Cd.shape[1] != n:
        raise ValueError("C must be a 2D matrix with the same number of columns as A.")

    if Dd.ndim != 2 or Dd.shape != (Cd.shape[0], Bd.shape[1]):
        raise ValueError("D must have shape (n_outputs, n_inputs).")

    p = Cd.shape[0]
    m = Bd.shape[1]

    # ------------------------------------------------------------------
    # Minimality check
    # ------------------------------------------------------------------
    ctrb_blocks = [Bd]
    Ak = np.eye(n)

    for _ in range(1, n):
        Ak = Ak @ Ad
        ctrb_blocks.append(Ak @ Bd)

    ctrb = np.hstack(ctrb_blocks)

    obsv_blocks = [Cd]
    Ak = np.eye(n)

    for _ in range(1, n):
        Ak = Ak @ Ad
        obsv_blocks.append(Cd @ Ak)

    obsv = np.vstack(obsv_blocks)

    rank_co = np.linalg.matrix_rank(ctrb, tol=tol_rank)
    rank_ob = np.linalg.matrix_rank(obsv, tol=tol_rank)

    if rank_co != n or rank_ob != n:
        warnings.warn(
            "[WARN] The system under conversion is not minimal",
            RuntimeWarning,
        )

    # ------------------------------------------------------------------
    # Spectral checks
    # ------------------------------------------------------------------
    eigs = np.linalg.eigvals(Ad)

    if np.any(np.abs(eigs) < tol_zero):
        raise ValueError("ZOH d2c cannot convert systems with poles at z = 0.")

    neg_real_mask = (np.real(eigs) < 0.0) & (np.abs(np.imag(eigs)) < tol_imag)
    neg_eigs = eigs[neg_real_mask]

    for ev in neg_eigs:
        mult = np.sum(np.abs(eigs - ev) < 1e-8)

        if mult != 1:
            raise NotImplementedError(
                "Repeated or non-simple negative real discrete poles are not supported."
            )

    # ------------------------------------------------------------------
    # Fast path: no negative real poles
    # ------------------------------------------------------------------
    if neg_eigs.size == 0:
        sys_c = _d2c_standard_no_neg_ss(Ad, Bd, Cd, Dd, Ts)

        info = {
            "Ts": Ts,
            "nu_neg": 0,
        }

        if check_roundtrip:
            theta = np.linspace(0.0, np.pi, 1000, endpoint=False)
            z_points = np.exp(1j * theta)

            sys_d_back = cnt.sample_system(sys_c, Ts, method="zoh")
            roundtrip_err = _relerr_freq(sys_d, sys_d_back, z_points)

            info["roundtrip_tf_error"] = roundtrip_err

            if roundtrip_err > tol_roundtrip:
                warnings.warn(
                    "d2c_zoh round-trip transfer-function check: "
                    f"relative error = {roundtrip_err:.3e} > "
                    f"tol_roundtrip = {tol_roundtrip:.3e}.",
                    RuntimeWarning,
                )

        if return_info:
            return sys_c, info

        return sys_c

    # ------------------------------------------------------------------
    # Negative-real-pole path: Schur reorder
    # ------------------------------------------------------------------
    neg_real_targets = np.array(
        sorted(float(np.real(ev)) for ev in neg_eigs),
        dtype=float,
    )

    tol_select = max(1e-8, 100.0 * tol_imag)

    def _select_negative_real_targets(x):
        """
        Select only the negative real poles already detected from eigvals(Ad).

        This avoids selecting complex-conjugate Schur blocks whose real part is
        negative but whose eigenvalues are not real negative poles.
        """
        xr = float(np.real(x))
        xi = float(np.imag(x))

        if abs(xi) > tol_imag:
            return False

        return bool(np.any(np.abs(xr - neg_real_targets) < tol_select))

    Tschur, Q, sdim = la.schur(
        Ad,
        output="real",
        sort=_select_negative_real_targets,
    )

    nu = int(sdim)
    Ae = Tschur
    Be = Q.T @ Bd
    Ce = Cd @ Q

    Aminus = Ae[:nu, :nu]
    A12 = Ae[:nu, nu:]
    Ar = Ae[nu:, nu:]

    if nu != len(neg_real_targets):
        raise ValueError(
            "Schur reordering did not isolate the expected number of "
            "negative real poles."
        )

    if nu > 0:
        eigs_minus = la.eigvals(Aminus)

        bad_minus = [
            ev for ev in eigs_minus
            if not (
                abs(np.imag(ev)) < tol_imag
                and np.any(np.abs(np.real(ev) - neg_real_targets) < tol_select)
            )
        ]

        if bad_minus:
            raise ValueError(
                "Schur reordering selected eigenvalues that are not simple "
                "negative real poles."
            )

    # ------------------------------------------------------------------
    # Sylvester decoupling
    # ------------------------------------------------------------------
    if nu > 0 and Ar.shape[0] > 0:
        X = la.solve_sylvester(Aminus, -Ar, -A12)

        M = np.block([
            [np.eye(nu), X],
            [np.zeros((Ar.shape[0], nu)), np.eye(Ar.shape[0])]
        ])

        Minv = np.block([
            [np.eye(nu), -X],
            [np.zeros((Ar.shape[0], nu)), np.eye(Ar.shape[0])]
        ])

        Adec = Minv @ Ae @ M
        Bdec = Minv @ Be
        Cdec = Ce @ M

    else:
        X = np.zeros((nu, Ar.shape[0]))
        Adec = Ae
        Bdec = Be
        Cdec = Ce

    Bminus = Bdec[:nu, :]
    Br = Bdec[nu:, :]
    Cminus = Cdec[:, :nu]
    Cr = Cdec[:, nu:]

    if nu > 0:
        sys_minus_d = cnt.ss(Aminus, Bminus, Cminus, np.zeros((p, m)), Ts)
    else:
        sys_minus_d = cnt.ss([], [], [], np.zeros((p, m)), Ts)

    if Ar.shape[0] > 0:
        sys_reg_d = cnt.ss(Ar, Br, Cr, Dd, Ts)
    else:
        sys_reg_d = cnt.ss([], [], [], Dd, Ts)

    # ------------------------------------------------------------------
    # Convert regular part
    # ------------------------------------------------------------------
    sys_reg_c = _d2c_standard_no_neg_ss(Ar, Br, Cr, Dd, Ts)

    # ------------------------------------------------------------------
    # Reconstruct negative part
    # ------------------------------------------------------------------
    if Aminus.shape[0] == 0:
        sys_minus_c = cnt.ss([], [], [], np.zeros((p, m)))
        neg_terms = []

    else:
        lam, V = la.eig(Aminus)
        Vinv = la.inv(V)

        sys_minus_c = cnt.ss([], [], [], np.zeros((p, m)))
        neg_terms = []

        for i, l in enumerate(lam):
            if abs(np.imag(l)) > tol_imag or np.real(l) >= 0.0:
                raise ValueError("Negative block contains unsupported eigenstructure.")

            v = np.real_if_close(V[:, i:i + 1], tol=1000)
            wT = np.real_if_close(Vinv[i:i + 1, :], tol=1000)

            ci = np.real_if_close(Cminus @ v, tol=1000)   # p x 1
            bi = np.real_if_close(wT @ Bminus, tol=1000)  # 1 x m
            Ri = np.real_if_close(ci @ bi, tol=1000)      # p x m

            zeta = -float(np.real(l))
            alpha = -math.log(zeta) / Ts
            omegaN = math.pi / Ts

            num = [
                alpha / (1.0 + zeta),
                (alpha**2 + omegaN**2) / (1.0 + zeta),
            ]

            den = [
                1.0,
                2.0 * alpha,
                alpha**2 + omegaN**2,
            ]

            ss1 = cnt.ss(cnt.tf(num, den))

            Bc = ss1.B @ np.array(bi, dtype=float)
            Cc = np.array(ci, dtype=float) @ ss1.C
            Dc = np.array(ci, dtype=float) @ ss1.D @ np.array(bi, dtype=float)

            sys_i = cnt.ss(
                np.real_if_close(ss1.A, tol=1000),
                np.real_if_close(Bc, tol=1000),
                np.real_if_close(Cc, tol=1000),
                np.real_if_close(Dc, tol=1000),
            )

            sys_minus_c = sys_minus_c + sys_i

            neg_terms.append({
                "lambda_d": float(np.real(l)),
                "zeta": zeta,
                "alpha": alpha,
                "omegaN": omegaN,
                "Ri": np.array(Ri, dtype=float),
            })

    # ------------------------------------------------------------------
    # Final continuous-time system
    # ------------------------------------------------------------------
    sys_c = sys_reg_c + sys_minus_c

    # ------------------------------------------------------------------
    # Minimal diagnostics
    # ------------------------------------------------------------------
    info = {
        "Ts": Ts,
        "nu_neg": nu,
        "negative_terms": neg_terms,
    }

    if nu > 0 and Ar.shape[0] > 0:
        info["decoupling_residual"] = np.linalg.norm(Adec[:nu, nu:], ord="fro")
    else:
        info["decoupling_residual"] = 0.0

    theta = np.linspace(0.0, np.pi, 1000, endpoint=False)
    z_points = np.exp(1j * theta)

    info["split_error"] = _relerr_freq(sys_d, sys_minus_d + sys_reg_d, z_points)

    if Aminus.shape[0] == 0:
        info["negative_residue_error"] = 0.0

    else:
        lam, V = la.eig(Aminus)
        Vinv = la.inv(V)

        sys_minus_d_res = cnt.ss([], [], [], np.zeros((p, m)), Ts)

        for i, l in enumerate(lam):
            if abs(np.imag(l)) > tol_imag or np.real(l) >= 0.0:
                raise ValueError("Negative block contains unsupported eigenstructure.")

            v = np.real_if_close(V[:, i:i + 1], tol=1000)
            wT = np.real_if_close(Vinv[i:i + 1, :], tol=1000)

            ci = np.real_if_close(Cminus @ v, tol=1000)
            bi = np.real_if_close(wT @ Bminus, tol=1000)

            zeta = -float(np.real(l))

            sys_i = cnt.ss(
                np.array([[-zeta]]),
                np.array(bi, dtype=float),
                np.array(ci, dtype=float),
                np.zeros((p, m)),
                Ts,
            )

            sys_minus_d_res = sys_minus_d_res + sys_i

        info["negative_residue_error"] = _relerr_freq(
            sys_minus_d,
            sys_minus_d_res,
            z_points,
        )

    if check_roundtrip:
        sys_d_back = cnt.sample_system(sys_c, Ts, method="zoh")
        roundtrip_err = _relerr_freq(sys_d, sys_d_back, z_points)

        info["roundtrip_tf_error"] = roundtrip_err

        if roundtrip_err > tol_roundtrip:
            warnings.warn(
                "d2c_zoh round-trip transfer-function check: "
                f"relative error = {roundtrip_err:.3e} > "
                f"tol_roundtrip = {tol_roundtrip:.3e}.",
                RuntimeWarning,
            )

    if return_info:
        return sys_c, info

    return sys_c
