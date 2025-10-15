"""High-performance helpers ported from the master branch optimization stack.

This module centralises the NLP-based input-output identification utilities
(`opt_id`, ARMAX/ARARX MISO/MIMO solvers, etc.) so that the harold branch can
reuse the exact reference algorithms while benefiting from performance
optimisations (Numba-powered preprocessing, memory-friendly data handling).

The goal is to provide drop-in compatible functions that mirror
``sippy_unipi`` whilst exposing richer metadata (raw coefficient vectors,
scaling factors) required by the refactored object-oriented API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import harold

    HAROLD_AVAILABLE = True
except ImportError:  # pragma: no cover - harold is an optional dependency
    HAROLD_AVAILABLE = False

try:  # pragma: no cover - CasADi is optional but required for NLP methods
    from casadi import DM, SX, mtimes, nlpsol, norm_inf, vertcat

    CASADI_AVAILABLE = True
except ImportError:  # pragma: no cover
    CASADI_AVAILABLE = False

# Performance utilities (Numba-backed with safe fallbacks)
from ...utils.compiled_utils import (
    rescale_compiled,
    rescale_multi_channel_compiled,
    Vn_mat_adaptive,
    build_armax_regression_miso_parallel,
)


@dataclass(slots=True)
class MISOResult:
    """Container for single-output identification results."""

    a_coeffs: np.ndarray
    b_coeffs: List[np.ndarray]
    c_coeffs: np.ndarray
    d_coeffs: np.ndarray
    f_coeffs: np.ndarray
    delay: np.ndarray
    y_hat: np.ndarray
    noise_variance: float
    reached_max: bool
    y_std: float
    u_std: np.ndarray

    def build_transfer_function(
        self, sample_time: float
    ) -> Tuple[Optional["harold.Transfer"], Optional["harold.Transfer"]]:
        """Return (G, H) transfer functions when harold is available."""

        if not HAROLD_AVAILABLE:
            return None, None

        # Build denominator polynomials A(q), D(q), F(q)
        a_poly = np.concatenate(([1.0], self.a_coeffs)) if self.a_coeffs.size else np.array([1.0])
        d_poly = np.concatenate(([1.0], self.d_coeffs)) if self.d_coeffs.size else np.array([1.0])
        f_poly = np.concatenate(([1.0], self.f_coeffs)) if self.f_coeffs.size else np.array([1.0])

        # Multiply polynomials with harold when available (safer, often faster)
        try:
            den_g = harold.haroldpolymul(a_poly, f_poly).tolist()
            den_h = harold.haroldpolymul(a_poly, d_poly).tolist()
        except Exception:
            den_g = np.convolve(a_poly, f_poly).tolist()
            den_h = np.convolve(a_poly, d_poly).tolist()

        # Construct numerators per input (respecting delays)
        if not self.b_coeffs:
            g_tf = harold.Transfer([0.0], den_g, dt=sample_time)
        else:
            num_g = []
            for b_vector, delay in zip(self.b_coeffs, self.delay):
                if b_vector.size == 0:
                    num_g.append([0.0])
                    continue
                leading_zeros = np.zeros(int(delay), dtype=float)
                num_g.append(np.concatenate((leading_zeros, b_vector)).tolist())
            g_tf = harold.Transfer(num_g, den_g, dt=sample_time)

        num_h = (np.concatenate(([1.0], self.c_coeffs)).tolist() if self.c_coeffs.size else [1.0])
        h_tf = harold.Transfer(num_h, den_h, dt=sample_time)

        return g_tf, h_tf


def _ensure_column_major(array: np.ndarray) -> np.ndarray:
    """Return a contiguous float64 array (column-major friendly view)."""

    return np.ascontiguousarray(array, dtype=np.float64)


def _rescale_signal(signal: np.ndarray) -> Tuple[float, np.ndarray]:
    """Scale by standard deviation (master branch behaviour)."""

    std = float(np.std(signal))
    if std < np.finfo(np.float64).eps:
        std = 1.0
    return std, signal / std


def opt_id(
    m: int,
    p: int,
    na: int,
    nb: np.ndarray,
    nc: int,
    nd: int,
    nf: int,
    n_coeff: int,
    theta: np.ndarray,
    n_tr: int,
    U: np.ndarray,
    Y: np.ndarray,
    flag: str,
    max_iterations: int,
    stab_marg: float,
    stability_cons: bool,
):
    """Port of master ``functionset_OPT.opt_id`` with identical semantics."""

    if not CASADI_AVAILABLE:
        raise RuntimeError("CasADi is required for NLP-based identification")

    Na = na
    Nb = np.sum(nb[:])
    Nc = nc
    Nd = nd
    Nf = nf

    N = Y.size
    n_aus = 3 * N if Nd != 0 else N
    n_opt = n_aus + n_coeff

    w_opt = SX.sym("w", n_opt)

    a = w_opt[0:Na]
    b = w_opt[Na : Na + Nb]
    c = w_opt[Na + Nb : Na + Nb + Nc]
    d = w_opt[Na + Nb + Nc : Na + Nb + Nc + Nd]
    f = w_opt[Na + Nb + Nd + Nc : Na + Nb + Nc + Nd + Nf]

    Yidw = w_opt[-N:]

    if Nd != 0:
        Ww = w_opt[-3 * N : -2 * N]
        Vw = w_opt[-2 * N : -N]

    w_lb = -1e2 * DM.ones(n_opt)
    w_ub = 1e2 * DM.ones(n_opt)

    if flag == "OE":
        coeff = vertcat(b, f)
    elif flag == "BJ":
        coeff = vertcat(b, f, c, d)
    elif flag == "ARMAX":
        coeff = vertcat(a, b, c)
    elif flag == "ARARX":
        coeff = vertcat(a, b, d)
    elif flag == "ARARMAX":
        coeff = vertcat(a, b, c, d)
    elif flag == "ARMA":
        coeff = vertcat(a, c)
    else:
        coeff = vertcat(a, b, f, c, d)

    Yid = Y * SX.ones(1)

    if Nd != 0:
        W = Y * SX.ones(1)
        V = Y * SX.ones(1)

        if Na != 0:
            coeff_v = a
        if Nf != 0:
            coeff_w = vertcat(b, f)
        else:
            coeff_w = vertcat(b)

    if Nc != 0:
        Epsi = SX.zeros(N)

    for k in range(N):
        if k >= n_tr:
            if Nb != 0:
                vecU = DM()
                for nb_i in range(m):
                    vecu = U[nb_i, :][k - nb[nb_i] - theta[nb_i] : k - theta[nb_i]][::-1]
                    vecU = vertcat(vecU, vecu)

            if Na != 0:
                vecY = Y[k - Na : k][::-1]

            if Nd != 0:
                vecV = Vw[k - Nd : k][::-1]
                if Nf != 0:
                    vecW = Ww[k - Nf : k][::-1]

            if Nc != 0:
                vecE = Epsi[k - Nc : k][::-1]

            if flag == "OE":
                vecY = Yidw[k - Nf : k][::-1]
                phi = vertcat(vecU, -vecY)
            elif flag == "BJ":
                phi = vertcat(vecU, -vecW, vecE, -vecV)
            elif flag == "ARMAX":
                phi = vertcat(-vecY, vecU, vecE)
            elif flag == "ARMA":
                phi = vertcat(-vecY, vecE)
            elif flag == "ARARX":
                phi = vertcat(-vecY, vecU, -vecV)
            elif flag == "ARARMAX":
                phi = vertcat(-vecY, vecU, vecE, -vecV)
            else:
                phi = vertcat(-vecY, vecU, -vecW, vecE, -vecV)

            Yid[k] = mtimes(phi.T, coeff)

            if Nc != 0:
                Epsi[k] = Y[k] - Yidw[k]

            if Nd != 0:
                if Nf != 0:
                    phiw = vertcat(vecU, -vecW)
                else:
                    phiw = vertcat(vecU)
                W[k] = mtimes(phiw.T, coeff_w)

                if Na == 0:
                    V[k] = Y[k] - Ww[k]
                else:
                    phiv = vertcat(vecY)
                    V[k] = Y[k] + mtimes(phiv.T, coeff_v) - Ww[k]

    DY = Y - Yidw
    f_obj = (1.0 / N) * mtimes(DY.T, DY)

    g = [Yid - Yidw]

    if Nd != 0:
        g.append(W - Ww)
        g.append(V - Vw)

    ng_norm = 0
    if stability_cons:
        if Na != 0:
            ng_norm += 1
            compA = SX.zeros(Na, Na)
            diagA = SX.eye(Na - 1) if Na > 1 else SX.zeros(0, 0)
            if Na > 1:
                compA[:-1, 1:] = diagA
            compA[-1, :] = -a[::-1]
            norm_comp_a = norm_inf(compA)
            g.append(norm_comp_a)

        if Nf != 0:
            ng_norm += 1
            compF = SX.zeros(Nf, Nf)
            diagF = SX.eye(Nf - 1) if Nf > 1 else SX.zeros(0, 0)
            if Nf > 1:
                compF[:-1, 1:] = diagF
            compF[-1, :] = -f[::-1]
            norm_comp_f = norm_inf(compF)
            g.append(norm_comp_f)

        if Nd != 0:
            ng_norm += 1
            compD = SX.zeros(Nd, Nd)
            diagD = SX.eye(Nd - 1) if Nd > 1 else SX.zeros(0, 0)
            if Nd > 1:
                compD[:-1, 1:] = diagD
            compD[-1, :] = -d[::-1]
            norm_comp_d = norm_inf(compD)
            g.append(norm_comp_d)

    g_vec = vertcat(*g)

    ng_total = g_vec.size1()
    g_lb = -1e-7 * DM.ones(ng_total, 1)
    g_ub = 1e-7 * DM.ones(ng_total, 1)

    if ng_norm != 0:
        g_ub[-ng_norm:] = stab_marg * DM.ones(ng_norm, 1)

    nlp = {"x": w_opt, "f": f_obj, "g": g_vec}

    sol_opts = {
        "ipopt.max_iter": max_iterations,
        "ipopt.print_level": 0,
        "ipopt.sb": "yes",
        "print_time": 0,
    }

    solver = nlpsol("solver", "ipopt", nlp, sol_opts)

    return solver, w_lb, w_ub, g_lb, g_ub


def _compute_polynomials(
    theta: np.ndarray,
    na: int,
    nb: Sequence[int],
    nc: int,
    nd: int,
    nf: int,
    y_std: float,
    u_std: np.ndarray,
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split the optimisation vector into AR/ARX polynomials."""

    pos = 0
    a_coeffs = theta[pos : pos + na]
    pos += na

    b_coeffs = []
    for idx, nb_i in enumerate(nb):
        coeff_slice = theta[pos : pos + nb_i]
        if nb_i > 0:
            coeff_slice = coeff_slice * (y_std / u_std[idx])
        b_coeffs.append(coeff_slice)
        pos += nb_i

    c_coeffs = theta[pos : pos + nc]
    pos += nc
    d_coeffs = theta[pos : pos + nd]
    pos += nd
    f_coeffs = theta[pos : pos + nf]

    return b_coeffs, a_coeffs, c_coeffs, d_coeffs, f_coeffs


def gen_miso_id(
    id_method: str,
    y: np.ndarray,
    u: np.ndarray,
    na: int,
    nb: Sequence[int],
    nc: int,
    nd: int,
    nf: int,
    theta: Sequence[int],
    max_iterations: int,
    stability_margin: float,
    enforce_stability: bool,
) -> MISOResult:
    """Solve the master NLP for a single output / multi-input structure."""

    nb = np.array(nb, dtype=int)
    theta = np.array(theta, dtype=int)
    u = _ensure_column_major(np.atleast_2d(u))
    y = _ensure_column_major(np.atleast_1d(y))

    # Use compiled rescaling for speed and stability
    y_std, y_scaled = rescale_compiled(y)
    u_std, u_scaled = rescale_multi_channel_compiled(u, axis=0)
    y_scaled = _ensure_column_major(y_scaled)
    u_scaled = _ensure_column_major(u_scaled)

    nb_theta = nb + theta
    nb_max = int(np.max(nb_theta)) if nb_theta.size else 0
    val = int(max(na, nb_max, nc, nd, nf))
    n_coeff = na + np.sum(nb) + nc + nd + nf

    solver, w_lb, w_ub, g_lb, g_ub = opt_id(
        m=u.shape[0],
        p=1,
        na=na,
        nb=nb,
        nc=nc,
        nd=nd,
        nf=nf,
        n_coeff=n_coeff,
        theta=theta,
        n_tr=val,
        U=u_scaled,
        Y=y_scaled,
        flag=id_method,
        max_iterations=max_iterations,
        stab_marg=stability_margin,
        stability_cons=enforce_stability,
    )

    w0 = np.zeros((1, n_coeff))
    w0 = np.hstack([w0, np.atleast_2d(y_scaled)])
    if id_method in {"BJ", "GEN", "ARARX", "ARARMAX"}:
        w0 = np.hstack([w0, np.atleast_2d(y_scaled), np.atleast_2d(y_scaled)])

    sol = solver(lbx=w_lb, ubx=w_ub, x0=w0, lbg=g_lb, ubg=g_ub)

    x_opt = sol["x"]
    iterations = solver.stats().get("iter_count", 0)
    reached_max = iterations >= max_iterations

    theta_vec = np.array(x_opt[:n_coeff])[:, 0]
    y_hat_scaled = x_opt[-y_scaled.size :].full()[:, 0]
    y_hat = y_hat_scaled * y_std

    Vn = (np.linalg.norm(y_hat - y, 2) ** 2) / (2 * y.size)

    b_coeffs, a_coeffs, c_coeffs, d_coeffs, f_coeffs = _compute_polynomials(
        theta_vec,
        na=na,
        nb=nb,
        nc=nc,
        nd=nd,
        nf=nf,
        y_std=y_std,
        u_std=u_std,
    )

    return MISOResult(
        a_coeffs=a_coeffs,
        b_coeffs=b_coeffs,
        c_coeffs=c_coeffs,
        d_coeffs=d_coeffs,
        f_coeffs=f_coeffs,
        delay=theta,
        y_hat=y_hat,
        noise_variance=Vn,
        reached_max=reached_max,
        y_std=y_std,
        u_std=u_std,
    )


def gen_mimo_id(
    id_method: str,
    y: np.ndarray,
    u: np.ndarray,
    na: Sequence[int],
    nb: np.ndarray,
    nc: Sequence[int],
    nd: Sequence[int],
    nf: Sequence[int],
    theta: np.ndarray,
    sample_time: float,
    max_iterations: int,
    stability_margin: float,
    enforce_stability: bool,
) -> Tuple[List[MISOResult], float]:
    """Run the optimisation for each output (MISO decomposition)."""

    y = np.atleast_2d(y)
    u = np.atleast_2d(u)

    ny, _ = y.shape
    results: List[MISOResult] = []
    variance_total = 0.0

    for idx in range(ny):
        result = gen_miso_id(
            id_method=id_method,
            y=y[idx],
            u=u,
            na=int(na[idx]),
            nb=nb[idx],
            nc=int(nc[idx]),
            nd=int(nd[idx]),
            nf=int(nf[idx]),
            theta=theta[idx],
            max_iterations=max_iterations,
            stability_margin=stability_margin,
            enforce_stability=enforce_stability,
        )
        results.append(result)
        variance_total += result.noise_variance

    return results, variance_total


def armax_miso_id(
    y: np.ndarray,
    u: np.ndarray,
    na: int,
    nb: Sequence[int],
    nc: int,
    theta: Sequence[int],
    max_iterations: int,
) -> MISOResult:
    """Iterative least squares ARMAX for multiple inputs (master-compatible)."""

    y = np.asarray(y, dtype=float)
    u = np.asarray(u, dtype=float)
    nb = np.asarray(nb, dtype=int)
    theta = np.asarray(theta, dtype=int)

    # Compiled rescaling (vectorized) for output and inputs
    y_std, y_scaled = rescale_compiled(y)
    u_std, u_scaled = rescale_multi_channel_compiled(u, axis=0)
    y_scaled = _ensure_column_major(y_scaled)
    u_scaled = _ensure_column_major(u_scaled)

    eps = np.zeros_like(y_scaled)

    nbth = nb + theta
    nb_max = int(np.max(nbth)) if nbth.size else 0
    val = int(max(na, nb_max, nc))
    N = y_scaled.size - val

    if N <= 0:
        raise ValueError("Not enough samples for ARMAX identification")

    sum_orders = na + np.sum(nb) + nc

    Vn = np.inf
    Vn_old = np.inf
    theta_vec = np.zeros(sum_orders)
    lamb = 0.5
    iterations = 0

    while (Vn_old > Vn or iterations == 0) and iterations < max_iterations:
        theta_prev = theta_vec.copy()
        Vn_old = Vn
        iterations += 1

        # Build regression in parallel (Numba) per iteration (eps updates MA block)
        Phi = build_armax_regression_miso_parallel(
            y_scaled, u_scaled, eps, int(na), nb.astype(np.int64), int(nc), theta.astype(np.int64), val, N
        )

        theta_vec = np.linalg.pinv(Phi) @ y_scaled[val:]
        # Adaptive residual variance (SIMD/parallel) for speed
        Vn = float(Vn_mat_adaptive(y_scaled[val:], Phi @ theta_vec)) / 2.0

        theta_new = theta_vec.copy()
        lamb = 0.5
        while Vn > Vn_old and iterations > 1:
            theta_vec = lamb * theta_new + (1 - lamb) * theta_prev
            Vn = (np.linalg.norm(y_scaled[val:] - Phi @ theta_vec, 2) ** 2) / (2 * N)
            if lamb < np.finfo(np.float32).eps:
                theta_vec = theta_prev
                Vn = Vn_old
                break
            lamb /= 2.0

        eps[val:] = y_scaled[val:] - Phi @ theta_vec

    y_hat = np.hstack((y_scaled[:val], Phi @ theta_vec)) * y_std

    pos = 0
    a_coeffs = theta_vec[pos : pos + na]
    pos += na

    b_coeffs = []
    for idx in range(u.shape[0]):
        nb_i = nb[idx]
        coeff = theta_vec[pos : pos + nb_i]
        if nb_i > 0:
            coeff = coeff * (y_std / u_std[idx])
        b_coeffs.append(coeff)
        pos += nb_i

    c_coeffs = theta_vec[pos : pos + nc]

    return MISOResult(
        a_coeffs=a_coeffs,
        b_coeffs=b_coeffs,
        c_coeffs=c_coeffs,
        d_coeffs=np.array([], dtype=float),
        f_coeffs=np.array([], dtype=float),
        delay=theta,
        y_hat=y_hat,
        noise_variance=Vn,
        reached_max=iterations >= max_iterations,
        y_std=y_std,
        u_std=u_std,
    )


def armax_mimo_id(
    y: np.ndarray,
    u: np.ndarray,
    na: Sequence[int],
    nb: np.ndarray,
    nc: Sequence[int],
    theta: np.ndarray,
    sample_time: float,
    max_iterations: int,
) -> Tuple[List[MISOResult], float]:
    """Run ARMAX MIMO identification via MISO decomposition."""

    y = np.atleast_2d(y)
    u = np.atleast_2d(u)
    ny = y.shape[0]

    results = []
    variance_total = 0.0

    for idx in range(ny):
        result = armax_miso_id(
            y=y[idx],
            u=u,
            na=int(na[idx]),
            nb=nb[idx],
            nc=int(nc[idx]),
            theta=theta[idx],
            max_iterations=max_iterations,
        )
        results.append(result)
        variance_total += result.noise_variance

    return results, variance_total
