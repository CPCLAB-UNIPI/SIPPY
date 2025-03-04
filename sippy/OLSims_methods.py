"""
Created on Thu Oct 12 2017

@author: Giuseppe Armenise
"""

from warnings import warn

import numpy as np
import scipy as sc
from numpy.linalg import pinv

from .functionset import information_criterion, rescale
from .functionsetSIM import (
    K_calc,
    SS_lsim_process_form,
    Vn_mat,
    Z_dot_PIort,
    check_types,
    impile,
    ordinate_sequence,
    reducingOrder,
)
from .typing import ICMethods, OLSimMethods


def SVD_weighted(y, u, f, l_, weights: OLSimMethods = "N4SID"):
    Yf, Yp = ordinate_sequence(y, f, f)
    Uf, Up = ordinate_sequence(u, f, f)
    Zp = impile(Up, Yp)

    YfdotPIort_Uf = Z_dot_PIort(Yf, Uf)
    ZpdotPIort_Uf = Z_dot_PIort(Zp, Uf)
    O_i = np.dot(np.dot(YfdotPIort_Uf, pinv(ZpdotPIort_Uf)), Zp)

    if weights == "MOESP":
        W1 = None
        #        W2 = PIort_Uf
        OidotPIort_Uf = Z_dot_PIort(O_i, Uf)  # np.dot(O_i, W2)
        U_n, S_n, V_n = np.linalg.svd(OidotPIort_Uf, full_matrices=False)

    elif weights == "CVA":
        W1 = np.linalg.inv(
            sc.linalg.sqrtm(np.dot(YfdotPIort_Uf, YfdotPIort_Uf.T)).real
        )
        W1dotOi = np.dot(W1, O_i)
        W1_dot_Oi_dot_PIort_Uf = Z_dot_PIort(W1dotOi, Uf)
        U_n, S_n, V_n = np.linalg.svd(
            W1_dot_Oi_dot_PIort_Uf, full_matrices=False
        )

    elif weights == "N4SID":
        W1 = None  # is identity
        U_n, S_n, V_n = np.linalg.svd(
            O_i, full_matrices=False
        )  # full matrices not used

    return U_n, S_n, V_n, W1, O_i


def algorithm_1(
    y,
    u,
    l_,
    m_,
    f,
    N,
    U_n,
    S_n,
    V_n,
    W1,
    O_i,
    threshold,
    max_order,
    D_required,
):
    U_n, S_n, V_n = reducingOrder(U_n, S_n, V_n, threshold, max_order)
    V_n = V_n.T
    n = S_n.size
    S_n = np.diag(S_n)
    if W1 is None:  # W1 is identity
        Ob = np.dot(U_n, sc.linalg.sqrtm(S_n))
    else:
        Ob = np.dot(np.linalg.inv(W1), np.dot(U_n, sc.linalg.sqrtm(S_n)))
    X_fd = np.dot(np.linalg.pinv(Ob), O_i)
    Sxterm = impile(X_fd[:, 1:N], y[:, f : f + N - 1])
    Dxterm = impile(X_fd[:, 0 : N - 1], u[:, f : f + N - 1])
    if D_required:
        M = np.dot(Sxterm, np.linalg.pinv(Dxterm))
    else:
        M = np.zeros((n + l_, n + m_))
        M[0:n, :] = np.dot(Sxterm[0:n], np.linalg.pinv(Dxterm))
        M[n::, 0:n] = np.dot(Sxterm[n::], np.linalg.pinv(Dxterm[0:n, :]))
    residuals = Sxterm - np.dot(M, Dxterm)
    return Ob, X_fd, M, n, residuals


def forcing_A_stability(M, n, Ob, l_, X_fd, N, u, f):
    Forced_A = False
    if np.max(np.abs(np.linalg.eigvals(M[0:n, 0:n]))) >= 1.0:
        Forced_A = True
        print("Forcing A stability")
        M[0:n, 0:n] = np.dot(
            np.linalg.pinv(Ob), impile(Ob[l_::, :], np.zeros((l_, n)))
        )
        M[0:n, n::] = np.dot(
            X_fd[:, 1:N] - np.dot(M[0:n, 0:n], X_fd[:, 0 : N - 1]),
            np.linalg.pinv(u[:, f : f + N - 1]),
        )
    res = (
        X_fd[:, 1:N]
        - np.dot(M[0:n, 0:n], X_fd[:, 0 : N - 1])
        - np.dot(M[0:n, n::], u[:, f : f + N - 1])
    )
    return M, res, Forced_A


def extracting_matrices(M, n):
    A = M[0:n, 0:n]
    B = M[0:n, n::]
    C = M[n::, 0:n]
    D = M[n::, n::]
    return A, B, C, D


def OLSims(
    y: np.ndarray,
    u: np.ndarray,
    weights: OLSimMethods,
    order: int = 0,
    threshold: float = 0.0,
    f: int = 20,
    D_required: bool = False,
    A_stability: bool = False,
):
    y = np.atleast_2d(y)
    u = np.atleast_2d(u)
    l_, _ = y.shape
    m_, L = u.shape

    N = L - 2 * f + 1
    U_std = np.zeros(m_)
    Ystd = np.zeros(l_)
    for j in range(m_):
        U_std[j], u[j] = rescale(u[j])
    for j in range(l_):
        Ystd[j], y[j] = rescale(y[j])
    U_n, S_n, V_n, W1, O_i = SVD_weighted(y, u, f, l_, weights)
    Ob, X_fd, M, n, residuals = algorithm_1(
        y,
        u,
        l_,
        m_,
        f,
        N,
        U_n,
        S_n,
        V_n,
        W1,
        O_i,
        threshold,
        order,
        D_required,
    )
    if A_stability:
        M, residuals[0:n, :], _ = forcing_A_stability(
            M, n, Ob, l_, X_fd, N, u, f
        )
    A, B, C, D = extracting_matrices(M, n)
    Covariances = np.dot(residuals, residuals.T) / (N - 1)
    Q = Covariances[0:n, 0:n]
    R = Covariances[n::, n::]
    S = Covariances[0:n, n::]
    _, Y_estimate = SS_lsim_process_form(A, B, C, D, u)

    Vn = Vn_mat(y, Y_estimate)

    K, K_calculated = K_calc(A, C, Q, R, S)
    for j in range(m_):
        B[:, j] = B[:, j] / U_std[j]
        D[:, j] = D[:, j] / U_std[j]
    for j in range(l_):
        C[j, :] = C[j, :] * Ystd[j]
        D[j, :] = D[j, :] * Ystd[j]
        if K_calculated:
            K[:, j] = K[:, j] / Ystd[j]
    return A, B, C, D, Vn, Q, R, S, K


def select_order_SIM(
    y: np.ndarray,
    u: np.ndarray,
    weights: OLSimMethods,
    orders: tuple[int, int] = (1, 10),
    ic_method: ICMethods = "AIC",
    f: int = 20,
    D_required: bool = False,
    A_stability: bool = False,
):
    y = np.atleast_2d(y)
    u = np.atleast_2d(u)
    min_ord = min(orders)
    l_, L = y.shape
    m_, L = u.shape
    if not check_types(0.0, np.nan, np.nan, f):
        return (
            np.array([[0.0]]),
            np.array([[0.0]]),
            np.array([[0.0]]),
            np.array([[0.0]]),
            np.inf,
            [],
            [],
            [],
            [],
        )
    else:
        if min_ord < 1:
            warn("The minimum model order will be set to 1")
            min_ord = 1
        max_ord = max(orders) + 1
        if f < min_ord:
            warn(
                "The horizon must be larger than the model order, min_order set as f"
            )
            min_ord = f
        if f < max_ord - 1:
            warn(
                "The horizon must be larger than the model order, max_order set as f"
            )
            max_ord = f + 1
        IC_old = np.inf
        N = L - 2 * f + 1
        U_std = np.zeros(m_)
        Ystd = np.zeros(l_)
        for j in range(m_):
            U_std[j], u[j] = rescale(u[j])
        for j in range(l_):
            Ystd[j], y[j] = rescale(y[j])
        U_n, S_n, V_n, W1, O_i = SVD_weighted(y, u, f, l_, weights)
        for i in range(min_ord, max_ord):
            Ob, X_fd, M, n, residuals = algorithm_1(
                y, u, l_, m_, f, N, U_n, S_n, V_n, W1, O_i, 0.0, i, D_required
            )
            if A_stability:
                M, residuals[0:n, :], ForcedA = forcing_A_stability(
                    M, n, Ob, l_, X_fd, N, u, f
                )
                if ForcedA:
                    print("at n=", n)
                    print("--------------------")
            A, B, C, D = extracting_matrices(M, n)
            Covariances = np.dot(residuals, residuals.T) / (N - 1)
            X_states, Y_estimate = SS_lsim_process_form(A, B, C, D, u)

            Vn = Vn_mat(y, Y_estimate)

            K_par = n * l_ + m_ * n
            if D_required:
                K_par = K_par + l_ * m_
            IC = information_criterion(K_par, L, Vn, ic_method)
            if IC < IC_old:
                n_min = i
                IC_old = IC
        print("The suggested order is: n=", n_min)
        Ob, X_fd, M, n, residuals = algorithm_1(
            y, u, l_, m_, f, N, U_n, S_n, V_n, W1, O_i, 0.0, n_min, D_required
        )
        if A_stability:
            M, residuals[0:n, :], _ = forcing_A_stability(
                M, n, Ob, l_, X_fd, N, u, f
            )
        A, B, C, D = extracting_matrices(M, n)
        Covariances = np.dot(residuals, residuals.T) / (N - 1)
        X_states, Y_estimate = SS_lsim_process_form(A, B, C, D, u)

        Vn = Vn_mat(y, Y_estimate)

        Q = Covariances[0:n, 0:n]
        R = Covariances[n::, n::]
        S = Covariances[0:n, n::]
        K, K_calculated = K_calc(A, C, Q, R, S)
        for j in range(m_):
            B[:, j] = B[:, j] / U_std[j]
            D[:, j] = D[:, j] / U_std[j]
        for j in range(l_):
            C[j, :] = C[j, :] * Ystd[j]
            D[j, :] = D[j, :] * Ystd[j]
            if K_calculated:
                K[:, j] = K[:, j] / Ystd[j]
        return A, B, C, D, Vn, Q, R, S, K
