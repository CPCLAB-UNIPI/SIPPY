"""
Created on Sat Nov 04 2017

@author: Giuseppe Armenise
"""

from typing import Literal
from warnings import warn

import numpy as np
import scipy as sc

from .functionset import information_criterion, rescale
from .functionsetSIM import (
    SS_lsim_predictor_form,
    SS_lsim_process_form,
    Vn_mat,
    Z_dot_PIort,
    impile,
    ordinate_sequence,
    reducingOrder,
)


def recalc_K(A, C, D, u):
    y_sim = []
    n_ord = A[:, 0].size
    m_input, L = u.shape
    l_ = C[:, 0].size
    n_simulations = n_ord + n_ord * m_input
    vect = np.zeros((n_simulations, 1))
    for i in range(n_simulations):
        vect[i, 0] = 1.0
        B = vect[0 : n_ord * m_input, :].reshape((n_ord, m_input))
        x0 = vect[n_ord * m_input : :, :].reshape((n_ord, 1))
        y_sim.append(
            (SS_lsim_process_form(A, B, C, D, u, x0=x0)[1]).reshape(
                (1, L * l_)
            )
        )
        vect[i, 0] = 0.0
    y_matrix = 1.0 * y_sim[0]
    for j in range(n_simulations - 1):
        y_matrix = impile(y_matrix, y_sim[j + 1])
    y_matrix = y_matrix.T
    return y_matrix


def estimating_y(H_K, Uf, G_K, Yf, i, m, l_):
    y_tilde = np.dot(H_K[0:l_, :], Uf[m * (i) : m * (i + 1), :])
    for j in range(1, i):
        y_tilde = (
            y_tilde
            + np.dot(
                H_K[l_ * j : l_ * (j + 1), :],
                Uf[m * (i - j) : m * (i - j + 1), :],
            )
            + np.dot(
                G_K[l_ * j : l_ * (j + 1), :],
                Yf[l_ * (i - j) : l_ * (i - j + 1), :],
            )
        )
    return y_tilde


def estimating_y_S(H_K, Uf, Yf, i, m, l_):
    y_tilde = np.dot(H_K[0:l_, :], Uf[m * (i) : m * (i + 1), :])
    for j in range(1, i):
        y_tilde = y_tilde + np.dot(
            H_K[l_ * j : l_ * (j + 1), :], Uf[m * (i - j) : m * (i - j + 1), :]
        )
    return y_tilde


def SVD_weighted_K(Uf, Zp, Gamma_L):
    W2 = sc.linalg.sqrtm(np.dot(Z_dot_PIort(Zp, Uf), Zp.T)).real
    U_n, S_n, V_n = np.linalg.svd(np.dot(Gamma_L, W2), full_matrices=False)
    return U_n, S_n, V_n


def simulations_sequence(A_K, C, L, y, u, l_, m_, n, D_required):
    y_sim = []
    if D_required:
        n_simulations = n * m_ + l_ * m_ + n * l_ + n
        vect = np.zeros((n_simulations, 1))
        for i in range(n_simulations):
            vect[i, 0] = 1.0
            B_K = vect[0 : n * m_, :].reshape((n, m_))
            D = vect[n * m_ : n * m_ + l_ * m_, :].reshape((l_, m_))
            K = vect[n * m_ + l_ * m_ : n * m_ + l_ * m_ + n * l_, :].reshape(
                (n, l_)
            )
            x0 = vect[n * m_ + l_ * m_ + n * l_ : :, :].reshape((n, 1))
            y_sim.append(
                (
                    SS_lsim_predictor_form(A_K, B_K, C, D, K, y, u, x0)[1]
                ).reshape((1, L * l_))
            )
            vect[i, 0] = 0.0
    else:
        D = np.zeros((l_, m_))
        n_simulations = n * m_ + n * l_ + n
        vect = np.zeros((n_simulations, 1))
        for i in range(n_simulations):
            vect[i, 0] = 1.0
            B_K = vect[0 : n * m_, :].reshape((n, m_))
            K = vect[n * m_ : n * m_ + n * l_, :].reshape((n, l_))
            x0 = vect[n * m_ + n * l_ : :, :].reshape((n, 1))
            y_sim.append(
                (
                    SS_lsim_predictor_form(A_K, B_K, C, D, K, y, u, x0)[1]
                ).reshape((1, L * l_))
            )
            vect[i, 0] = 0.0
    y_matrix = 1.0 * y_sim[0]
    for j in range(n_simulations - 1):
        y_matrix = impile(y_matrix, y_sim[j + 1])
    y_matrix = y_matrix.T
    return y_matrix


def simulations_sequence_S(A_K, C, L, K, y, u, l_, m_, n, D_required):
    y_sim = []
    if D_required:
        n_simulations = n * m_ + l_ * m_ + n
        vect = np.zeros((n_simulations, 1))
        for i in range(n_simulations):
            vect[i, 0] = 1.0
            B_K = vect[0 : n * m_, :].reshape((n, m_))
            D = vect[n * m_ : n * m_ + l_ * m_, :].reshape((l_, m_))
            x0 = vect[n * m_ + l_ * m_ : :, :].reshape((n, 1))
            y_sim.append(
                (
                    SS_lsim_predictor_form(A_K, B_K, C, D, K, y, u, x0)[1]
                ).reshape((1, L * l_))
            )
            vect[i, 0] = 0.0
    else:
        n_simulations = n * m_ + n
        vect = np.zeros((n_simulations, 1))
        D = np.zeros((l_, m_))
        for i in range(n_simulations):
            vect[i, 0] = 1.0
            B_K = vect[0 : n * m_, :].reshape((n, m_))
            x0 = vect[n * m_ : :, :].reshape((n, 1))
            y_sim.append(
                (
                    SS_lsim_predictor_form(A_K, B_K, C, D, K, y, u, x0)[1]
                ).reshape((1, L * l_))
            )
            vect[i, 0] = 0.0
    y_matrix = 1.0 * y_sim[0]
    for j in range(n_simulations - 1):
        y_matrix = impile(y_matrix, y_sim[j + 1])
    y_matrix = y_matrix.T
    return y_matrix


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


def sim_observed_seq(y, u, f, D_required, l_, L, m, U_n, S_n):
    n = S_n.size
    S_n = np.diag(S_n)
    Ob_K = np.dot(U_n, sc.linalg.sqrtm(S_n))
    A_K = np.dot(np.linalg.pinv(Ob_K[0 : l_ * (f - 1), :]), Ob_K[l_::, :])
    C = Ob_K[0:l_, :]
    y_sim = simulations_sequence(A_K, C, L, y, u, l_, m, n, D_required)
    return y_sim, A_K, C


def parsim(
    mode: Literal["PARSIM-K", "PARSIM-S", "PARSIM-P"],
    y: np.ndarray,
    u: np.ndarray,
    order: int | tuple[int, int] = 0,
    threshold: float = 0.0,
    f: int = 20,
    p: int = 20,
    D_required: bool = False,
    B_recalc: bool = False,
    ic_method: Literal["AIC", "AICc", "BIC"] = "AIC",
):
    if isinstance(order, tuple):
        min_ord, max_ord = order[0], order[-1] + 1
        if min_ord < 1:
            warn("The minimum model order will be set to 1")
            min_ord = 1
        if f < min_ord:
            warn(
                f"The horizon must be larger than the model order, min_order set to {f}"
            )
            min_ord = f
        if f < max_ord - 1:
            warn(
                f"The horizon must be larger than the model order, max_order set to {f}"
            )
            max_ord = f + 1

    y = 1.0 * np.atleast_2d(y)
    u = 1.0 * np.atleast_2d(u)

    l_, L = y.shape
    m_ = u[:, 0].size
    Ustd = np.zeros(m_)
    Ystd = np.zeros(l_)
    for j in range(m_):
        Ustd[j], u[j] = rescale(u[j])
    for j in range(l_):
        Ystd[j], y[j] = rescale(y[j])
    Yf, Yp = ordinate_sequence(y, f, p)
    Uf, Up = ordinate_sequence(u, f, p)
    Zp = impile(Up, Yp)

    Gamma_L = compute_gamma_matrix(mode, f, l_, m_, Yf, Uf, Zp)

    U_n, S_n, V_n = SVD_weighted_K(Uf, Zp, Gamma_L)

    if isinstance(order, tuple):
        min_order = order[0]
        IC_old = np.inf
        for i in range(min_ord, max_ord):
            U_n, S_n, V_n = reducingOrder(U_n, S_n, V_n, threshold, i)
            if mode == "PARSIM-K":
                y_sim, A_K, C = sim_observed_seq(
                    y, u, f, D_required, l_, L, m_, U_n, S_n
                )

            else:
                A, C, A_K, K, n = AK_C_estimating_S_P(
                    U_n, S_n, V_n, l_, f, m_, Zp, Uf, Yf
                )
                y_sim = simulations_sequence_S(
                    A_K, C, L, K, y, u, l_, m_, n, D_required
                )
            vect = np.dot(np.linalg.pinv(y_sim), y.reshape((L * l_, 1)))
            Y_estimate = np.dot(y_sim, vect)
            Vn = Vn_mat(y.reshape((L * l_, 1)), Y_estimate)

            K_par = 2 * n * l_ + m_ * n
            if D_required:
                K_par = K_par + l_ * m_
            IC = information_criterion(K_par, L, Vn, ic_method)
            if IC < IC_old:
                min_order = i
                IC_old = IC

        order = min_order
        print("The suggested order is: n=", order)

    U_n, S_n, V_n = reducingOrder(U_n, S_n, V_n, threshold, order)
    n = S_n.size
    if mode == "PARSIM-K":
        y_sim, A_K, C = sim_observed_seq(
            y, u, f, D_required, l_, L, m_, U_n, S_n
        )

    else:
        A, C, A_K, K, n = AK_C_estimating_S_P(
            U_n, S_n, V_n, l_, f, m_, Zp, Uf, Yf
        )
        y_sim = simulations_sequence_S(
            A_K, C, L, K, y, u, l_, m_, n, D_required
        )

    vect = np.dot(np.linalg.pinv(y_sim), y.reshape((L * l_, 1)))
    Y_estimate = np.dot(y_sim, vect)
    Vn = Vn_mat(y.reshape((L * l_, 1)), Y_estimate)
    B_K = vect[0 : n * m_, :].reshape((n, m_))

    if D_required:
        D = vect[n * m_ : n * m_ + l_ * m_, :].reshape((l_, m_))
        if mode == "PARSIM-K":
            K = vect[n * m_ + l_ * m_ : n * m_ + l_ * m_ + n * l_, :].reshape(
                (n, l_)
            )
            x0 = vect[n * m_ + l_ * m_ + n * l_ : :, :].reshape((n, 1))
        else:
            x0 = vect[n * m_ + l_ * m_ : :, :].reshape((n, 1))
    else:
        D = np.zeros((l_, m_))
        if mode == "PARSIM-K":
            K = vect[n * m_ : n * m_ + n * l_, :].reshape((n, l_))
            x0 = vect[n * m_ + n * l_ : :, :].reshape((n, 1))
        else:
            x0 = vect[n * m_ : :, :].reshape((n, 1))

    if mode == "PARSIM-K":
        A = A_K + np.dot(K, C)
        if B_recalc:
            y_sim = recalc_K(A, C, D, u)
            vect = np.dot(np.linalg.pinv(y_sim), y.reshape((L * l_, 1)))
            Y_estimate = np.dot(y_sim, vect)
            Vn = Vn_mat(y.reshape((L * l_, 1)), Y_estimate)
            B = vect[0 : n * m_, :].reshape((n, m_))
            x0 = vect[n * m_ : :, :].reshape((n, 1))
            B_K = B - np.dot(K, D)

    for j in range(m_):
        B_K[:, j] = B_K[:, j] / Ustd[j]
        D[:, j] = D[:, j] / Ustd[j]
    for j in range(l_):
        K[:, j] = K[:, j] / Ystd[j]
        C[j, :] = C[j, :] * Ystd[j]
        D[j, :] = D[j, :] * Ystd[j]
    B = B_K + np.dot(K, D)
    return A_K, C, B_K, D, K, A, B, x0, Vn


def compute_gamma_matrix(mode, f, l_, m_, Yf, Uf, Zp):
    if mode == "PARSIM-K":
        M = np.dot(Yf[0:l_, :], np.linalg.pinv(impile(Zp, Uf[0:m_, :])))
        Matrix_pinv = np.linalg.pinv(
            impile(Zp, impile(Uf[0:m_, :], Yf[0:l_, :]))
        )
    else:
        Matrix_pinv = np.linalg.pinv(impile(Zp, Uf[0:m_, :]))
        M = np.dot(Yf[0:l_, :], Matrix_pinv)
    Gamma_L = M[:, 0 : (m_ + l_) * f]

    H = M[:, (m_ + l_) * f : :]
    G = np.zeros((l_, l_))
    for i in range(1, f):
        if mode == "PARSIM-K":
            y_tilde = estimating_y(H, Uf, G, Yf, i, m_, l_)
            M = np.dot((Yf[l_ * i : l_ * (i + 1)] - y_tilde), Matrix_pinv)
            H = impile(H, M[:, (m_ + l_) * f : (m_ + l_) * f + m_])
            G = impile(G, M[:, (m_ + l_) * f + m_ : :])
        elif mode == "PARSIM-S":
            y_tilde = estimating_y_S(H, Uf, Yf, i, m_, l_)
            M = np.dot((Yf[l_ * i : l_ * (i + 1)] - y_tilde), Matrix_pinv)
            H = impile(H, M[:, (m_ + l_) * f : :])
        elif mode == "PARSIM-P":
            Matrix_pinv = np.linalg.pinv(impile(Zp, Uf[0 : m_ * (i + 1), :]))
            M = np.dot((Yf[l_ * i : l_ * (i + 1)]), Matrix_pinv)
        Gamma_L = impile(Gamma_L, (M[:, 0 : (m_ + l_) * f]))
    return Gamma_L
