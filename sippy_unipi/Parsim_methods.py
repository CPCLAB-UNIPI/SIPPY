"""Created on Sat Nov 04 2017

@author: Giuseppe Armenise
"""

import sys

import control.matlab as cnt
import numpy as np
import scipy as sc

from .functionset import information_criterion, rescale
from .functionsetSIM import (
    SS_lsim_predictor_form,
    SS_lsim_process_form,
    Vn_mat,
    Z_dot_PIort,
    check_inputs,
    check_types,
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


def simulations_sequence(A_K, C, L, y, u, l_, m, n, D_required):
    y_sim = []
    if D_required:
        n_simulations = n * m + l_ * m + n * l_ + n
        vect = np.zeros((n_simulations, 1))
        for i in range(n_simulations):
            vect[i, 0] = 1.0
            B_K = vect[0 : n * m, :].reshape((n, m))
            D = vect[n * m : n * m + l_ * m, :].reshape((l_, m))
            K = vect[n * m + l_ * m : n * m + l_ * m + n * l_, :].reshape(
                (n, l_)
            )
            x0 = vect[n * m + l_ * m + n * l_ : :, :].reshape((n, 1))
            y_sim.append(
                (
                    SS_lsim_predictor_form(A_K, B_K, C, D, K, y, u, x0)[1]
                ).reshape((1, L * l_))
            )
            vect[i, 0] = 0.0
    else:
        D = np.zeros((l_, m))
        n_simulations = n * m + n * l_ + n
        vect = np.zeros((n_simulations, 1))
        for i in range(n_simulations):
            vect[i, 0] = 1.0
            B_K = vect[0 : n * m, :].reshape((n, m))
            K = vect[n * m : n * m + n * l_, :].reshape((n, l_))
            x0 = vect[n * m + n * l_ : :, :].reshape((n, 1))
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


def simulations_sequence_S(A_K, C, L, K, y, u, l_, m, n, D_required):
    y_sim = []
    if D_required:
        n_simulations = n * m + l_ * m + n
        vect = np.zeros((n_simulations, 1))
        for i in range(n_simulations):
            vect[i, 0] = 1.0
            B_K = vect[0 : n * m, :].reshape((n, m))
            D = vect[n * m : n * m + l_ * m, :].reshape((l_, m))
            x0 = vect[n * m + l_ * m : :, :].reshape((n, 1))
            y_sim.append(
                (
                    SS_lsim_predictor_form(A_K, B_K, C, D, K, y, u, x0)[1]
                ).reshape((1, L * l_))
            )
            vect[i, 0] = 0.0
    else:
        n_simulations = n * m + n
        vect = np.zeros((n_simulations, 1))
        D = np.zeros((l_, m))
        for i in range(n_simulations):
            vect[i, 0] = 1.0
            B_K = vect[0 : n * m, :].reshape((n, m))
            x0 = vect[n * m : :, :].reshape((n, 1))
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


def PARSIM_K(
    y,
    u,
    f=20,
    p=20,
    threshold=0.1,
    max_order=np.nan,
    fixed_order=np.nan,
    D_required=False,
    B_recalc=False,
):
    y = 1.0 * np.atleast_2d(y)
    u = 1.0 * np.atleast_2d(u)
    l_, L = y.shape
    m = u[:, 0].size
    if not check_types(threshold, max_order, fixed_order, f, p):
        return (
            np.array([[0.0]]),
            np.array([[0.0]]),
            np.array([[0.0]]),
            np.array([[0.0]]),
            np.array([[0.0]]),
            np.array([[0.0]]),
            np.array([[0.0]]),
            np.array([[0.0]]),
            np.inf,
        )
    else:
        threshold, max_order = check_inputs(
            threshold, max_order, fixed_order, f
        )
        # N = L - f - p + 1
        Ustd = np.zeros(m)
        Ystd = np.zeros(l_)
        for j in range(m):
            Ustd[j], u[j] = rescale(u[j])
        for j in range(l_):
            Ystd[j], y[j] = rescale(y[j])
        Yf, Yp = ordinate_sequence(y, f, p)
        Uf, Up = ordinate_sequence(u, f, p)
        Zp = impile(Up, Yp)
        M = np.dot(Yf[0:l_, :], np.linalg.pinv(impile(Zp, Uf[0:m, :])))
        Matrix_pinv = np.linalg.pinv(
            impile(Zp, impile(Uf[0:m, :], Yf[0:l_, :]))
        )
        Gamma_L = M[:, 0 : (m + l_) * f]
        H_K = M[:, (m + l_) * f : :]
        G_K = np.zeros((l_, l_))
        for i in range(1, f):
            y_tilde = estimating_y(H_K, Uf, G_K, Yf, i, m, l_)
            M = np.dot((Yf[l_ * i : l_ * (i + 1)] - y_tilde), Matrix_pinv)
            H_K = impile(H_K, M[:, (m + l_) * f : (m + l_) * f + m])
            G_K = impile(G_K, M[:, (m + l_) * f + m : :])
            Gamma_L = impile(Gamma_L, (M[:, 0 : (m + l_) * f]))
        U_n, S_n, V_n = SVD_weighted_K(Uf, Zp, Gamma_L)
        U_n, S_n, V_n = reducingOrder(U_n, S_n, V_n, threshold, max_order)
        n = S_n.size  # states number
        S_n = np.diag(S_n)
        Ob_K = np.dot(U_n, sc.linalg.sqrtm(S_n))
        A_K = np.dot(np.linalg.pinv(Ob_K[0 : l_ * (f - 1), :]), Ob_K[l_::, :])
        C = Ob_K[0:l_, :]
        y_sim = simulations_sequence(A_K, C, L, y, u, l_, m, n, D_required)
        vect = np.dot(np.linalg.pinv(y_sim), y.reshape((L * l_, 1)))
        Y_estimate = np.dot(y_sim, vect)
        Vn = Vn_mat(y.reshape((L * l_, 1)), Y_estimate)
        B_K = vect[0 : n * m, :].reshape((n, m))
        if D_required:
            D = vect[n * m : n * m + l_ * m, :].reshape((l_, m))
            K = vect[n * m + l_ * m : n * m + l_ * m + n * l_, :].reshape(
                (n, l_)
            )
            x0 = vect[n * m + l_ * m + n * l_ : :, :].reshape((n, 1))
        else:
            D = np.zeros((l_, m))
            K = vect[n * m : n * m + n * l_, :].reshape((n, l_))
            x0 = vect[n * m + n * l_ : :, :].reshape((n, 1))
        A = A_K + np.dot(K, C)
        if B_recalc:
            y_sim = recalc_K(A, C, D, u)
            vect = np.dot(np.linalg.pinv(y_sim), y.reshape((L * l_, 1)))
            Y_estimate = np.dot(y_sim, vect)
            Vn = Vn_mat(y.reshape((L * l_, 1)), Y_estimate)
            B = vect[0 : n * m, :].reshape((n, m))
            x0 = vect[n * m : :, :].reshape((n, 1))
            B_K = B - np.dot(K, D)
        for j in range(m):
            B_K[:, j] = B_K[:, j] / Ustd[j]
            D[:, j] = D[:, j] / Ustd[j]
        for j in range(l_):
            K[:, j] = K[:, j] / Ystd[j]
            C[j, :] = C[j, :] * Ystd[j]
            D[j, :] = D[j, :] * Ystd[j]
        B = B_K + np.dot(K, D)
        return A_K, C, B_K, D, K, A, B, x0, Vn


def select_order_PARSIM_K(
    y,
    u,
    f=20,
    p=20,
    method="AIC",
    orders=[1, 10],
    D_required=False,
    B_recalc=False,
):
    y = 1.0 * np.atleast_2d(y)
    u = 1.0 * np.atleast_2d(u)
    min_ord = min(orders)
    l_, L = y.shape
    m, L = u.shape
    if not check_types(0.0, np.nan, np.nan, f, p):
        return (
            np.array([[0.0]]),
            np.array([[0.0]]),
            np.array([[0.0]]),
            np.array([[0.0]]),
            np.array([[0.0]]),
            np.array([[0.0]]),
            np.array([[0.0]]),
            np.array([[0.0]]),
            np.inf,
        )
    else:
        if min_ord < 1:
            sys.stdout.write("\033[0;35m")
            print("Warning: The minimum model order will be setted to 1")
            sys.stdout.write(" ")
            min_ord = 1
        max_ord = max(orders) + 1
        if f < min_ord:
            sys.stdout.write("\033[0;35m")
            print(
                "Warning! The horizon must be larger than the model order, min_order setted as f"
            )
            sys.stdout.write(" ")
            min_ord = f
        if f < max_ord - 1:
            sys.stdout.write("\033[0;35m")
            print(
                "Warning! The horizon must be larger than the model order, max_order setted as f"
            )
            sys.stdout.write(" ")
            max_ord = f + 1
        IC_old = np.inf
        # N = L - f - p + 1
        Ustd = np.zeros(m)
        Ystd = np.zeros(l_)
        for j in range(m):
            Ustd[j], u[j] = rescale(u[j])
        for j in range(l_):
            Ystd[j], y[j] = rescale(y[j])
        Yf, Yp = ordinate_sequence(y, f, p)
        Uf, Up = ordinate_sequence(u, f, p)
        Zp = impile(Up, Yp)
        M = np.dot(Yf[0:l_, :], np.linalg.pinv(impile(Zp, Uf[0:m, :])))
        Matrix_pinv = np.linalg.pinv(
            impile(Zp, impile(Uf[0:m, :], Yf[0:l_, :]))
        )
        Gamma_L = M[:, 0 : (m + l_) * f]
        H_K = M[:, (m + l_) * f : :]
        G_K = np.zeros((l_, l_))
        for i in range(1, f):
            y_tilde = estimating_y(H_K, Uf, G_K, Yf, i, m, l_)
            M = np.dot((Yf[l_ * i : l_ * (i + 1)] - y_tilde), Matrix_pinv)
            H_K = impile(H_K, M[:, (m + l_) * f : (m + l_) * f + m])
            G_K = impile(G_K, M[:, (m + l_) * f + m : :])
            Gamma_L = impile(Gamma_L, (M[:, 0 : (m + l_) * f]))
        U_n0, S_n0, V_n0 = SVD_weighted_K(Uf, Zp, Gamma_L)
        for i in range(min_ord, max_ord):
            U_n, S_n, V_n = reducingOrder(U_n0, S_n0, V_n0, 0.0, i)
            n = S_n.size
            S_n = np.diag(S_n)
            Ob_K = np.dot(U_n, sc.linalg.sqrtm(S_n))
            A_K = np.dot(
                np.linalg.pinv(Ob_K[0 : l_ * (f - 1), :]), Ob_K[l_::, :]
            )
            C = Ob_K[0:l_, :]
            y_sim = simulations_sequence(A_K, C, L, y, u, l_, m, n, D_required)
            vect = np.dot(np.linalg.pinv(y_sim), y.reshape((L * l_, 1)))
            Y_estimate = np.dot(y_sim, vect)
            Vn = Vn_mat(y.reshape((L * l_, 1)), Y_estimate)
            K_par = 2 * n * l_ + m * n
            if D_required:
                K_par = K_par + l_ * m
            IC = information_criterion(K_par, L, Vn, method)
            if IC < IC_old:
                n_min = i
                IC_old = IC
        print("The suggested order is: n=", n_min)
        U_n, S_n, V_n = reducingOrder(U_n0, S_n0, V_n0, 0.0, n_min)
        n = S_n.size
        S_n = np.diag(S_n)
        Ob_K = np.dot(U_n, sc.linalg.sqrtm(S_n))
        A_K = np.dot(np.linalg.pinv(Ob_K[0 : l_ * (f - 1), :]), Ob_K[l_::, :])
        C = Ob_K[0:l_, :]
        y_sim = simulations_sequence(A_K, C, L, y, u, l_, m, n, D_required)
        vect = np.dot(np.linalg.pinv(y_sim), y.reshape((L * l_, 1)))
        Y_estimate = np.dot(y_sim, vect)
        Vn = Vn_mat(y.reshape((L * l_, 1)), Y_estimate)
        B_K = vect[0 : n * m, :].reshape((n, m))
        if D_required:
            D = vect[n * m : n * m + l_ * m, :].reshape((l_, m))
            K = vect[n * m + l_ * m : n * m + l_ * m + n * l_, :].reshape(
                (n, l_)
            )
            x0 = vect[n * m + l_ * m + n * l_ : :, :].reshape((n, 1))
        else:
            D = np.zeros((l_, m))
            K = vect[n * m : n * m + n * l_, :].reshape((n, l_))
            x0 = vect[n * m + n * l_ : :, :].reshape((n, 1))
        A = A_K + np.dot(K, C)
        if B_recalc:
            y_sim = recalc_K(A, C, D, u)
            vect = np.dot(np.linalg.pinv(y_sim), y.reshape((L * l_, 1)))
            Y_estimate = np.dot(y_sim, vect)
            Vn = Vn_mat(y.reshape((L * l_, 1)), Y_estimate)
            B = vect[0 : n * m, :].reshape((n, m))
            x0 = vect[n * m : :, :].reshape((n, 1))
            B_K = B - np.dot(K, D)
        for j in range(m):
            B_K[:, j] = B_K[:, j] / Ustd[j]
            D[:, j] = D[:, j] / Ustd[j]
        for j in range(l_):
            K[:, j] = K[:, j] / Ystd[j]
            C[j, :] = C[j, :] * Ystd[j]
            D[j, :] = D[j, :] * Ystd[j]
        B = B_K + np.dot(K, D)
        return A_K, C, B_K, D, K, A, B, x0, Vn


def PARSIM_S(
    y,
    u,
    f=20,
    p=20,
    threshold=0.1,
    max_order=np.nan,
    fixed_order=np.nan,
    D_required=False,
):
    y = 1.0 * np.atleast_2d(y)
    u = 1.0 * np.atleast_2d(u)
    l_, L = y.shape
    m = u[:, 0].size
    if not check_types(threshold, max_order, fixed_order, f, p):
        return (
            np.array([[0.0]]),
            np.array([[0.0]]),
            np.array([[0.0]]),
            np.array([[0.0]]),
            np.array([[0.0]]),
            np.array([[0.0]]),
            np.array([[0.0]]),
            np.array([[0.0]]),
            np.inf,
        )
    else:
        threshold, max_order = check_inputs(
            threshold, max_order, fixed_order, f
        )
        # N = L - f - p + 1
        Ustd = np.zeros(m)
        Ystd = np.zeros(l_)
        for j in range(m):
            Ustd[j], u[j] = rescale(u[j])
        for j in range(l_):
            Ystd[j], y[j] = rescale(y[j])
        Yf, Yp = ordinate_sequence(y, f, p)
        Uf, Up = ordinate_sequence(u, f, p)
        Zp = impile(Up, Yp)
        Matrix_pinv = np.linalg.pinv(impile(Zp, Uf[0:m, :]))
        M = np.dot(Yf[0:l_, :], Matrix_pinv)
        Gamma_L = M[:, 0 : (m + l_) * f]
        H = M[:, (m + l_) * f : :]
        for i in range(1, f):
            y_tilde = estimating_y_S(H, Uf, Yf, i, m, l_)
            M = np.dot((Yf[l_ * i : l_ * (i + 1)] - y_tilde), Matrix_pinv)
            Gamma_L = impile(Gamma_L, (M[:, 0 : (m + l_) * f]))
            H = impile(H, M[:, (m + l_) * f : :])
        U_n, S_n, V_n = SVD_weighted_K(Uf, Zp, Gamma_L)
        U_n, S_n, V_n = reducingOrder(U_n, S_n, V_n, threshold, max_order)
        A, C, A_K, K, n = AK_C_estimating_S_P(
            U_n, S_n, V_n, l_, f, m, Zp, Uf, Yf
        )
        y_sim = simulations_sequence_S(
            A_K, C, L, K, y, u, l_, m, n, D_required
        )
        vect = np.dot(np.linalg.pinv(y_sim), y.reshape((L * l_, 1)))
        Y_estimate = np.dot(y_sim, vect)
        Vn = Vn_mat(y.reshape((L * l_, 1)), Y_estimate)
        B_K = vect[0 : n * m, :].reshape((n, m))
        if D_required:
            D = vect[n * m : n * m + l_ * m, :].reshape((l_, m))
            x0 = vect[n * m + l_ * m : :, :].reshape((n, 1))
        else:
            D = np.zeros((l_, m))
            x0 = vect[n * m : :, :].reshape((n, 1))
        for j in range(m):
            B_K[:, j] = B_K[:, j] / Ustd[j]
            D[:, j] = D[:, j] / Ustd[j]
        for j in range(l_):
            K[:, j] = K[:, j] / Ystd[j]
            C[j, :] = C[j, :] * Ystd[j]
            D[j, :] = D[j, :] * Ystd[j]
        B = B_K + np.dot(K, D)
        return A_K, C, B_K, D, K, A, B, x0, Vn


def select_order_PARSIM_S(
    y, u, f=20, p=20, method="AIC", orders=[1, 10], D_required=False
):
    y = 1.0 * np.atleast_2d(y)
    u = 1.0 * np.atleast_2d(u)
    min_ord = min(orders)
    l_, L = y.shape
    m, _ = u.shape
    if not check_types(0.0, np.nan, np.nan, f, p):
        return (
            np.array([[0.0]]),
            np.array([[0.0]]),
            np.array([[0.0]]),
            np.array([[0.0]]),
            np.array([[0.0]]),
            np.array([[0.0]]),
            np.array([[0.0]]),
            np.array([[0.0]]),
            np.inf,
        )
    else:
        if min_ord < 1:
            sys.stdout.write("\033[0;35m")
            print("Warning: The minimum model order will be setted to 1")
            sys.stdout.write(" ")
            min_ord = 1
        max_ord = max(orders) + 1
        if f < min_ord:
            sys.stdout.write("\033[0;35m")
            print(
                "Warning! The horizon must be larger than the model order, min_order setted as f"
            )
            sys.stdout.write(" ")
            min_ord = f
        if f < max_ord - 1:
            sys.stdout.write("\033[0;35m")
            print(
                "Warning! The horizon must be larger than the model order, max_order setted as f"
            )
            sys.stdout.write(" ")
            max_ord = f + 1
        IC_old = np.inf
        # N = L - f - p + 1
        Ustd = np.zeros(m)
        Ystd = np.zeros(l_)
        for j in range(m):
            Ustd[j], u[j] = rescale(u[j])
        for j in range(l_):
            Ystd[j], y[j] = rescale(y[j])
        Yf, Yp = ordinate_sequence(y, f, p)
        Uf, Up = ordinate_sequence(u, f, p)
        Zp = impile(Up, Yp)
        Matrix_pinv = np.linalg.pinv(impile(Zp, Uf[0:m, :]))
        M = np.dot(Yf[0:l_, :], Matrix_pinv)
        Gamma_L = M[:, 0 : (m + l_) * f]
        H = M[:, (m + l_) * f : :]
        for i in range(1, f):
            y_tilde = estimating_y_S(H, Uf, Yf, i, m, l_)
            M = np.dot((Yf[l_ * i : l_ * (i + 1)] - y_tilde), Matrix_pinv)
            Gamma_L = impile(Gamma_L, (M[:, 0 : (m + l_) * f]))
            H = impile(H, M[:, (m + l_) * f : :])
        U_n0, S_n0, V_n0 = SVD_weighted_K(Uf, Zp, Gamma_L)
        for i in range(min_ord, max_ord):
            U_n, S_n, V_n = reducingOrder(U_n0, S_n0, V_n0, 0.0, i)
            A, C, A_K, K, n = AK_C_estimating_S_P(
                U_n, S_n, V_n, l_, f, m, Zp, Uf, Yf
            )
            y_sim = simulations_sequence_S(
                A_K, C, L, K, y, u, l_, m, n, D_required
            )
            vect = np.dot(np.linalg.pinv(y_sim), y.reshape((L * l_, 1)))
            Y_estimate = np.dot(y_sim, vect)
            Vn = Vn_mat(y.reshape((L * l_, 1)), Y_estimate)
            K_par = 2 * n * l_ + m * n
            if D_required:
                K_par = K_par + l_ * m
            IC = information_criterion(K_par, L, Vn, method)
            if IC < IC_old:
                n_min = i
                IC_old = IC
        print("The suggested order is: n=", n_min)
        U_n, S_n, V_n = reducingOrder(U_n0, S_n0, V_n0, 0.0, n_min)
        A, C, A_K, K, n = AK_C_estimating_S_P(
            U_n, S_n, V_n, l_, f, m, Zp, Uf, Yf
        )
        y_sim = simulations_sequence_S(
            A_K, C, L, K, y, u, l_, m, n, D_required
        )
        vect = np.dot(np.linalg.pinv(y_sim), y.reshape((L * l_, 1)))
        Y_estimate = np.dot(y_sim, vect)
        Vn = Vn_mat(y.reshape((L * l_, 1)), Y_estimate)
        B_K = vect[0 : n * m, :].reshape((n, m))
        if D_required:
            D = vect[n * m : n * m + l_ * m, :].reshape((l_, m))
            x0 = vect[n * m + l_ * m : :, :].reshape((n, 1))
        else:
            D = np.zeros((l_, m))
            x0 = vect[n * m : :, :].reshape((n, 1))
        for j in range(m):
            B_K[:, j] = B_K[:, j] / Ustd[j]
            D[:, j] = D[:, j] / Ustd[j]
        for j in range(l_):
            K[:, j] = K[:, j] / Ystd[j]
            C[j, :] = C[j, :] * Ystd[j]
            D[j, :] = D[j, :] * Ystd[j]
        B = B_K + np.dot(K, D)
        return A_K, C, B_K, D, K, A, B, x0, Vn


def PARSIM_P(
    y,
    u,
    f=20,
    p=20,
    threshold=0.1,
    max_order=np.nan,
    fixed_order=np.nan,
    D_required=False,
):
    y = 1.0 * np.atleast_2d(y)
    u = 1.0 * np.atleast_2d(u)
    l_, L = y.shape
    m = u[:, 0].size
    if not check_types(threshold, max_order, fixed_order, f, p):
        return (
            np.array([[0.0]]),
            np.array([[0.0]]),
            np.array([[0.0]]),
            np.array([[0.0]]),
            np.array([[0.0]]),
            np.array([[0.0]]),
            np.array([[0.0]]),
            np.array([[0.0]]),
            np.inf,
        )
    else:
        threshold, max_order = check_inputs(
            threshold, max_order, fixed_order, f
        )
        # N = L - f - p + 1
        Ustd = np.zeros(m)
        Ystd = np.zeros(l_)
        for j in range(m):
            Ustd[j], u[j] = rescale(u[j])
        for j in range(l_):
            Ystd[j], y[j] = rescale(y[j])
        Yf, Yp = ordinate_sequence(y, f, p)
        Uf, Up = ordinate_sequence(u, f, p)
        Zp = impile(Up, Yp)
        Matrix_pinv = np.linalg.pinv(impile(Zp, Uf[0:m, :]))
        M = np.dot(Yf[0:l_, :], Matrix_pinv)
        Gamma_L = M[:, 0 : (m + l_) * f]
        for i in range(1, f):
            Matrix_pinv = np.linalg.pinv(impile(Zp, Uf[0 : m * (i + 1), :]))
            M = np.dot((Yf[l_ * i : l_ * (i + 1)]), Matrix_pinv)
            Gamma_L = impile(Gamma_L, (M[:, 0 : (m + l_) * f]))
        U_n, S_n, V_n = SVD_weighted_K(Uf, Zp, Gamma_L)
        U_n, S_n, V_n = reducingOrder(U_n, S_n, V_n, threshold, max_order)
        A, C, A_K, K, n = AK_C_estimating_S_P(
            U_n, S_n, V_n, l_, f, m, Zp, Uf, Yf
        )
        y_sim = simulations_sequence_S(
            A_K, C, L, K, y, u, l_, m, n, D_required
        )
        vect = np.dot(np.linalg.pinv(y_sim), y.reshape((L * l_, 1)))
        Y_estimate = np.dot(y_sim, vect)
        Vn = Vn_mat(y.reshape((L * l_, 1)), Y_estimate)
        B_K = vect[0 : n * m, :].reshape((n, m))
        if D_required:
            D = vect[n * m : n * m + l_ * m, :].reshape((l_, m))
            x0 = vect[n * m + l_ * m : :, :].reshape((n, 1))
        else:
            D = np.zeros((l_, m))
            x0 = vect[n * m : :, :].reshape((n, 1))
        for j in range(m):
            B_K[:, j] = B_K[:, j] / Ustd[j]
            D[:, j] = D[:, j] / Ustd[j]
        for j in range(l_):
            K[:, j] = K[:, j] / Ystd[j]
            C[j, :] = C[j, :] * Ystd[j]
            D[j, :] = D[j, :] * Ystd[j]
        B = B_K + np.dot(K, D)
        return A_K, C, B_K, D, K, A, B, x0, Vn


def select_order_PARSIM_P(
    y, u, f=20, p=20, method="AIC", orders=[1, 10], D_required=False
):
    y = 1.0 * np.atleast_2d(y)
    u = 1.0 * np.atleast_2d(u)
    min_ord = min(orders)
    l_, L = y.shape
    m, _ = u.shape
    if not check_types(0.0, np.nan, np.nan, f, p):
        return (
            np.array([[0.0]]),
            np.array([[0.0]]),
            np.array([[0.0]]),
            np.array([[0.0]]),
            np.array([[0.0]]),
            np.array([[0.0]]),
            np.array([[0.0]]),
            np.array([[0.0]]),
            np.inf,
        )
    else:
        if min_ord < 1:
            sys.stdout.write("\033[0;35m")
            print("Warning: The minimum model order will be set to 1")
            sys.stdout.write(" ")
            min_ord = 1
        max_ord = max(orders) + 1
        if f < min_ord:
            sys.stdout.write("\033[0;35m")
            print(
                "Warning! The horizon must be larger than the model order, min_order set as f"
            )
            sys.stdout.write(" ")
            min_ord = f
        if f < max_ord - 1:
            sys.stdout.write("\033[0;35m")
            print(
                "Warning! The horizon must be larger than the model order, max_order set as f"
            )
            sys.stdout.write(" ")
            max_ord = f + 1
        IC_old = np.inf
        # N = L - f - p + 1
        Ustd = np.zeros(m)
        Ystd = np.zeros(l_)
        for j in range(m):
            Ustd[j], u[j] = rescale(u[j])
        for j in range(l_):
            Ystd[j], y[j] = rescale(y[j])
        Yf, Yp = ordinate_sequence(y, f, p)
        Uf, Up = ordinate_sequence(u, f, p)
        Zp = impile(Up, Yp)
        Matrix_pinv = np.linalg.pinv(impile(Zp, Uf[0:m, :]))
        M = np.dot(Yf[0:l_, :], Matrix_pinv)
        Gamma_L = M[:, 0 : (m + l_) * f]
        for i in range(1, f):
            Matrix_pinv = np.linalg.pinv(impile(Zp, Uf[0 : m * (i + 1), :]))
            M = np.dot((Yf[l_ * i : l_ * (i + 1)]), Matrix_pinv)
            Gamma_L = impile(Gamma_L, (M[:, 0 : (m + l_) * f]))
        U_n0, S_n0, V_n0 = SVD_weighted_K(Uf, Zp, Gamma_L)
        for i in range(min_ord, max_ord):
            U_n, S_n, V_n = reducingOrder(U_n0, S_n0, V_n0, 0.0, i)
            A, C, A_K, K, n = AK_C_estimating_S_P(
                U_n, S_n, V_n, l_, f, m, Zp, Uf, Yf
            )
            y_sim = simulations_sequence_S(
                A_K, C, L, K, y, u, l_, m, n, D_required
            )
            vect = np.dot(np.linalg.pinv(y_sim), y.reshape((L * l_, 1)))
            Y_estimate = np.dot(y_sim, vect)
            Vn = Vn_mat(y.reshape((L * l_, 1)), Y_estimate)
            K_par = 2 * n * l_ + m * n
            if D_required:
                K_par = K_par + l_ * m
            IC = information_criterion(K_par, L, Vn, method)
            if IC < IC_old:
                n_min = i
                IC_old = IC
        print("The suggested order is: n=", n_min)
        U_n, S_n, V_n = reducingOrder(U_n0, S_n0, V_n0, 0.0, n_min)
        A, C, A_K, K, n = AK_C_estimating_S_P(
            U_n, S_n, V_n, l_, f, m, Zp, Uf, Yf
        )
        y_sim = simulations_sequence_S(
            A_K, C, L, K, y, u, l_, m, n, D_required
        )
        vect = np.dot(np.linalg.pinv(y_sim), y.reshape((L * l_, 1)))
        Y_estimate = np.dot(y_sim, vect)
        Vn = Vn_mat(y.reshape((L * l_, 1)), Y_estimate)
        B_K = vect[0 : n * m, :].reshape((n, m))
        if D_required:
            D = vect[n * m : n * m + l_ * m, :].reshape((l_, m))
            x0 = vect[n * m + l_ * m : :, :].reshape((n, 1))
        else:
            D = np.zeros((l_, m))
            x0 = vect[n * m : :, :].reshape((n, 1))
        for j in range(m):
            B_K[:, j] = B_K[:, j] / Ustd[j]
            D[:, j] = D[:, j] / Ustd[j]
        for j in range(l_):
            K[:, j] = K[:, j] / Ystd[j]
            C[j, :] = C[j, :] * Ystd[j]
            D[j, :] = D[j, :] * Ystd[j]
        B = B_K + np.dot(K, D)
        return A_K, C, B_K, D, K, A, B, x0, Vn


# creating object SS model
class SS_PARSIM_model:
    def __init__(self, A, B, C, D, K, A_K, B_K, x0, ts, Vn):
        self.n = A[:, 0].size
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.Vn = Vn
        self.K = K
        self.G = cnt.ss(A, B, C, D, ts)
        self.ts = ts
        self.x0 = x0
        self.A_K = A_K
        self.B_K = B_K
