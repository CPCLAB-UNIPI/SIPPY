# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 2017

@author: Giuseppe Armenise
"""
from __future__ import absolute_import, division, print_function

import sys
from builtins import object

import scipy as sc
from numpy.linalg import pinv  

from .functionsetSIM import *


def SVD_weighted(y, u, f, l, weights='N4SID'):
    Yf, Yp = ordinate_sequence(y, f, f)
    Uf, Up = ordinate_sequence(u, f, f)
    Zp = impile(Up, Yp)
    
    YfdotPIort_Uf = Z_dot_PIort(Yf,Uf)
    ZpdotPIort_Uf = Z_dot_PIort(Zp,Uf)
    O_i = np.dot(np.dot(YfdotPIort_Uf,pinv(ZpdotPIort_Uf)),Zp) 

    if weights == 'MOESP':
        W1 = None
        #        W2 = PIort_Uf
        OidotPIort_Uf = Z_dot_PIort(O_i, Uf)  # np.dot(O_i, W2)
        U_n, S_n, V_n = np.linalg.svd(OidotPIort_Uf, full_matrices=False)
                
    elif weights == 'CVA':        
        W1 = np.linalg.inv(sc.linalg.sqrtm(np.dot(YfdotPIort_Uf,YfdotPIort_Uf.T)).real)
        W1dotOi = np.dot(W1, O_i)
        W1_dot_Oi_dot_PIort_Uf = Z_dot_PIort(W1dotOi,Uf) 
        U_n, S_n, V_n = np.linalg.svd(W1_dot_Oi_dot_PIort_Uf, full_matrices=False)  
   
    elif weights == 'N4SID':
        W1 = None # is identity
        W2 = None # not used in 'N4SID'
        U_n, S_n, V_n = np.linalg.svd(O_i,full_matrices=False) #full matrices not used
    
    return U_n, S_n, V_n, W1, O_i


def algorithm_1(y, u, l, m, f, N, U_n, S_n, V_n, W1, O_i, threshold, max_order, D_required):
    U_n, S_n, V_n = reducingOrder(U_n, S_n, V_n, threshold, max_order)
    V_n = V_n.T
    n = S_n.size
    S_n = np.diag(S_n)
    if W1 is None: #W1 is identity
        Ob = np.dot(U_n, sc.linalg.sqrtm(S_n))
    else:
        Ob = np.dot(np.linalg.inv(W1), np.dot(U_n, sc.linalg.sqrtm(S_n)))
    X_fd = np.dot(np.linalg.pinv(Ob), O_i)
    Sxterm = impile(X_fd[:, 1:N], y[:, f:f + N - 1])
    Dxterm = impile(X_fd[:, 0:N - 1], u[:, f:f + N - 1])
    if D_required == True:
        M = np.dot(Sxterm, np.linalg.pinv(Dxterm))
    else:
        M = np.zeros((n + l, n + m))
        M[0:n, :] = np.dot(Sxterm[0:n], np.linalg.pinv(Dxterm))
        M[n::, 0:n] = np.dot(Sxterm[n::], np.linalg.pinv(Dxterm[0:n, :]))
    residuals = Sxterm - np.dot(M, Dxterm)
    return Ob, X_fd, M, n, residuals


def forcing_A_stability(M, n, Ob, l, X_fd, N, u, f):
    Forced_A = False
    if np.max(np.abs(np.linalg.eigvals(M[0:n, 0:n]))) >= 1.:
        Forced_A = True
        print("Forcing A stability")
        M[0:n, 0:n] = np.dot(np.linalg.pinv(Ob), impile(Ob[l::, :], np.zeros((l, n))))
        M[0:n, n::] = np.dot(X_fd[:, 1:N] - np.dot(M[0:n, 0:n], X_fd[:, 0:N - 1]),
                             np.linalg.pinv(u[:, f:f + N - 1]))
    res = X_fd[:, 1:N] - np.dot(M[0:n, 0:n], X_fd[:, 0:N - 1]) - np.dot(M[0:n, n::],
                                                                        u[:, f:f + N - 1])
    return M, res, Forced_A


def extracting_matrices(M, n):
    A = M[0:n, 0:n]
    B = M[0:n, n::]
    C = M[n::, 0:n]
    D = M[n::, n::]
    return A, B, C, D


def OLSims(y, u, f, weights='N4SID', threshold=0.1, max_order=np.NaN, fixed_order=np.NaN,
           D_required=False, A_stability=False):
    y = 1. * np.atleast_2d(y)
    u = 1. * np.atleast_2d(u)
    l, L = y.shape
    m = u[:, 0].size
    if check_types(threshold, max_order, fixed_order, f) == False:
        return np.array([[0.0]]), np.array([[0.0]]), np.array([[0.0]]), np.array(
                [[0.0]]), np.inf, [], [], [], []
    else:
        threshold, max_order = check_inputs(threshold, max_order, fixed_order, f)
        N = L - 2 * f + 1
        Ustd = np.zeros(m)
        Ystd = np.zeros(l)
        for j in range(m):
            Ustd[j], u[j] = rescale(u[j])
        for j in range(l):
            Ystd[j], y[j] = rescale(y[j])
        U_n, S_n, V_n, W1, O_i = SVD_weighted(y, u, f, l, weights)
        Ob, X_fd, M, n, residuals = algorithm_1(y, u, l, m, f, N, U_n, S_n, V_n, W1, O_i, threshold,
                                                max_order, D_required)
        if A_stability == True:
            M, residuals[0:n, :], useless = forcing_A_stability(M, n, Ob, l, X_fd, N, u, f)
        A, B, C, D = extracting_matrices(M, n)
        Covariances = old_div(np.dot(residuals, residuals.T), (N - 1))
        Q = Covariances[0:n, 0:n]
        R = Covariances[n::, n::]
        S = Covariances[0:n, n::]
        X_states, Y_estimate = SS_lsim_process_form(A, B, C, D, u)
                
        Vn = Vn_mat(y, Y_estimate)
        
        K, K_calculated = K_calc(A, C, Q, R, S)
        for j in range(m):
            B[:, j] = old_div(B[:, j], Ustd[j])
            D[:, j] = old_div(D[:, j], Ustd[j])
        for j in range(l):
            C[j, :] = C[j, :] * Ystd[j]
            D[j, :] = D[j, :] * Ystd[j]
            if K_calculated == True:
                K[:, j] = old_div(K[:, j], Ystd[j])
        return A, B, C, D, Vn, Q, R, S, K


def select_order_SIM(y, u, f=20, weights='N4SID', method='AIC', orders=[1, 10], D_required=False,
                     A_stability=False):
    y = 1. * np.atleast_2d(y)
    u = 1. * np.atleast_2d(u)
    min_ord = min(orders)
    l, L = y.shape
    m, L = u.shape
    if check_types(0.0, np.NaN, np.NaN, f) == False:
        return np.array([[0.0]]), np.array([[0.0]]), np.array([[0.0]]), np.array(
                [[0.0]]), np.inf, [], [], [], []
    else:
        if min_ord < 1:
            sys.stdout.write("\033[0;35m")
            print("Warning: The minimum model order will be set to 1");
            sys.stdout.write(" ")
            min_ord = 1
        max_ord = max(orders) + 1
        if f < min_ord:
            sys.stdout.write("\033[0;35m")
            print('Warning! The horizon must be larger than the model order, min_order set as f');
            sys.stdout.write(" ")
            min_ord = f
        if f < max_ord - 1:
            sys.stdout.write("\033[0;35m")
            print('Warning! The horizon must be larger than the model order, max_order set as f');
            sys.stdout.write(" ")
            max_ord = f + 1
        IC_old = np.inf
        N = L - 2 * f + 1
        Ustd = np.zeros(m)
        Ystd = np.zeros(l)
        for j in range(m):
            Ustd[j], u[j] = rescale(u[j])
        for j in range(l):
            Ystd[j], y[j] = rescale(y[j])
        U_n, S_n, V_n, W1, O_i = SVD_weighted(y, u, f, l, weights)
        for i in range(min_ord, max_ord):
            Ob, X_fd, M, n, residuals = algorithm_1(y, u, l, m, f, N, U_n, S_n, V_n, W1, O_i, 0.0,
                                                    i, D_required)
            if A_stability == True:
                M, residuals[0:n, :], ForcedA = forcing_A_stability(M, n, Ob, l, X_fd, N, u, f)
                if ForcedA == True:
                    print("at n=", n)
                    print("--------------------")
            A, B, C, D = extracting_matrices(M, n)
            Covariances = old_div(np.dot(residuals, residuals.T), (N - 1))
            X_states, Y_estimate = SS_lsim_process_form(A, B, C, D, u)

            Vn = Vn_mat(y, Y_estimate)

            K_par = n * l + m * n
            if D_required == True:
                K_par = K_par + l * m
            IC = information_criterion(K_par, L, Vn, method)
            if IC < IC_old:
                n_min = i
                IC_old = IC
        print("The suggested order is: n=", n_min)
        Ob, X_fd, M, n, residuals = algorithm_1(y, u, l, m, f, N, U_n, S_n, V_n, W1, O_i, 0.0,
                                                n_min, D_required)
        if A_stability == True:
            M, residuals[0:n, :], useless = forcing_A_stability(M, n, Ob, l, X_fd, N, u, f)
        A, B, C, D = extracting_matrices(M, n)
        Covariances = old_div(np.dot(residuals, residuals.T), (N - 1))
        X_states, Y_estimate = SS_lsim_process_form(A, B, C, D, u)
 
        Vn = Vn_mat(y, Y_estimate)
 
        Q = Covariances[0:n, 0:n]
        R = Covariances[n::, n::]
        S = Covariances[0:n, n::]
        K, K_calculated = K_calc(A, C, Q, R, S)
        for j in range(m):
            B[:, j] = old_div(B[:, j], Ustd[j])
            D[:, j] = old_div(D[:, j], Ustd[j])
        for j in range(l):
            C[j, :] = C[j, :] * Ystd[j]
            D[j, :] = D[j, :] * Ystd[j]
            if K_calculated == True:
                K[:, j] = old_div(K[:, j], Ystd[j])
        return A, B, C, D, Vn, Q, R, S, K


# creating object SS model
class SS_model(object):
    def __init__(self, A, B, C, D, K, Q, R, S, ts, Vn):
        self.n = A[:, 0].size
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.Vn = Vn
        self.Q = Q
        self.R = R
        self.S = S
        self.K = K
        self.G = cnt.ss(A, B, C, D, ts)
        self.ts = ts
        self.x0 = np.zeros((A[:, 0].size, 1))
        try:
            A_K = A - np.dot(K, C)
            B_K = B - np.dot(K, D)
        except:
            A_K = []
            B_K = []
        self.A_K = A_K
        self.B_K = B_K
