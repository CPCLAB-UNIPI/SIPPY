# -*- coding: utf-8 -*-
"""
Created on Sun Oct 08 2017

@author: Giuseppe Armenise
"""
from __future__ import absolute_import, print_function
import control.matlab as cnt
import math
from .functionset import *
# from functionset import *



def ordinate_sequence(y, f, p):
    [l, L] = y.shape
    N = L - p - f + 1
    Yp = np.zeros((l * f, N))
    Yf = np.zeros((l * f, N))
    for i in range(1, f + 1):
        Yf[l * (i - 1):l * i] = y[:, p + i - 1:L - f + i]
        Yp[l * (i - 1):l * i] = y[:, i - 1:L - f - p + i]
    return Yf, Yp

def Z_dot_PIort(z, X):
    """
    Compute the scalar product between a vector z and $I - x^T \cdot pinv(X^T)$, avoiding the direct computation of the matrix
    
    PI = np.dot(X.T, np.linalg.pinv(X.T)), causing high memory usage
    
    
    Parameters
    ----------
    z : (...) vector array_like
    
    X : (...) matrix array_like
  
    """
        
    Z_dot_PIort = (z - np.dot(np.dot(z, X.T), np.linalg.pinv(X.T)))
    return Z_dot_PIort

def Vn_mat(y,yest):
    """
    Compute the variance of the model residuals
    
    Parameters
    ----------
    y : (L*l,1) vectorized matrix of output of the process
    
    yest : (L*l,1) vectorized matrix of output of the estimated model
  
    """ 
    y = y.flatten()    
    yest = yest.flatten()
    eps = y - yest
    Vn = (eps@eps)/(max(y.shape))   # @ is dot
    return Vn
    

def impile(M1, M2):
    M = np.zeros((M1[:, 0].size + M2[:, 0].size, M1[0, :].size))
    M[0:M1[:, 0].size] = M1
    M[M1[:, 0].size::] = M2
    return M


def reducingOrder(U_n, S_n, V_n, threshold=0.1, max_order=10):
    s0 = S_n[0]
    index = S_n.size
    for i in range(S_n.size):
        if S_n[i] < threshold * s0 or i >= max_order:
            index = i
            break
    return U_n[:, 0:index], S_n[0:index], V_n[0:index, :]


def check_types(threshold, max_order, fixed_order, f, p=20):
    if threshold < 0. or threshold >= 1.:
        print("Error! The threshold value must be >=0. and <1.")
        return False
    if (np.isnan(max_order)) == False:
        if type(max_order) != int:
            print("Error! The max_order value must be integer")
            return False
    if (np.isnan(fixed_order)) == False:
        if type(fixed_order) != int:
            print("Error! The fixed_order value must be integer")
            return False
    if type(f) != int:
        print("Error! The future horizon (f) must be integer")
        return False
    if type(p) != int:
        print("Error! The past horizon (p) must be integer")
        return False
    return True


def check_inputs(threshold, max_order, fixed_order, f):
    if (math.isnan(fixed_order)) == False:
        threshold = 0.0
        max_order = fixed_order
    if f < max_order:
        print('Warning! The horizon must be larger than the model order, max_order setted as f')
    if (max_order < f) == False:
        max_order = f
    return threshold, max_order


def SS_lsim_process_form(A, B, C, D, u, x0='None'):
    m, L = u.shape
    l, n = C.shape
    y = np.zeros((l, L))
    x = np.zeros((n, L))
    if type(x0) != str:
        x[:, 0] = x0[:, 0]
    y[:, 0] = np.dot(C, x[:, 0]) + np.dot(D, u[:, 0])
    for i in range(1, L):
        x[:, i] = np.dot(A, x[:, i - 1]) + np.dot(B, u[:, i - 1])
        y[:, i] = np.dot(C, x[:, i]) + np.dot(D, u[:, i])
    return x, y


def SS_lsim_predictor_form(A_K, B_K, C, D, K, y, u, x0='None'):
    m, L = u.shape
    l, n = C.shape
    y_hat = np.zeros((l, L))
    x = np.zeros((n, L + 1))
    if type(x0) != str:
        x[:, 0] = x0[:, 0]
    for i in range(0, L):
        x[:, i + 1] = np.dot(A_K, x[:, i]) + np.dot(B_K, u[:, i]) + np.dot(K, y[:, i])
        y_hat[:, i] = np.dot(C, x[:, i]) + np.dot(D, u[:, i])
    return x, y_hat


def SS_lsim_innovation_form(A, B, C, D, K, y, u, x0='None'):
    m, L = u.shape
    l, n = C.shape
    y_hat = np.zeros((l, L))
    x = np.zeros((n, L + 1))
    if type(x0) != str:
        x[:, 0] = x0[:, 0]
    for i in range(0, L):
        y_hat[:, i] = np.dot(C, x[:, i]) + np.dot(D, u[:, i])
        x[:, i + 1] = np.dot(A, x[:, i]) + np.dot(B, u[:, i]) + np.dot(K, y[:, i] - y_hat[:, i])
    return x, y_hat


def K_calc(A, C, Q, R, S):
    n_A = A[0, :].size
    try:
        P, L, G = cnt.dare(A.T, C.T, Q, R, S, np.identity(n_A))
        K = np.dot(np.dot(A, P), C.T) + S
        K = np.dot(K, np.linalg.inv(np.dot(np.dot(C, P), C.T) + R))
        Calculated = True
    except:
        K = []
        print("Kalman filter cannot be calculated")
        Calculated = False
    return K, Calculated
