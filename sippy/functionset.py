# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 2017

@author: Giuseppe Armenise
"""
from __future__ import division, print_function

import sys
from builtins import range

import numpy as np
from past.utils import old_div

import control.matlab as cnt


# function which generates a sequence of inputs GBN
# N: sequence length (total number of samples)
# p_swd: desired probability of switching (no switch: 0<x<1 :always switch)
# Nmin: minimum number of samples between two switches
# Range: input range
# Tol: tolerance on switching probability relative error
# nit_max: maximum number of iterations
def GBN_seq(N, p_swd, Nmin=1, Range=[-1.0, 1.0], Tol = 0.01, nit_max = 30):
    min_Range = min(Range)
    max_Range = max(Range)
    prob = np.random.random()
    # set first value
    if prob < 0.5:
        gbn = -1.0*np.ones(N)
    else:
        gbn = 1.0*np.ones(N)
    # init. variables
    p_sw = p_sw_b = 2.0             # actual switch probability
    nit = 0; 
    while (np.abs(p_sw - p_swd))/p_swd > Tol and nit <= nit_max:
        i_fl = 0; Nsw = 0;
        for i in range(N - 1):
            gbn[i + 1] = gbn[i]
            # test switch probability
            if (i - i_fl >= Nmin):
                prob = np.random.random()
                # track last test of p_sw
                i_fl = i
                if (prob < p_swd):
                    # switch and then count it
                    gbn[i + 1] = -gbn[i + 1]
                    Nsw = Nsw + 1
        # check actual switch probability
        p_sw = Nmin*(Nsw+1)/N; #print("p_sw", p_sw);
        # set best iteration
        if np.abs(p_sw - p_swd) < np.abs(p_sw_b - p_swd):
            p_sw_b = p_sw
            Nswb = Nsw
            gbn_b = gbn.copy()
        # increase iteration number
        nit = nit + 1; #print("nit", nit)
    # rescale GBN
    for i in range(N):
        if gbn_b[i] > 0.:
            gbn_b[i] = max_Range
        else:
            gbn_b[i] = min_Range
    return gbn_b, p_sw_b, Nswb

# function which generates a sequence of inputs as Random walk
# N: sequence length (total number of samples);
# sigma: standard deviation (mobility) of randow walk
# rw0: initial value    
def RW_seq(N,rw0, sigma = 1):
    rw = rw0*np.ones(N)
    for i in range (N-1):
        # return random sample from a normal (Gaussian) distribution with:
        # mean = 0.0, standard deviation = sigma, and length = 1
        delta = np.random.normal(0., sigma, 1)
        # refresh input
        rw[i+1] = rw[i] + delta
    return rw 


# this function adds a white noise to a signal
# y:clean signal
# A_rel: relative amplitude (0<x<1) to the standard deviation of y (example: 0.05)
# noise amplitude=  A_rel*(standard deviation of y)
def white_noise(y, A_rel):
    num = y.size
    errors = np.zeros(num)
    y_err = np.zeros(num)
    Ystd = np.std(y)
    scale = np.abs(A_rel * Ystd)
    if scale < np.finfo(np.float32).eps:
        scale = np.finfo(np.float32).eps
        sys.stdout.write("\033[0;35m")
        print("Warning: A_rel may be too small, its value set to the lowest default one");
        sys.stdout.write(" ")
    errors = np.random.normal(0., scale, num)
    y_err = y + errors
    return errors, y_err


# this function generates a white noise matrix (rows with zero mean), L:size (columns), Var: variance vector
# e.g.   noise=white_noise_var(100,[1,1]) , noise matrix has two row vectors with variance=1
def white_noise_var(L, Var):
    Var = np.array(Var)
    n = Var.size
    noise = np.zeros((n, L))
    for i in range(n):
        if Var[i] < np.finfo(np.float32).eps:
            Var[i] = np.finfo(np.float32).eps
            sys.stdout.write("\033[0;35m")
            print("Warning: Var[", i,
                  "] may be too small, its value set to the lowest default one");
            sys.stdout.write(" ")
        noise[i, :] = np.random.normal(0., Var[i] ** 0.5, L)
    return noise


# rescaling an array to its standard deviation. It gives the array rescaled: y=y/(standard deviation of y)
# and thestandard deviation: ex [Ystd,Y]=rescale(Y)
def rescale(y):
    ystd = np.std(y)
    y_scaled = old_div(y, ystd)
    return ystd, y_scaled


def information_criterion(K, N, Variance, method='AIC'):
    if method == 'AIC':
        IC = N * np.log(Variance) + 2 * K
    elif method == 'AICc':
        if N - K - 1 > 0:
            IC = N * np.log(Variance) + 2 * K + 2 * K * (K + 1) / (N - K - 1)
        else:
            IC = np.inf
            sys.exit('Number of data is less than the number of parameters, AICc cannot be applied')
    elif method == 'BIC':
        IC = N * np.log(Variance) + K * np.log(N)
    return IC


def mean_square_error(predictions, targets):
    return ((predictions - targets) ** 2).mean()

# Function for model validation (one-step and k-step ahead predictor)
# SYS: system to validate (identified ARX or ARMAX model)     
# u: input data
# y: output data
# Time: time sequence
# k: k-step ahead     
def validation(SYS,u,y,Time, k = 1, centering = 'None'):
    # check dimensions
    y = 1. * np.atleast_2d(y)
    u = 1. * np.atleast_2d(u)
    [n1, n2] = y.shape
    ydim = min(n1, n2)
    ylength = max(n1, n2)
    if ylength == n1:
        y = y.T
    [n1, n2] = u.shape
    udim = min(n1, n2)
    ulength = max(n1, n2)  
    if ulength == n1:
        u = u.T
    
    Yval = np.zeros((ydim,ylength))
    
    # Data centering
    #y_rif = np.zeros(ydim)
    #u_rif = np.zeros(udim)
    if centering == 'InitVal':
        y_rif = 1. * y[:, 0]
        u_rif = 1. * u[:, 0]
    elif centering == 'MeanVal':
        for i in range(ydim):
            y_rif = np.mean(y,1)
            #y_rif[i] = np.mean(y[i, :])
        for i in range(udim):
            u_rif = np.mean(u,1)
            #u_rif[i] = np.mean(u[i, :])
    elif centering == 'None':
        y_rif = np.zeros(ydim)
        u_rif = np.zeros(udim)
    else:
    # elif centering != 'None':
        sys.stdout.write("\033[0;35m")
        print("Warning! \'Centering\' argument is not valid, its value has been reset to \'None\'")
        sys.stdout.write(" ")   
    
    # MISO approach  
    # centering inputs and outputs
    for i in range(u.shape[0]):
        u[i,:] = u[i,:] - u_rif[i]
    for i in range(ydim):
        # one-step ahead predictor 
        if k == 1:
            Y_u, T, Xv = cnt.lsim((1/SYS.H[i,0])*SYS.G[i,:], u, Time)
            Y_y, T, Xv = cnt.lsim(1 - (1/SYS.H[i,0]), y[i,:] - y_rif[i], Time)
            #Yval[i,:] = np.atleast_2d(Y_u + Y_y + y_rif[i])
            Yval[i,:] = (Y_u.T + np.atleast_2d(Y_y) + y_rif[i])
        else:
        # k-step ahead predictor
            # impulse response of disturbance model H
            hout, T = cnt.impulse(SYS.H[i,0], Time)
            # extract first k-1 coefficients
            h_k_num = hout[0:k]
            # set denumerator
            h_k_den = np.hstack((np.ones((1,1)), np.zeros((1,k-1))))
            # FdT of impulse response
            Hk = cnt.tf(h_k_num,h_k_den[0],SYS.ts)
            # plot impulse response
            # plt.subplot(ydim,1,i+1)
            # plt.plot(Time,hout)
            # plt.grid()
            # if i == 0:
            #     plt.title('Impulse Response')
            # plt.ylabel("h_j ")
            # plt.xlabel("Time [min]")
            # k-step ahead prediction
            Y_u, T, Xv = cnt.lsim(Hk*(1/SYS.H[i,0])*SYS.G[i,:], u, Time)
            #Y_y, T, Xv = cnt.lsim(1 - Hk*(1/SYS.H[i,0]), y[i,:] - y[i,0], Time)
            Y_y, T, Xv = cnt.lsim(1 - Hk*(1/SYS.H[i,0]), y[i,:] - y_rif[i], Time)
            #Yval[i,:] = np.atleast_2d(Y_u + Y_y + y[i,0])
            Yval[i,:] = np.atleast_2d(Y_u + Y_y + y_rif[i]) 
      
    return Yval



    
