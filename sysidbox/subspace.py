# -*- coding: utf-8 -*-
"""
Created on 2017

@author: Giuseppe Armenise

@updates: Riccardo Bacci di Capaci, Marco Vaccari, Federico Pelagagge
"""
from __future__ import print_function

import sys
from builtins import range
from .OLSims_methods import *
import numpy as np
## SIPPY package: main file


def system_identification(y, u, id_method, centering = 'None', IC = 'None', \
                            tsample = 1., SS_f = 20, SS_threshold = 0.1, \
                            SS_max_order = np.NaN, SS_fixed_order = np.NaN, \
                            SS_orders = [1, 10], SS_D_required = False, SS_A_stability = False):
    
    y = 1. * np.atleast_2d(y)
    u = 1. * np.atleast_2d(u)
    [n1, n2] = y.shape
    ydim = min(n1, n2)
    ylength = max(n1, n2)
    if ylength == n1:
        y = y.T
    [n1, n2] = u.shape
    ulength = max(n1, n2)
    udim = min(n1, n2)
    if ulength == n1:
        u = u.T
        
    # Checking data consinstency    
    if ulength != ylength:
        sys.stdout.write("\033[0;35m")
        print(
            "Warning! y and u lengths are not the same. The minor value between the two lengths has been chosen. The perfomed indentification may be not correct, be sure to check your input and output data alignement")
        sys.stdout.write(" ")
        # Recasting data cutting out the over numbered data
        minlength = min(ulength, ylength)
        y = y[:, :minlength]
        u = u[:, :minlength]

    # Data centering
    if centering == 'InitVal':
        y_rif = 1. * y[:, 0]
        u_init = 1. * u[:, 0]
        for i in range(ylength):
            y[:, i] = y[:, i] - y_rif
            u[:, i] = u[:, i] - u_init
    elif centering == 'MeanVal':
        y_rif = np.zeros(ydim)
        u_mean = np.zeros(udim)
        for i in range(ydim):
            y_rif[i] = np.mean(y[i, :])
        for i in range(udim):
            u_mean[i] = np.mean(u[i, :])
        for i in range(ylength):
            y[:, i] = y[:, i] - y_rif
            u[:, i] = u[:, i] - u_mean
    elif centering == 'None':
        y_rif = 0. * y[:, 0]       
    else:
    # elif centering != 'None':
        sys.stdout.write("\033[0;35m")
        print("Warning! \'Centering\' argument is not valid, its value has been reset to \'None\'")
        sys.stdout.write(" ")
        
    # Defining default values for orders
    na = [0]*ydim; nb = [0]*ydim; nc = [0]*ydim; nd= [0]*ydim; nf = [0]*ydim 
    
    ##### Check Information Criterion #####
        
    ### MODE 1) fixed orders
    if (IC == 'AIC' or IC == 'AICc' or IC == 'BIC') == False:
        # if none IC is selected
        
        # if something goes wrong
        if IC != 'None':
            sys.stdout.write("\033[0;35m")
            print(
                "Warning, no correct information criterion selected, its value has been reset to \'None\'")
            sys.stdout.write(" ")
               
        ###### MODEL choice
 
        ## SS MODELS
            
        # N4SID-MOESP-CVA    
        if id_method == 'N4SID' or id_method == 'MOESP' or id_method == 'CVA':
            A, B, C, D, Vn, Q, R, S, K = OLSims(y, u, SS_f, id_method, SS_threshold,
                                                               SS_max_order, SS_fixed_order,
                                                               SS_D_required, SS_A_stability)
            model = SS_model(A, B, C, D, K, Q, R, S, tsample, Vn)
            
        # NO method selected
        else:
            sys.exit("Error! No identification method selected")

    
    ##### 
    ### MODE 2) order range
    # if an IC is selected
    else:
        
        # method choice
        ## SS-MODELS
            
        ## N4SID-MOESP-CVA
        if id_method == 'N4SID' or id_method == 'MOESP' or id_method == 'CVA':
            A, B, C, D, Vn, Q, R, S, K = select_order_SIM(y, u, SS_f, id_method, IC,
                                                                         SS_orders, SS_threshold,
                                                                         SS_D_required, SS_A_stability)
            model = SS_model(A, B, C, D, K, Q, R, S, tsample, Vn)
        
        # NO method selected
        else:
            sys.exit("Error! No identification method selected")
            model = 'None'

    return model

# Data recentering
def data_recentering(y,y_rif,ylength):  
    for i in range(ylength):
        y[:, i] = y[:, i] + y_rif
    return y