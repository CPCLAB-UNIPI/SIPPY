# -*- coding: utf-8 -*-
"""
Created on 2017

@author: Giuseppe Armenise

@updates: Riccardo Bacci di Capaci, Marco Vaccari, Federico Pelagagge
"""
from __future__ import print_function

import sys
from builtins import range

import numpy as np

## SIPPY package: main file


def system_identification(y, u, id_method, centering='None', IC='None', \
                          tsample=1., ARX_orders=[1, 1, 0], ARMAX_orders=[1, 1, 1, 0], \
                          na_ord=[0, 5], nb_ord=[1, 5], nc_ord=[0, 5], delays=[0, 5], \
                          ARMAX_max_iterations=100, SS_f=20, SS_p=20, SS_threshold=0.1, \
                          SS_max_order=np.NaN, SS_fixed_order=np.NaN, \
                          SS_orders=[1, 10], SS_D_required=False, SS_A_stability=False, \
                          SS_PK_B_reval=False):
    
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
        
    
    # Check Information Criterion
    if (IC == 'AIC' or IC == 'AICc' or IC == 'BIC') == False:
        # if none IC is selected
        
        # if something goes wrong
        if IC != 'None':
            sys.stdout.write("\033[0;35m")
            print(
                "Warning, no correct information criterion selected, its value has been reset to \'None\'")
            sys.stdout.write(" ")
               
        # method choice
        
        ## ARX    
        if id_method == 'ARX':
            
            # not 3 inputs
            if len(ARX_orders) != 3:
                sys.exit("Error! ARX identification takes three arguments in ARX_orders")
                model = 'None'
                
            # assigned orders   
            if (type(ARX_orders[0]) == list and type(ARX_orders[1]) == list and type(ARX_orders[2]) == list):
                na = ARX_orders[0]
                nb = ARX_orders[1]
                theta = ARX_orders[2]

            # not assigned orders (read default)    
            elif (type(ARX_orders[0]) == int and type(ARX_orders[1]) == int and type(ARX_orders[2]) == int):
                na = (ARX_orders[0] * np.ones((ydim,), dtype=np.int)).tolist()
                nb = (ARX_orders[1] * np.ones((ydim, udim), dtype=np.int)).tolist()
                theta = (ARX_orders[2] * np.ones((ydim, udim), dtype=np.int)).tolist()
            
            # something goes worong
            else:
                sys.exit(
                    "Error! ARX_orders must be a list containing three lists or three integers")
                model = 'None'
            
            # id ARX
            from . import arxMIMO
            #import arxMIMO
            # id ARX MIMO (also SISO case)
            DENOMINATOR, NUMERATOR, G, H, Vn_tot, Yid = arxMIMO.ARX_MIMO_id(y, u, na, nb, theta,tsample)
            # recentering
            Yid = data_recentering(Yid,y_rif,ylength) 
            # form model
            model = arxMIMO.ARX_MIMO_model(na, nb, theta, tsample, NUMERATOR, DENOMINATOR, G, H, Vn_tot, Yid)
                
        ## ARMAX        
        elif id_method == 'ARMAX':
            
            # not 4 inputs
            if len(ARMAX_orders) != 4:
                sys.exit("Error! ARMAX identification takes four arguments in ARMAX_orders")

            # assigned orders    
            if (type(ARMAX_orders[0]) == list and type(ARMAX_orders[1]) == list and type(ARMAX_orders[2]) == list and type(ARMAX_orders[3]) == list):
                na = ARMAX_orders[0]
                nb = ARMAX_orders[1]
                nc = ARMAX_orders[2]
                theta = ARMAX_orders[3]
            
            # not assigned orders (read default)   
            elif (type(ARMAX_orders[0]) == int and type(ARMAX_orders[1]) == int and type(ARMAX_orders[2]) == int and type(ARMAX_orders[3]) == int):
                na = (ARMAX_orders[0] * np.ones((ydim,), dtype=np.int)).tolist()
                nb = (ARMAX_orders[1] * np.ones((ydim, udim), dtype=np.int)).tolist()
                nc = (ARMAX_orders[2] * np.ones((ydim,), dtype=np.int)).tolist()
                theta = (ARMAX_orders[3] * np.ones((ydim, udim), dtype=np.int)).tolist()
            
            # something goes worong
            else:
                sys.exit(
                    "Error! ARMAX_orders must be a list containing four lists or four integers")
            
            # id ARMAX
            from . import armaxMIMO
            #import armaxMIMO
            # id ARX MIMO (also SISO case)
            DENOMINATOR, NUMERATOR, DENOMINATOR_H, NUMERATOR_H, G, H, Vn_tot, Yid = armaxMIMO.ARMAX_MIMO_id(
                    y, u, na, nb, nc, theta, tsample, ARMAX_max_iterations)
            # recentering
            Yid = data_recentering(Yid,y_rif,ylength)
            # form model
            model = armaxMIMO.ARMAX_MIMO_model(na, nb, nc, theta, tsample,
                                               NUMERATOR, DENOMINATOR, NUMERATOR_H, DENOMINATOR_H, G, H, Vn_tot, Yid)

        ## SS methods
            
        # N4SID-MOESP-CVA    
        elif id_method == 'N4SID' or id_method == 'MOESP' or id_method == 'CVA':
            from . import OLSims_methods
            A, B, C, D, Vn, Q, R, S, K = OLSims_methods.OLSims(y, u, SS_f, id_method, SS_threshold,
                                                               SS_max_order, SS_fixed_order,
                                                               SS_D_required, SS_A_stability)
            model = OLSims_methods.SS_model(A, B, C, D, K, Q, R, S, tsample, Vn)
            
        # PARSIM-K
        elif id_method == 'PARSIM-K':
            from . import Parsim_methods
            A_K, C, B_K, D, K, A, B, x0, Vn = Parsim_methods.PARSIM_K(y, u, SS_f, SS_p,
                                                                      SS_threshold, SS_max_order,
                                                                      SS_fixed_order, SS_D_required,
                                                                      SS_PK_B_reval)
            model = Parsim_methods.SS_PARSIM_model(A, B, C, D, K, A_K, B_K, x0, tsample,Vn)
        
        # PARSIM-S  
        elif id_method == 'PARSIM-S':
            from . import Parsim_methods
            A_K, C, B_K, D, K, A, B, x0, Vn = Parsim_methods.PARSIM_S(y, u, SS_f, SS_p,
                                                                      SS_threshold, SS_max_order,
                                                                      SS_fixed_order, SS_D_required)
            model = Parsim_methods.SS_PARSIM_model(A, B, C, D, K, A_K, B_K, x0, tsample,Vn)
        
        # PARSIM-P
        elif id_method == 'PARSIM-P':
            from . import Parsim_methods
            A_K, C, B_K, D, K, A, B, x0, Vn = Parsim_methods.PARSIM_P(y, u, SS_f, SS_p,
                                                                      SS_threshold, SS_max_order,
                                                                      SS_fixed_order, SS_D_required)
            model = Parsim_methods.SS_PARSIM_model(A, B, C, D, K, A_K, B_K, x0, tsample,Vn)
        
        # NO method selected
        else:
            sys.exit("Error! No identification method selected")

    # if an IC is selected
    else:
        
        # method choice
        
        ## ARX or ARMAX (MIMO case --> not implemented)
        if (id_method == 'ARX' or id_method == 'ARMAX'):
            if (ydim != 1 or udim != 1):
                sys.exit(
                    "Error! Information criteria are implemented in SISO case for ARX and ARMAX model sets.  Use subspace methods instead for MIMO cases")
        
        ## ARX        
        if id_method == 'ARX':
            from . import arx
            # import arx
            na, nb, theta, g_identif, h_identif, NUMERATOR, DENOMINATOR, Vn, Yid = arx.select_order_ARX(y[0], u[0], tsample, na_ord, nb_ord, delays, IC)
            # recentering
            Yid = data_recentering(Yid,y_rif,ylength)
            model = arx.ARX_model(na, nb, theta, tsample, NUMERATOR, DENOMINATOR, g_identif, h_identif, Vn, Yid)
        
        ## ARMAX
        elif id_method == 'ARMAX':
            from . import armax
            # import armax
            # file updated by people external from CPCLAB
            model = armax.Armax(na_ord, nb_ord, nc_ord, delays, tsample, IC, ARMAX_max_iterations)
            #
            model.find_best_estimate(y[0], u[0])
            # recentering
            Yid = data_recentering(model.Yid,y_rif,ylength)

        ## N4SID-MOESP-CVA
        elif id_method == 'N4SID' or id_method == 'MOESP' or id_method == 'CVA':
            from . import OLSims_methods
            A, B, C, D, Vn, Q, R, S, K = OLSims_methods.select_order_SIM(y, u, SS_f, id_method, IC,
                                                                         SS_orders, SS_D_required,
                                                                         SS_A_stability)
            model = OLSims_methods.SS_model(A, B, C, D, K, Q, R, S, tsample, Vn)
        
        ## PARSIM-K
        elif id_method == 'PARSIM-K':
            from . import Parsim_methods
            A_K, C, B_K, D, K, A, B, x0, Vn = Parsim_methods.select_order_PARSIM_K(y, u, SS_f, SS_p,
                                                                                   IC, SS_orders,
                                                                                   SS_D_required,
                                                                                   SS_PK_B_reval)
            model = Parsim_methods.SS_PARSIM_model(A, B, C, D, K, A_K, B_K, x0, tsample,Vn)
        
        ## PARSIM-S
        elif id_method == 'PARSIM-S':
            from . import Parsim_methods
            A_K, C, B_K, D, K, A, B, x0, Vn = Parsim_methods.select_order_PARSIM_S(y, u, SS_f, SS_p,
                                                                                   IC, SS_orders,
                                                                                   SS_D_required)
            model = Parsim_methods.SS_PARSIM_model(A, B, C, D, K, A_K, B_K, x0, tsample,Vn)
        
        # PARSIM-P
        elif id_method == 'PARSIM-P':
            from . import Parsim_methods
            A_K, C, B_K, D, K, A, B, x0, Vn = Parsim_methods.select_order_PARSIM_P(y, u, SS_f, SS_p,
                                                                                   IC, SS_orders,
                                                                                   SS_D_required)
            model = Parsim_methods.SS_PARSIM_model(A, B, C, D, K, A_K, B_K, x0, tsample,Vn)
        
        # NO method selected
        else:
            sys.exit("Error! No identification method selected")

    return model

# Data recentering
def data_recentering(y,y_rif,ylength):  
    for i in range(ylength):
        y[:, i] = y[:, i] + y_rif
    return y