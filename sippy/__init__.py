# -*- coding: utf-8 -*-
"""
Created on Wed Dec 06 

@author: Giuseppe Armenise
"""
from __future__ import print_function

import sys
from builtins import range

import numpy as np


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
        y_init = 1. * y[:, 0]
        u_init = 1. * u[:, 0]
        for i in range(ylength):
            y[:, i] = y[:, i] - y_init
            u[:, i] = u[:, i] - u_init
    elif centering == 'MeanVal':
        y_mean = np.zeros(ydim)
        u_mean = np.zeros(udim)
        for i in range(ydim):
            y_mean[i] = np.mean(y[i, :])
        for i in range(udim):
            u_mean[i] = np.mean(u[i, :])
        for i in range(ylength):
            y[:, i] = y[:, i] - y_mean
            u[:, i] = u[:, i] - u_mean
    elif centering != 'None':
        sys.stdout.write("\033[0;35m")
        print("Warning! \'Centering\' argument is not valid, its value has been reset to \'None\'")
        sys.stdout.write(" ")

    if (IC == 'AIC' or IC == 'AICc' or IC == 'BIC') == False:
        if IC != 'None':
            sys.stdout.write("\033[0;35m")
            print(
                "Warning, no correct information criterion selected, its value has been reset to \'None\'")
            sys.stdout.write(" ")
        if id_method == 'ARX':
            if len(ARX_orders) != 3:
                sys.exit("Error! ARX identification takes three arguments in ARX_orders")
                model = 'None'
            if (type(ARX_orders[0]) == list and type(ARX_orders[1]) == list and type(
                    ARX_orders[2]) == list):
                na = ARX_orders[0]
                nb = ARX_orders[1]
                theta = ARX_orders[2]
                from . import arxMIMO
                DENOMINATOR, NUMERATOR, G, H, Vn_tot = arxMIMO.ARX_MIMO_id(y, u, na, nb, theta,
                                                                           tsample)
                model = arxMIMO.ARX_MIMO_model(na, nb, theta, tsample, NUMERATOR,
                                                           DENOMINATOR, G, H, Vn_tot)
            elif (type(ARX_orders[0]) == int and type(ARX_orders[1]) == int and type(
                    ARX_orders[2]) == int):
                na = (ARX_orders[0] * np.ones((ydim,), dtype=np.int)).tolist()
                nb = (ARX_orders[1] * np.ones((ydim, udim), dtype=np.int)).tolist()
                theta = (ARX_orders[2] * np.ones((ydim, udim), dtype=np.int)).tolist()
                from . import arxMIMO
                DENOMINATOR, NUMERATOR, G, H, Vn_tot = arxMIMO.ARX_MIMO_id(y, u, na, nb, theta,
                                                                           tsample)
                model = arxMIMO.ARX_MIMO_model(na, nb, theta, tsample, NUMERATOR,
                                                           DENOMINATOR, G, H, Vn_tot)
            else:
                sys.exit(
                    "Error! ARX_orders must be a list containing three lists or three integers")
                model = 'None'
        elif id_method == 'ARMAX':
            if len(ARMAX_orders) != 4:
                sys.exit("Error! ARMAX identification takes four arguments in ARMAX_orders")
            #                identified_system='None'
            if (type(ARMAX_orders[0]) == list and type(ARMAX_orders[1]) == list and type(
                    ARMAX_orders[2]) == list and type(ARMAX_orders[3]) == list):
                na = ARMAX_orders[0]
                nb = ARMAX_orders[1]
                nc = ARMAX_orders[2]
                theta = ARMAX_orders[3]
                from . import armaxMIMO
                DENOMINATOR, NUMERATOR, DENOMINATOR_H, NUMERATOR_H, G, H, Vn_tot = armaxMIMO.ARMAX_MIMO_id(
                    y, u, na, nb, nc, theta, tsample, ARMAX_max_iterations)
                model = armaxMIMO.ARMAX_MIMO_model(na, nb, nc, theta, tsample,
                                                               NUMERATOR, DENOMINATOR, NUMERATOR_H,
                                                               DENOMINATOR_H, G, H, Vn_tot)
            elif (type(ARMAX_orders[0]) == int and type(ARMAX_orders[1]) == int and type(
                    ARMAX_orders[2]) == int and type(ARMAX_orders[3]) == int):
                na = (ARMAX_orders[0] * np.ones((ydim,), dtype=np.int)).tolist()
                nb = (ARMAX_orders[1] * np.ones((ydim, udim), dtype=np.int)).tolist()
                nc = (ARMAX_orders[2] * np.ones((ydim,), dtype=np.int)).tolist()
                theta = (ARMAX_orders[3] * np.ones((ydim, udim), dtype=np.int)).tolist()
                from . import armaxMIMO
                DENOMINATOR, NUMERATOR, DENOMINATOR_H, NUMERATOR_H, G, H, Vn_tot = armaxMIMO.ARMAX_MIMO_id(
                    y, u, na, nb, nc, theta, tsample, ARMAX_max_iterations)
                model = armaxMIMO.ARMAX_MIMO_model(na, nb, nc, theta, tsample,
                                                               NUMERATOR, DENOMINATOR, NUMERATOR_H,
                                                               DENOMINATOR_H, G, H, Vn_tot)
            else:
                sys.exit(
                    "Error! ARMAX_orders must be a list containing four lists or four integers")
        #                identified_system='None'
        elif id_method == 'N4SID' or id_method == 'MOESP' or id_method == 'CVA':
            from . import OLSims_methods
            A, B, C, D, Vn, Q, R, S, K = OLSims_methods.OLSims(y, u, SS_f, id_method, SS_threshold,
                                                               SS_max_order, SS_fixed_order,
                                                               SS_D_required, SS_A_stability)
            model = OLSims_methods.SS_model(A, B, C, D, K, Q, R, S, tsample, Vn)
        elif id_method == 'PARSIM-K':
            from . import Parsim_methods
            A_K, C, B_K, D, K, A, B, x0, Vn = Parsim_methods.PARSIM_K(y, u, SS_f, SS_p,
                                                                      SS_threshold, SS_max_order,
                                                                      SS_fixed_order, SS_D_required,
                                                                      SS_PK_B_reval)
            model = Parsim_methods.SS_PARSIM_model(A, B, C, D, K, A_K, B_K, x0, tsample,
                                                               Vn)
        elif id_method == 'PARSIM-S':
            from . import Parsim_methods
            A_K, C, B_K, D, K, A, B, x0, Vn = Parsim_methods.PARSIM_S(y, u, SS_f, SS_p,
                                                                      SS_threshold, SS_max_order,
                                                                      SS_fixed_order, SS_D_required)
            model = Parsim_methods.SS_PARSIM_model(A, B, C, D, K, A_K, B_K, x0, tsample,
                                                               Vn)
        elif id_method == 'PARSIM-P':
            from . import Parsim_methods
            A_K, C, B_K, D, K, A, B, x0, Vn = Parsim_methods.PARSIM_P(y, u, SS_f, SS_p,
                                                                      SS_threshold, SS_max_order,
                                                                      SS_fixed_order, SS_D_required)
            model = Parsim_methods.SS_PARSIM_model(A, B, C, D, K, A_K, B_K, x0, tsample,
                                                               Vn)
        else:
            sys.exit("Error! No identification method selected")

    else:
        if (id_method == 'ARX' or id_method == 'ARMAX'):
            if (ydim != 1 or udim != 1):
                sys.exit(
                    "Error! Information criteria are implemented in SISO case for ARX and ARMAX model sets.  Use subspace methods instead for MIMO cases")
        #                identified_system='None'
        if id_method == 'ARX':
            from . import arx
            na, nb, theta, g_identif, h_identif, NUMERATOR, DENOMINATOR, Vn = arx.select_order_ARX(
                    y[0], u[0], tsample, na_ord, nb_ord, delays, IC)
            model = arx.ARX_model(na, nb, theta, tsample, NUMERATOR, DENOMINATOR,
                                              g_identif, h_identif, Vn)
        elif id_method == 'ARMAX':
            from . import armax
            model = armax.Armax(na_ord, nb_ord, nc_ord, delays, tsample, IC, ARMAX_max_iterations)
            model.find_best_estimate(y[0], u[0])

        elif id_method == 'N4SID' or id_method == 'MOESP' or id_method == 'CVA':
            from . import OLSims_methods
            A, B, C, D, Vn, Q, R, S, K = OLSims_methods.select_order_SIM(y, u, SS_f, id_method, IC,
                                                                         SS_orders, SS_D_required,
                                                                         SS_A_stability)
            model = OLSims_methods.SS_model(A, B, C, D, K, Q, R, S, tsample, Vn)
        elif id_method == 'PARSIM-K':
            from . import Parsim_methods
            A_K, C, B_K, D, K, A, B, x0, Vn = Parsim_methods.select_order_PARSIM_K(y, u, SS_f, SS_p,
                                                                                   IC, SS_orders,
                                                                                   SS_D_required,
                                                                                   SS_PK_B_reval)
            model = Parsim_methods.SS_PARSIM_model(A, B, C, D, K, A_K, B_K, x0, tsample,
                                                               Vn)
        elif id_method == 'PARSIM-S':
            from . import Parsim_methods
            A_K, C, B_K, D, K, A, B, x0, Vn = Parsim_methods.select_order_PARSIM_S(y, u, SS_f, SS_p,
                                                                                   IC, SS_orders,
                                                                                   SS_D_required)
            model = Parsim_methods.SS_PARSIM_model(A, B, C, D, K, A_K, B_K, x0, tsample,
                                                               Vn)
        elif id_method == 'PARSIM-P':
            from . import Parsim_methods
            A_K, C, B_K, D, K, A, B, x0, Vn = Parsim_methods.select_order_PARSIM_P(y, u, SS_f, SS_p,
                                                                                   IC, SS_orders,
                                                                                   SS_D_required)
            model = Parsim_methods.SS_PARSIM_model(A, B, C, D, K, A_K, B_K, x0, tsample,
                                                               Vn)
        else:
            sys.exit("Error! No identification method selected")
    #            identified_system='None'

    return model
