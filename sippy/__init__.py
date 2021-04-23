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


def system_identification(y, u, id_method, centering = 'None', IC = 'None', \
                                  tsample = 1., FIR_orders = [1, 0], ARX_orders = [1, 1, 0], \
                                  ARMA_orders = [1, 0], ARMAX_orders = [1, 1, 1, 0], \
                                  ARARX_orders = [1, 1, 1, 0], ARARMAX_orders = [1, 1, 1, 1, 0],\
                                  OE_orders = [1, 1, 0], BJ_orders = [1, 1, 1, 1, 0], GEN_orders = [1, 1, 1, 1, 1, 0], \
                                  na_ord = [0, 5], nb_ord = [1, 5], nc_ord = [0, 5], nd_ord = [0, 5], nf_ord = [0, 5], delays = [0, 5], \
                                  FIR_mod = 'LLS', ARX_mod = 'LLS', ARMAX_mod = 'ILLS', OE_mod = 'OPT', 
                          max_iterations = 200, stab_marg = 1.0, stab_cons = False,\
                          SS_f = 20, SS_p = 20, SS_threshold = 0.1, \
                          SS_max_order = np.NaN, SS_fixed_order = np.NaN, \
                          SS_orders = [1, 10], SS_D_required = False, SS_A_stability = False, \
                          SS_PK_B_reval = False):
    
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
        
        ## INPUT-OUTPUT MODELS
        
        # FIR or ARX   
        if id_method == 'FIR' or id_method == 'ARX':
            
            if id_method == 'FIR':
                # not 2 inputs
                if len(FIR_orders) != 2:
                    sys.exit("Error! FIR identification takes two arguments in FIR_orders")
                    model = 'None'
                    
                # assigned orders   
                if (type(FIR_orders[0]) == list and type(FIR_orders[1]) == list):
                    # na is set to 0  
                    # na = [0]*ydim                  
                    nb = FIR_orders[0]
                    theta = FIR_orders[1]
    
                # not assigned orders (read default)    
                elif (type(FIR_orders[0]) == int and type(FIR_orders[1]) == int):
                    # na is set to 0 
                    # na = [0]*ydim
                    nb = (FIR_orders[0] * np.ones((ydim, udim), dtype=np.int)).tolist()
                    theta = (FIR_orders[1] * np.ones((ydim, udim), dtype=np.int)).tolist()
                
                # something goes wrong
                else:
                    sys.exit(
                        "Error! FIR_orders must be a list containing two lists or two integers")
                    model = 'None'
            
            elif id_method == 'ARX':
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
                
                # something goes wrong
                else:
                    sys.exit(
                        "Error! ARX_orders must be a list containing three lists or three integers")
                    model = 'None'
                  
            # id ARX (also for FIR, which is a subcase)
            
            # Standard Linear Least Square
            if ARX_mod == 'LLS' or FIR_mod == 'LLS':
            
                from . import arxMIMO
                #import arxMIMO
                # id ARX MIMO (also SISO case)
                DENOMINATOR, NUMERATOR, G, H, Vn_tot, Yid = arxMIMO.ARX_MIMO_id(y, u, na, nb, theta, tsample)
                # recentering
                Yid = data_recentering(Yid,y_rif,ylength) 
                # form model
                model = arxMIMO.ARX_MIMO_model(na, nb, theta, tsample, NUMERATOR, DENOMINATOR, G, H, Vn_tot, Yid)
             
            # Recursive Least Square
            elif ARX_mod == 'RLLS' or FIR_mod == 'RLLS':
                
                from . import io_rlsMIMO
                #import io_rlsMIMO
                # id ARMAX RLS MIMO (also SISO case)
                DENOMINATOR, NUMERATOR, DENOMINATOR_H, NUMERATOR_H, G, H, Vn_tot, Yid = io_rlsMIMO.GEN_MIMO_id(
                        id_method, y, u, na, nb, nc, nd, nf, theta, tsample, max_iterations)
                # recentering
                Yid = data_recentering(Yid,y_rif,ylength)
                model = io_rlsMIMO.GEN_MIMO_model(na, nb, nc, nd, nf, theta, tsample, NUMERATOR, DENOMINATOR, NUMERATOR_H, DENOMINATOR_H, G, H, Vn_tot, Yid)
                
        # ARMAX        
        elif id_method == 'ARMAX':
            
            # not 4 inputs
            if len(ARMAX_orders) != 4:
                sys.exit("Error! ARMAX identification takes four arguments in ARMAX_orders")

            # assigned orders    
            if (type(ARMAX_orders[0]) == list and type(ARMAX_orders[1]) == list and type(ARMAX_orders[2]) == list and type(ARMAX_orders[3]) == list):
                na = ARMAX_orders[0]
                nb = ARMAX_orders[1]
                nc = ARMAX_orders[2]
                # nd = [0]*ydim
                # nf = [0]*ydim
                theta = ARMAX_orders[3]
            
            # not assigned orders (read default)   
            elif (type(ARMAX_orders[0]) == int and type(ARMAX_orders[1]) == int and type(ARMAX_orders[2]) == int and type(ARMAX_orders[3]) == int):
                na = (ARMAX_orders[0] * np.ones((ydim,), dtype=np.int)).tolist()
                nb = (ARMAX_orders[1] * np.ones((ydim, udim), dtype=np.int)).tolist()
                nc = (ARMAX_orders[2] * np.ones((ydim,), dtype=np.int)).tolist()
                # nd = [0]*ydim
                # nf = [0]*ydim
                theta = (ARMAX_orders[3] * np.ones((ydim, udim), dtype=np.int)).tolist()
            
            # something goes wrong
            else:
                sys.exit(
                    "Error! ARMAX_orders must be a list containing four lists or four integers")
                model = 'None'
            
            # check identification method
                
            # Iterative Linear Least Squares
            if ARMAX_mod == 'ILLS':
                
                from . import armaxMIMO
                #import armaxMIMO
                # id ARMAX MIMO (also SISO case)
                DENOMINATOR, NUMERATOR, DENOMINATOR_H, NUMERATOR_H, G, H, Vn_tot, Yid = armaxMIMO.ARMAX_MIMO_id(
                        y, u, na, nb, nc, theta, tsample, max_iterations)
                # recentering
                Yid = data_recentering(Yid,y_rif,ylength)
                model = armaxMIMO.ARMAX_MIMO_model(na, nb, nc, theta, tsample, NUMERATOR, DENOMINATOR, NUMERATOR_H, DENOMINATOR_H, G, H, Vn_tot, Yid)
                
            # Recursive Least Square
            elif ARMAX_mod == 'RLLS':
                
                from . import io_rlsMIMO
                #import io_rlsMIMO
                # id ARMAX RLS MIMO (also SISO case)
                DENOMINATOR, NUMERATOR, DENOMINATOR_H, NUMERATOR_H, G, H, Vn_tot, Yid = io_rlsMIMO.GEN_MIMO_id(
                        id_method, y, u, na, nb, nc, nd, nf, theta, tsample, max_iterations)
                # recentering
                Yid = data_recentering(Yid,y_rif,ylength)
                model = io_rlsMIMO.GEN_MIMO_model(na, nb, nc, nd, nf, theta, tsample, NUMERATOR, DENOMINATOR, NUMERATOR_H, DENOMINATOR_H, G, H, Vn_tot, Yid)
                         
            # OPTMIZATION-BASED
            elif ARMAX_mod == 'OPT':
                from . import io_optMIMO
                #import io_optMIMO
                # id GEN MIMO (also SISO case)
                DENOMINATOR, NUMERATOR, DENOMINATOR_H, NUMERATOR_H, G, H, Vn_tot, Yid = io_optMIMO.GEN_MIMO_id(
                    id_method, y, u, na, nb, nc, nd, nf, theta, tsample, max_iterations, stab_marg, stab_cons)
                # recentering
                Yid = data_recentering(Yid,y_rif,ylength)
                # form model
                model = io_optMIMO.GEN_MIMO_model(na, nb, nc, nd, nf, theta, tsample,
                                               NUMERATOR, DENOMINATOR, NUMERATOR_H, DENOMINATOR_H, G, H, Vn_tot, Yid)
            
            else:
                # error in writing the identification mode
                print('Warning: the selected method for solving the ARMAX model is not correct.')
                model = 'None'
                    
              
        # (OE) Output-Error
        elif id_method == 'OE':
            
            # not 3 inputs
            if len(OE_orders) != 3:
                sys.exit("Error! OE identification takes three arguments in OE_orders")
                model = 'None'
                
            # assigned orders   
            if (type(OE_orders[0]) == list and type(OE_orders[1]) == list and type(OE_orders[2]) == list):
                # na = [0]*ydim
                nb = OE_orders[0]
                # nc = [0]*ydim
                # nd = [0]*ydim
                nf = OE_orders[1]
                theta = OE_orders[2]

            # not assigned orders (read default)   
            elif (type(OE_orders[0]) == int and type(OE_orders[1]) == int and type(OE_orders[2]) == int):
                # na = [0]*ydim
                nb = (OE_orders[0] * np.ones((ydim, udim), dtype=np.int)).tolist()
                # nc = [0]*ydim
                # nd = [0]*ydim
                nf = (OE_orders[1] * np.ones((ydim,), dtype=np.int)).tolist()
                theta = (OE_orders[2] * np.ones((ydim, udim), dtype=np.int)).tolist()
            
            # something goes wrong
            else:
                sys.exit(
                    "Error! OE_orders must be a list containing three lists or three integers")
                model = 'None'
                
            # check identification method
                
            # Iterative Linear Least Squares
            if OE_mod == 'RLLS':
                
                from . import io_rlsMIMO
                #import io_rlsMIMO
                # id ARMAX RLS MIMO (also SISO case)
                DENOMINATOR, NUMERATOR, DENOMINATOR_H, NUMERATOR_H, G, H, Vn_tot, Yid = io_rlsMIMO.GEN_MIMO_id(
                        id_method, y, u, na, nb, nc, nd, nf, theta, tsample, max_iterations)
                # recentering
                Yid = data_recentering(Yid,y_rif,ylength)
                model = io_rlsMIMO.GEN_MIMO_model(na, nb, nc, nd, nf, theta, tsample, NUMERATOR, DENOMINATOR, NUMERATOR_H, DENOMINATOR_H, G, H, Vn_tot, Yid)
                    
                         
            # OPTMIZATION-BASED
            elif OE_mod == 'OPT':
                from . import io_optMIMO
                #import io_optMIMO
                # id GEN MIMO (also SISO case)
                DENOMINATOR, NUMERATOR, DENOMINATOR_H, NUMERATOR_H, G, H, Vn_tot, Yid = io_optMIMO.GEN_MIMO_id(
                    id_method, y, u, na, nb, nc, nd, nf, theta, tsample, max_iterations, stab_marg, stab_cons)
                # recentering
                Yid = data_recentering(Yid,y_rif,ylength)
                # form model
                model = io_optMIMO.GEN_MIMO_model(na, nb, nc, nd, nf, theta, tsample,
                                               NUMERATOR, DENOMINATOR, NUMERATOR_H, DENOMINATOR_H, G, H, Vn_tot, Yid)
            
            else:
                # error in writing the identification mode
                print('Warning: the selected method for solving the OE model is not correct.')
                model = 'None'
                
                
        
        # other INPUT-OUTPUT STRUCTURES OPTIMIZATION-BASED:
        # ARMA, ARARX, ARARMAX            
        # BJ (BOX-JENKINS), GEN-MOD (GENERALIZED MODEL) 
                         
        elif id_method == 'ARMA' or id_method == 'ARARX' or id_method == 'ARARMAX' \
            or id_method == 'GEN' or id_method == 'BJ' or id_method == 'OE':
            
            # ARMA       
            if id_method == 'ARMA':
            
                # not 3 inputs
                if len(ARMA_orders) != 3:
                    sys.exit("Error! ARMA identification takes three arguments in ARMA_orders")
    
                # assigned orders    
                if (type(ARMA_orders[0]) == list and type(ARMA_orders[1]) == list and type(ARMA_orders[2]) == list):
                    na = ARMA_orders[0]
                    nb = np.zeros((ydim, udim), dtype=np.int).tolist()
                    nc = ARMA_orders[1]
                    # nd = [0]*ydim
                    # nf = [0]*ydim
                    theta = ARMA_orders[2]
                
                # not assigned orders (read default)   
                elif (type(ARMA_orders[0]) == int and type(ARMA_orders[1]) == int and type(ARMA_orders[2]) == int):
                    na = (ARMA_orders[0] * np.ones((ydim,), dtype=np.int)).tolist()
                    #nb = (ARMA_orders[1] * np.ones((ydim, udim), dtype=np.int)).tolist()
                    nb = np.zeros((ydim, udim), dtype=np.int).tolist()
                    nc = ARMA_orders[1]
                    # nd = [0]*ydim
                    # nf = [0]*ydim
                    theta = (ARMA_orders[2] * np.ones((ydim, udim), dtype=np.int)).tolist()
                
                # something goes wrong
                else:
                    sys.exit(
                        "Error! ARMA_orders must be a list containing three lists or three integers")
                    model = 'None'
                
            # ARARX       
            elif id_method == 'ARARX':
            
                # not 4 inputs
                if len(ARARX_orders) != 4:
                    sys.exit("Error! ARARX identification takes four arguments in ARARX_orders")
    
                # assigned orders    
                if (type(ARARX_orders[0]) == list and type(ARARX_orders[1]) == list and type(ARARX_orders[2]) == list and type(ARARX_orders[3]) == list):
                    na = ARARX_orders[0]
                    nb = ARARX_orders[1]
                    # nc = [0]*ydim
                    nd = ARARX_orders[2]
                    # nf = [0]*ydim
                    theta = ARARX_orders[3]
                
                # not assigned orders (read default)   
                elif (type(ARARX_orders[0]) == int and type(ARARX_orders[1]) == int and type(ARARX_orders[2]) == int and type(ARARX_orders[3]) == int):
                    na = (ARARX_orders[0] * np.ones((ydim,), dtype=np.int)).tolist()
                    nb = (ARARX_orders[1] * np.ones((ydim, udim), dtype=np.int)).tolist()
                    nc = [0]*ydim
                    nd = (ARARX_orders[2] * np.ones((ydim,), dtype=np.int)).tolist()
                    nf = [0]*ydim
                    theta = (ARARX_orders[3] * np.ones((ydim, udim), dtype=np.int)).tolist()
                
                # something goes wrong
                else:
                    sys.exit(
                        "Error! ARARX_orders must be a list containing four lists or four integers")
                    model = 'None'
        
            # ARARMAX        
            elif id_method == 'ARARMAX':
            
                # not 5 inputs
                if len(ARARMAX_orders) != 5:
                    sys.exit("Error! ARARMAX identification takes five arguments in ARARMAX_orders")
    
                # assigned orders    
                if (type(ARARMAX_orders[0]) == list and type(ARARMAX_orders[1]) == list and type(ARARMAX_orders[2]) == list and type(ARARMAX_orders[3]) == list and type(ARARMAX_orders[4]) == list):
                    na = ARARMAX_orders[0]
                    nb = ARARMAX_orders[1]
                    nc = ARARMAX_orders[2]
                    nd = ARARMAX_orders[3]
                    # nf = [0]*ydim
                    theta = ARARMAX_orders[4]
                
                # not assigned orders (read default)   
                elif (type(ARARMAX_orders[0]) == int and type(ARARMAX_orders[1]) == int and type(ARARMAX_orders[2]) == int and type(ARARMAX_orders[3]) == int and type(ARARMAX_orders[4]) == int):
                    na = (ARARMAX_orders[0] * np.ones((ydim,), dtype=np.int)).tolist()
                    nb = (ARARMAX_orders[1] * np.ones((ydim, udim), dtype=np.int)).tolist()
                    nc = (ARARMAX_orders[2] * np.ones((ydim,), dtype=np.int)).tolist()
                    nd = (ARARMAX_orders[3] * np.ones((ydim,), dtype=np.int)).tolist()
                    # nf = [0]*ydim
                    theta = (ARARMAX_orders[4] * np.ones((ydim, udim), dtype=np.int)).tolist()
                
                # something goes wrong
                else:
                    sys.exit(
                        "Error! ARARMAX_orders must be a list containing five lists or five integers")
                    model = 'None'
            

            # BJ 
            elif id_method == 'BJ':
            
                # not 5 inputs
                if len(BJ_orders) != 5:
                    sys.exit("Error! BJ identification takes five arguments in BJ_orders")
    
                # assigned orders    
                if (type(BJ_orders[0]) == list and type(BJ_orders[1]) == list and type(BJ_orders[2]) == list \
                    and type(BJ_orders[3]) == list and type(BJ_orders[4]) == list):
                    # na = [0]*ydim 
                    nb = BJ_orders[0]
                    nc = BJ_orders[1]
                    nd = BJ_orders[2]
                    nf = BJ_orders[3]
                    theta = BJ_orders[4]
                
                # not assigned orders (read default)   
                elif (type(BJ_orders[0]) == int and type(BJ_orders[1]) == int and type(BJ_orders[2]) == int \
                      and type(BJ_orders[3]) == int and type(BJ_orders[4]) == int):
                    # na = [0]*ydim 
                    nb = (BJ_orders[0] * np.ones((ydim, udim), dtype=np.int)).tolist()
                    nc = (BJ_orders[1] * np.ones((ydim,), dtype=np.int)).tolist()
                    nd = (BJ_orders[2] * np.ones((ydim,), dtype=np.int)).tolist()
                    nf = (BJ_orders[3] * np.ones((ydim,), dtype=np.int)).tolist()
                    theta = (BJ_orders[4] * np.ones((ydim, udim), dtype=np.int)).tolist()
                
                # something goes wrong
                else:
                    sys.exit(
                        "Error! BJ_orders must be a list containing five lists or five integers")
                    model = 'None'
                        
            # GEN
            elif id_method == 'GEN':
            
                # not 6 inputs
                if len(GEN_orders) != 6:
                    sys.exit("Error! GEN-MODEL identification takes six arguments in GEN_orders")
    
                # assigned orders    
                if (type(GEN_orders[0]) == list and type(GEN_orders[1]) == list and type(GEN_orders[2]) == list \
                    and type(GEN_orders[3]) == list and type(GEN_orders[4]) == list and type(GEN_orders[5]) == list):
                    na = GEN_orders[0]
                    nb = GEN_orders[1]
                    nc = GEN_orders[2]
                    nd = GEN_orders[3]
                    nf = GEN_orders[4]
                    theta = GEN_orders[5]
                
                # not assigned orders (read default)   
                elif (type(GEN_orders[0]) == int and type(GEN_orders[1]) == int and type(GEN_orders[2]) == int \
                      and type(GEN_orders[3]) == int and type(GEN_orders[4]) == int and type(GEN_orders[5]) == int): 
                    na = (GEN_orders[0] * np.ones((ydim,), dtype=np.int)).tolist()
                    nb = (GEN_orders[1] * np.ones((ydim, udim), dtype=np.int)).tolist()
                    nc = (GEN_orders[2] * np.ones((ydim,), dtype=np.int)).tolist()
                    nd = (GEN_orders[3] * np.ones((ydim,), dtype=np.int)).tolist()
                    nf = (GEN_orders[4] * np.ones((ydim,), dtype=np.int)).tolist()
                    theta = (GEN_orders[5] * np.ones((ydim, udim), dtype=np.int)).tolist()
                
                # something goes wrong
                else:
                    sys.exit(
                        "Error! GEN_orders must be a list containing six lists or six integers")
                    model = 'None'
            
            # id MODEL: ARMA, ARARX, ARARMAX, BJ, GEN
            from . import io_optMIMO
            #import io_optMIMO
            # id MIMO (also SISO case)
            DENOMINATOR, NUMERATOR, DENOMINATOR_H, NUMERATOR_H, G, H, Vn_tot, Yid = io_optMIMO.GEN_MIMO_id(
                    id_method, y, u, na, nb, nc, nd, nf, theta, tsample, max_iterations, stab_marg, stab_cons)
            # recentering
            Yid = data_recentering(Yid,y_rif,ylength)
            # form model
            model = io_optMIMO.GEN_MIMO_model(na, nb, nc, nd, nf, theta, tsample,
                                               NUMERATOR, DENOMINATOR, NUMERATOR_H, DENOMINATOR_H, G, H, Vn_tot, Yid)
            
            
        
        ## SS MODELS
            
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

    
    ##### 
    ### MODE 2) order range
    # if an IC is selected
    else:
        
        # method choice
        
        ## INPUT-OUTPUT MODELS (MIMO case --> not implemented)
        if (id_method == 'FIR' or id_method == 'ARX' or id_method == 'ARMA' or id_method == 'ARMAX' \
            or id_method == 'ARARX' or id_method == 'ARARMAX' or id_method == 'OE' or id_method == 'BJ' or id_method == 'GEN' \
                or id_method == 'EARMAX' or id_method == 'EOE'):
            if (ydim != 1 or udim != 1):
                sys.exit(
                    "Error! Information criteria are implemented ONLY in SISO case for INPUT-OUTPUT model sets.  Use subspace methods instead for MIMO cases")
                model = 'None'
        
        ## FIR or ARX      
        if id_method == 'FIR' or id_method == 'ARX':
            
            if ARX_mod == 'LLS' or FIR_mod == 'LLS':
                
                from . import arx
                if id_method == 'FIR':
                    # no iteration on A order: rewrite na
                    na_ord = [0,0]
                # import arx
                na, nb, theta, g_identif, h_identif, NUMERATOR, DENOMINATOR, Vn, Yid = arx.select_order_ARX(y[0], u[0], tsample, na_ord, nb_ord, delays, IC)
                # recentering
                Yid = data_recentering(Yid,y_rif,ylength)
                model = arx.ARX_model(na, nb, theta, tsample, NUMERATOR, DENOMINATOR, g_identif, h_identif, Vn, Yid)
            
            elif ARX_mod == 'RLLS' or FIR_mod == 'RLLS':
                # no iteration on C, D and F orders: rewrite nc, nd and nf
                if id_method == 'FIR':
                    # no iteration on A order: rewrite na
                    na_ord = [0,0]
                nc_ord = [0,0]    
                nd_ord = [0,0]
                nf_ord = [0,0]
                from . import io_rls
                #import io_rlsMIMO
                # id IO RLS MIMO (also SISO case)
                na, nb, nc, nd, nf, theta, g_identif, h_identif, NUMERATOR, DENOMINATOR, Vn, Yid = io_rls.select_order_GEN(id_method, y[0], u[0],
                                                        tsample, na_ord, nb_ord, nc_ord, nd_ord, nf_ord, delays, IC, max_iterations)
                # recentering
                Yid = data_recentering(Yid,y_rif,ylength)
                model = io_rls.GEN_model(na, nb, nc, nd, nf, theta, tsample, NUMERATOR, DENOMINATOR, g_identif, h_identif, Vn, Yid)
                
                    
        
        ## ARMAX
        elif id_method == 'ARMAX':
            
            # Iterative Linear Least Squares
            if ARMAX_mod == 'ILLS':               
            
                from . import armax
                # import armax
                # file updated by people external from CPCLAB
                model = armax.Armax(na_ord, nb_ord, nc_ord, delays, tsample, IC, max_iterations)
                #
                model.find_best_estimate(y[0], u[0])
                # recentering
                Yid = data_recentering(model.Yid,y_rif,ylength)
                
            # Recursive Least Square
            elif ARMAX_mod == 'RLLS':
                # no iteration on D and F orders: rewrite nd and nf
                nd_ord = [0,0]
                nf_ord = [0,0]
                from . import io_rls
                #import io_rlsMIMO
                # id IO RLS MIMO (also SISO case)
                na, nb, nc, nd, nf, theta, g_identif, h_identif, NUMERATOR, DENOMINATOR, Vn, Yid = io_rls.select_order_GEN(id_method, y[0], u[0],
                                                        tsample, na_ord, nb_ord, nc_ord, nd_ord, nf_ord, delays, IC, max_iterations)
                # recentering
                Yid = data_recentering(Yid,y_rif,ylength)
                model = io_rls.GEN_model(na, nb, nc, nd, nf, theta, tsample, NUMERATOR, DENOMINATOR, g_identif, h_identif, Vn, Yid)
                   
            
            # OPTMIZATION-BASED
            elif ARMAX_mod == 'OPT':
                # no iteration on D and F orders: rewrite nd and nf
                nd_ord = [0,0]
                nf_ord = [0, 0]
                from . import io_opt
                # import io_opt
                na, nb, nc, nd, nf, theta, g_identif, h_identif, NUMERATOR, DENOMINATOR, Vn, Yid = io_opt.select_order_GEN(id_method, y[0], u[0], 
                                                        tsample, na_ord, nb_ord, nc_ord, nd_ord, nf_ord, delays, IC, max_iterations, stab_marg, stab_cons)
                # recentering
                Yid = data_recentering(Yid,y_rif,ylength)
                model = io_opt.GEN_model(na, nb, nc, nd, nf, theta, tsample, NUMERATOR, DENOMINATOR, g_identif, h_identif, Vn, Yid)
                
            else: # error in writing the mode
                print('Warning: the selected method for solving the ARMAX model is not correct.') 
                model = 'None'
         
        # (OE) Output-Error    
        elif id_method == 'OE':
            
            if OE_mod == 'RLLS':
                # no iteration on A, C, and D orders: rewrite na, nc, nd
                na_ord = [0,0]
                nc_ord = [0,0]
                nd_ord = [0,0]
                
                from . import io_rls
                #import io_rlsMIMO
                # id IO RLS MIMO (also SISO case)
                na, nb, nc, nd, nf, theta, g_identif, h_identif, NUMERATOR, DENOMINATOR, Vn, Yid = io_rls.select_order_GEN(id_method, y[0], u[0],
                                                        tsample, na_ord, nb_ord, nc_ord, nd_ord, nf_ord, delays, IC, max_iterations)
                # recentering
                Yid = data_recentering(Yid,y_rif,ylength)
                model = io_rls.GEN_model(na, nb, nc, nd, nf, theta, tsample, NUMERATOR, DENOMINATOR, g_identif, h_identif, Vn, Yid)
            
            elif OE_mod == 'OPT':
                # no iteration on A, C, and D orders: rewrite na, nc, nd
                na_ord = [0,0]
                nc_ord = [0,0]
                nd_ord = [0,0]
                
                from . import io_opt
                # import io_opt
                na, nb, nc, nd, nf, theta, g_identif, h_identif, NUMERATOR, DENOMINATOR, Vn, Yid = io_opt.select_order_GEN(id_method, y[0], u[0], 
                                                        tsample, na_ord, nb_ord, nc_ord, nd_ord, nf_ord, delays, IC, max_iterations, stab_marg, stab_cons)
                # recentering
                Yid = data_recentering(Yid,y_rif,ylength)
                model = io_opt.GEN_model(na, nb, nc, nd, nf, theta, tsample, NUMERATOR, DENOMINATOR, g_identif, h_identif, Vn, Yid)
                
            else: # error in writing the mode
                print('Warning: the selected method for solving the ARMAX model is not correct.') 
                model = 'None'
                
        # (EOE or EARMAX) Extended Output-Error and Extended ARMAX   
        elif id_method == 'EOE' or id_method == 'EARMAX':
            
            if OE_mod == 'EOE':
                nc_ord = [0, 0]
                
            from . import io_ex_rls
            #import io_ex_rlsMIMO
            # id IO RLS MIMO (also SISO case)
            na, nb, nc, theta, g_identif, h_identif, NUMERATOR, DENOMINATOR, Vn, Yid = io_ex_rls.select_order_GEN(id_method, y[0], u[0],
                                                    tsample, nf_ord, nb_ord, nc_ord, delays, IC, max_iterations)
            # recentering
            Yid = data_recentering(Yid,y_rif,ylength)
            model = io_ex_rls.GEN_model(na, nb, nc, theta, tsample, NUMERATOR, DENOMINATOR, g_identif, h_identif, Vn, Yid)
            
  
        # INPUT-OUTPUT STRUCTURES OPTIMIZATION-BASED:
        # ARMA, ARARX, ARARMAX and OE, BJ (BOX-JENKINS), GEN-MOD (GENERALIZED MODEL)           
            
        elif id_method == 'ARMA' or id_method == 'ARARX' or id_method == 'ARARMAX' \
            or id_method == 'GEN' or id_method == 'BJ':
            
            # ARMA           
            if id_method == 'ARMA':
                nb_ord = [1,1]
                nd_ord = [0,0]
                nf_ord = [0,0]
            
            # ARARX            
            elif id_method == 'ARARX':
                nc_ord = [0,0]
                nf_ord = [0,0]
            
            # ARARMAX
            elif id_method == 'ARARMAX':
                nf_ord = [0,0]
            
            # BJ
            elif id_method == 'BJ':
                na_ord = [0,0]
            
            ## GEN MODEL (all parameters already defined or by-default)

            from . import io_opt
            # import io_opt
            na, nb, nc, nd, nf, theta, g_identif, h_identif, NUMERATOR, DENOMINATOR, Vn, Yid = io_opt.select_order_GEN(id_method, y[0], u[0], 
                                                    tsample, na_ord, nb_ord, nc_ord, nd_ord, nf_ord, delays, IC, max_iterations, stab_marg, stab_cons)
            # recentering
            Yid = data_recentering(Yid,y_rif,ylength)
            model = io_opt.GEN_model(na, nb, nc, nd, nf, theta, tsample, NUMERATOR, DENOMINATOR, g_identif, h_identif, Vn, Yid)
            

        ## SS-MODELS
            
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
            model = 'None'

    return model

# Data recentering
def data_recentering(y,y_rif,ylength):  
    for i in range(ylength):
        y[:, i] = y[:, i] + y_rif
    return y