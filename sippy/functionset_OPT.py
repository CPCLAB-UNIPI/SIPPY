#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2021

@author: RBdC & MV
"""

from __future__ import division, print_function

import sys
from builtins import range

import numpy as np
from past.utils import old_div

import control.matlab as cnt

from casadi import *
from casadi.tools import *


###### Defining the optimization problem
def opt_id(m, p, na, nb, nc, nd, nf, n_coeff, theta, n_tr, U, Y, FLAG, max_iterations, stab_marg, stability_cons): 
    
    # orders
    Na = na; Nb = np.sum(nb[:]); Nc = nc; Nd = nd; Nf = nf
    
    # Augment the optmization variables with Y vector to build a multiple shooting problem
    N = Y.size
    
    # Augment the optmization variables with auxiliary variables
    if Nd!= 0:
        n_aus =  3*N 
    else:
        n_aus = N
    
    # Optmization variables
    n_opt = n_aus + n_coeff
 
    # Define symbolic optimization variables
    w_opt = SX.sym("w",n_opt)  
    
    
    # Build optimization variable
    # Get subset a
    a = w_opt[0:Na]
    
    # Get subset b
    b = w_opt[Na:Na+Nb]
    
    # Get subsets c and d
    c = w_opt[Na+Nb:Na+Nb+Nc]
    d = w_opt[Na+Nb+Nc:Na+Nb+Nc+Nd]
    
    # Get subset f
    f = w_opt[Na+Nb+Nd+Nc:Na+Nb+Nc+Nd+Nf]
    
    # Optimization variables
    Yidw = w_opt[-N:]
    
    # Additional optimization variables  
    if Nd!= 0:
        Ww = w_opt[-3*N:-2*N]
        Vw = w_opt[-2*N:-N]
       
    
    # Initializing bounds on optimization variables
    w_lb = - 1e0*DM.inf(n_opt)
    w_ub =  1e0*DM.inf(n_opt)
    #
    w_lb = - 1e2*DM.ones(n_opt)
    w_ub =  1e2*DM.ones(n_opt)
    
    
    ## Build Regressor
    # depending on the model structure
    
    # Building coefficient vector
    if FLAG == 'OE':        
        coeff = vertcat(b,f)
    elif FLAG == 'BJ':          
        coeff = vertcat(b,f,c,d)
    elif FLAG == 'ARMAX':
        coeff = vertcat(a,b,c)
    elif FLAG == 'ARARX': 
        coeff = vertcat(a,b,d)
    elif FLAG == 'ARARMAX': 
        coeff = vertcat(a,b,c,d)
    elif FLAG == 'ARMA':
        coeff = vertcat(a,c)
    else: # GEN
        coeff = vertcat(a,b,f,c,d)
       
    # Define Yid output model
    Yid = Y*SX.ones(1)
    Yid1 = Y*SX.ones(1)
        
    # Preallocate internal variables
    if Nd!=0:    
        W = Y*SX.ones(1)    #w = B * u or w = B/F * u
        V = Y*SX.ones(1)    #v = A*y - w
        
        if Na != 0:
            coeff_v = a
        if Nf != 0: # BJ, GEN
            coeff_w = vertcat(b,f)
        else: # ARARX, ARARMAX
            coeff_w = vertcat(b) 
            
    if Nc!=0:
        Epsi = SX.zeros(N)
        
    for k in range(N):
        # n_tr: number of not identifiable outputs
        if k >= n_tr:
        # building regressor
            if Nb != 0:
                # inputs
                vecU = []
                for nb_i in range(m):
                    vecu = U[nb_i, :][k-nb[nb_i]-theta[nb_i]:k-theta[nb_i]][::-1]
                    vecU = vertcat(vecU,vecu)
                
            # measured output Y
            if Na != 0: 
                vecY = Y[k-Na:k][::-1]
            
            # auxiliary variable V 
            if Nd != 0:
                vecV = Vw[k-Nd:k][::-1]
                
                # auxiliary variable W    
                if Nf != 0:
                    vecW = Ww[k-Nf:k][::-1] 
            
            # prediction error
            if Nc != 0:
                vecE = Epsi[k-Nc:k][::-1]
            
            # regressor
            if FLAG == 'OE':
                vecY = Yidw[k-Nf:k][::-1]
                phi = vertcat(vecU,-vecY)
            elif FLAG == 'BJ': 
                phi = vertcat(vecU, -vecW, vecE, -vecV)
            elif FLAG == 'ARMAX':
                phi = vertcat(-vecY, vecU, vecE)
            elif FLAG == 'ARMA':
                phi = vertcat(-vecY, vecE)
            elif FLAG == 'ARARX': 
                phi = vertcat(-vecY, vecU, -vecV)
            elif FLAG == 'ARARMAX': 
                phi = vertcat(-vecY, vecU, vecE, -vecV)
            else:
                phi = vertcat(-vecY, vecU, -vecW, vecE, -vecV)
            
            # update prediction
            Yid[k] = mtimes(phi.T,coeff)
            
            # pred. error
            if Nc != 0:
                Epsi[k] = Y[k] - Yidw[k]
                
            # auxiliary variable W
            if Nd != 0:
                if Nf != 0:
                    phiw = vertcat(vecU,-vecW)  # BJ, GEN
                else:
                    phiw = vertcat(vecU)       # ARARX, ARARMAX
                W[k] = mtimes(phiw.T,coeff_w)
                
                # auxiliary variable V
                if Na == 0:                     # 'BJ'  [A(z) = 1]
                    V[k] = Y[k] - Ww[k]
                else:               #[A(z) div 1]
                    phiv = vertcat(vecY)
                    V[k] = Y[k] + mtimes(phiv.T,coeff_v) - Ww[k]

    # Objective Function    
    DY = Y - Yidw   
    
    f_obj = (1.0/(N))*mtimes(DY.T,DY) 
    
    #if  FLAG != 'ARARX' or FLAG != 'OE':
    #   f_obj += 1e-4*mtimes(c.T,c)   # weighting c
     
    
    ## Getting constrains
    g = []
    
    # Equality constraints
    g.append(Yid - Yidw) 
    
    if Nd != 0:
          g.append(W - Ww)
          g.append(V - Vw)
    
    # Stability check
    ng_norm = 0
    if stability_cons is True:
        if Na != 0:
            ng_norm += 1
            # companion matrix A(z) polynomial
            compA = SX.zeros(Na,Na)
            diagA = SX.eye(Na-1)
            compA[:-1,1:] = diagA
            compA[-1,:] = -a[::-1] # opposite reverse coeficient a
            
            # infinite-norm
            norm_CompA = norm_inf(compA)
            
            # append on eq. constraints
            g.append(norm_CompA)
            
        if Nf != 0:
            ng_norm += 1
            # companion matrix F(z) polynomial
            compF = SX.zeros(Nf,Nf)
            diagF = SX.eye(Nf-1)
            compF[:-1,1:] = diagF
            compF[-1,:] = -f[::-1] # opposite reverse coeficient f
            
            # infinite-norm
            norm_CompF = norm_inf(compF)
            
            # append on eq. constraints
            g.append(norm_CompF)
        
        if Nd != 0:
            ng_norm += 1
            # companion matrix D(z) polynomial
            compD = SX.zeros(Nd,Nd)
            diagD = SX.eye(Nd-1)
            compD[:-1,1:] = diagD
            compD[-1,:] = -d[::-1] # opposite reverse coeficient D
            
            # infinite-norm
            norm_CompD = norm_inf(compD)
            
            # append on eq. constraints
            g.append(norm_CompD)
    
    # constraint vector
    g = vertcat(*g)
    
    # Constraint bounds
    ng = g.size1()
    g_lb = -1e-7*DM.ones(ng,1)
    g_ub = 1e-7*DM.ones(ng,1)
    
    # Force system stability
    # note: norm_inf(X) >= Spectral radius (A)
    if ng_norm != 0:
        g_ub[-ng_norm:] = stab_marg*DM.ones(ng_norm,1)    
        # for i in range(ng_norm):
        #     f_obj += 1e1*fmax(0,g_ub[-i-1:]-g[-i-1:])

    # NL optimization variables    
    nlp = {'x':w_opt, 'f':f_obj, 'g':g}

    # Solver options
    #sol_opts = {'ipopt.max_iter':max_iterations}#, 'ipopt.tol':1e-10}#,'ipopt.print_level':0,'ipopt.sb':"yes",'print_time':0}
    sol_opts = {'ipopt.max_iter':max_iterations, 'ipopt.print_level':0,'ipopt.sb':"yes",'print_time':0}
    
    # Defining the solver
    solver = nlpsol('solver', 'ipopt', nlp, sol_opts)
    
    return [solver, w_lb, w_ub, g_lb, g_ub]



