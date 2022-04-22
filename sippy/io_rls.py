# -*- coding: utf-8 -*-
"""
Created on 2021

@author: RBdC & MV
"""
from __future__ import absolute_import, division, print_function

import sys
from builtins import object
import control.matlab as cnt
from .functionset import *
from .functionset_OPT import *
# from functionset import *


def GEN_RLS_id(id_method, y, u, na, nb, nc, nd, nf, theta, max_iterations):

    ylength = y.size
        
    # input/output number
    m = 1; p = 1
    
    # max number of non predictable data
    nbth = nb + theta
    val = max(na, nbth, nc, nd, nf)
    # whole data
    N = ylength
    
    # Total Order: both LTI and time varying part
    nt = na + nb + nc + nd + nf + 1
    nh = max([na,nc])
    
    ## Iterative Identification Algorithm

    ## Parameters Initialization
    # Confidence Parameter
    Beta = 1e4
    # Covariance matrix of parameter teta
    p_t = Beta*np.eye(nt-1,nt-1)
    P_t = np.zeros((nt-1,nt-1,N))
    for i in range(N):
            P_t[:,:,i] = p_t
    # Gain
    K_t = np.zeros((nt-1,N))
        
    # First estimatate
    teta = np.zeros((nt-1,N))
    eta = np.zeros(N)
    # Forgetting factors
    L_t = 1
    l_t = L_t*np.ones(N)
    #
    Yp = np.zeros(N)
    E = np.zeros(N)
    fi = np.zeros((1,nt-1,N))
   
    ## Propagation
    for k in range(N):
        if k > val:
            ## Step 1: Regressor vector
            vecY = y[k-na:k][::-1]                      # Y vector
            vecYp = Yp[k-nf:k][::-1]                    # Yp vector
            #             
            vecU = u[k-nb-theta:k-theta][::-1]          # U vector
            # 
            #vecE = E[k-nh:k][::-1]                     # E vector
            vecE = E[k-nc:k][::-1] 
                       
           # choose input-output model
            if id_method == 'ARMAX':
                fi[:,:,k] = np.hstack((-vecY, vecU, vecE))
            elif id_method == 'ARX':
                fi[:,:,k] = np.hstack((-vecY, vecU))
            elif id_method == 'OE':
                fi[:,:,k] = np.hstack((-vecYp, vecU))
            elif id_method == 'FIR':
                fi[:,:,k] = np.hstack((vecU))
            phi = fi[:,:,k].T
            
            ## Step 2: Gain Update
            # Gain of parameter teta
            K_t[:,k:k+1] = np.dot(np.dot(P_t[:,:,k-1],phi),np.linalg.inv(l_t[k-1] + np.dot(np.dot(phi.T,P_t[:,:,k-1]),phi)))

            ## Step 3: Parameter Update
            teta[:,k] = teta[:,k-1] + np.dot(K_t[:,k:k+1],(y[k] - np.dot(phi.T,teta[:,k-1])))
            
            ## Step 4: A posteriori prediction-error
            Yp[k] = np.dot(phi.T,teta[:,k]) + eta[k]
            E[k] = y[k] - Yp[k]

            ## Step 5. Parameter estimate covariance update
            P_t[:,:,k] = (1/l_t[k-1])*(np.dot(np.eye(nt-1) - np.dot(K_t[:,k:k+1],phi.T),P_t[:,:,k-1]))

            ## Step 6: Forgetting factor update
            l_t[k] = 1.0
        
    # Error Norm
    Vn = old_div((np.linalg.norm(y - Yp, 2) ** 2), (2 * (N-val)))
    
    # Model Output
    y_id = Yp
    
    # Parameters
    THETA = teta[:,-1]
    
    # building TF coefficient vectors
    valH = max(nc, na + nd)
    valG = max(nb + theta, na + nf)   
     
    # G    
    # numG (B)
    if id_method == 'ARMA':
        NUM = 1.0     
    else: 
        NUM = np.zeros(valG)
        ng = nf if id_method == 'OE' else na
        NUM[theta:nb + theta] = THETA[ng:nb+ng]
    # denG (A*F)
    A = cnt.tf(np.hstack((1, np.zeros((na)))), np.hstack((1, THETA[:na])),1)
    
    if id_method == 'OE':
        F = cnt.tf(np.hstack((1, np.zeros((nf)))), np.hstack((1, THETA[:nf])),1)
    else:
        F = cnt.tf(np.hstack((1, np.zeros((nf)))), np.hstack((1, THETA[na+nb+nc+nd:na+nb+nc+nd+nf])),1)
    _, deng = cnt.tfdata(A*F) 
    denG = np.array(deng[0])
    DEN = np.zeros(valG + 1)
    DEN[0:na+nf+1] = denG
    
    # H
    # numH (C)
    if id_method == 'OE':
        NUMH = 1
    else:
        NUMH = np.zeros(valH + 1)
        NUMH[0] = 1.
        NUMH[1:nc + 1] = THETA[na+nb:na+nb+nc]
    # denH (A*D)
    D = cnt.tf(np.hstack((1, np.zeros((nd)))), np.hstack((1, THETA[na+nb+nc:na+nb+nc+nd])),1)
    _, denh = cnt.tfdata(A*D)
    denH = np.array(denh[0])
    DENH = np.zeros(valH + 1)
    DENH[0:na+nd+1] = denH
    
    
    return NUM, DEN, NUMH, DENH, Vn, y_id


def select_order_GEN(id_method, y, u, tsample=1., na_ord=[0, 5], nb_ord=[1, 5], nc_ord=[0, 5], nd_ord=[0, 5], nf_ord=[0, 5], delays=[0, 5], method='AIC', max_iterations = 200):
    # order ranges
    na_Min = min(na_ord)
    na_MAX = max(na_ord) + 1
    nb_Min = min(nb_ord)
    nb_MAX = max(nb_ord) + 1
    nc_Min = min(nc_ord)
    nc_MAX = max(nc_ord) + 1
    nd_Min = min(nd_ord)
    nd_MAX = max(nd_ord) + 1
    nf_Min = min(nf_ord)
    nf_MAX = max(nf_ord) + 1
    theta_Min = min(delays)
    theta_Max = max(delays) + 1
    # check orders
    sum_ords = np.sum(na_Min + na_MAX + nb_Min + nb_MAX + nc_Min + nc_MAX + nd_Min + nd_MAX + nf_Min + nf_MAX + theta_Min + theta_Max)
    if ((np.issubdtype(sum_ords, np.signedinteger) or np.issubdtype(sum_ords, np.unsignedinteger)) 
        and na_Min >= 0 and nb_Min > 0 and nc_Min >= 0 and nd_Min >= 0 and nf_Min >= 0 and theta_Min >= 0) is False:
        sys.exit("Error! na, nc, nd, nf, theta must be positive integers, nb must be strictly positive integer")
    #        return 0.,0.,0.,0.,0.,0.,0.,np.inf
    elif y.size != u.size:
        sys.exit("Error! y and u must have tha same length")
    #        return 0.,0.,0.,0.,0.,0.,0.,np.inf
    else:
        ystd, y = rescale(y)
        Ustd, u = rescale(u)
        IC_old = np.inf
        for i_a in range(na_Min, na_MAX):
            for i_b in range(nb_Min, nb_MAX):
                for i_c in range(nc_Min, nc_MAX):
                    for i_d in range(nd_Min, nd_MAX):
                        for i_f in range(nf_Min, nf_MAX):  
                            for i_t in range(theta_Min, theta_Max):
                                useless1, useless2, useless3, useless4, Vn, y_id = GEN_RLS_id(id_method, y, u, i_a, i_b, i_c, i_d, i_f, i_t, max_iterations)
                                IC = information_criterion(i_a + i_b + i_c + i_d + i_f, y.size - max(i_a, i_b + i_t, i_c, i_d, i_f), Vn * 2, method)
                                # --> nota: non mi torna cosa scritto su ARMAX
                                if IC < IC_old:
                                    na_min, nb_min, nc_min, nd_min, nf_min, theta_min = i_a, i_b, i_c, i_d, i_f, i_t
                                    IC_old = IC
        print("suggested orders are: Na=", na_min, "; Nb=", nb_min, "; Nc=", nc_min, "; Nd=", nd_min, "; Nf=", nf_min, "; Delay: ", theta_min)
        
        # rerun identification
        NUM, DEN, NUMH, DENH, Vn, y_id = GEN_RLS_id(id_method, y, u, na_min, nb_min, nc_min, nd_min, nf_min, theta_min, max_iterations)
        Y_id = np.atleast_2d(y_id) * ystd
        
        # rescale NUM coeff
        NUM[theta_min:nb_min + theta_min] = NUM[theta_min:nb_min + theta_min] * ystd / Ustd
        
        # FdT
        g_identif = cnt.tf(NUM, DEN, tsample)
        h_identif = cnt.tf(NUMH, DENH, tsample)
        return na_min, nb_min, nc_min, nd_min, nf_min, theta_min, g_identif, h_identif, NUM, DEN, Vn, Y_id


# creating object GEN model
class GEN_model(object):
    def __init__(self, na, nb, nc, nd, nf, theta, ts, NUMERATOR, DENOMINATOR, G, H, Vn, Yid):
        self.na = na
        self.nb = nb
        self.nc = nc
        self.nc = nd
        self.nc = nf
        self.theta = theta
        self.ts = ts
        self.NUMERATOR = NUMERATOR
        self.DENOMINATOR = DENOMINATOR
        self.G = G
        self.H = H
        self.Vn = Vn
        self.Yid = Yid
