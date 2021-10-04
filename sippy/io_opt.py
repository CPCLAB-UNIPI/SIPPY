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


def GEN_id(id_method, y, u, na, nb, nc, nd, nf, theta, max_iterations, st_m, st_c):

    ylength = y.size
    
    # max predictable order 
    val = max(nb + theta, na, nc, nd, nf)
        
    # input/output number
    m = 1; p = 1
    
    # number of optimization variables
    n_coeff = na + nb + nc + nd + nf
    
    # Calling the optimization problem
    (solver, w_lb, w_ub, g_lb, g_ub) = opt_id(m, p, na, np.array([nb]), nc, nd, nf, n_coeff, np.array([theta]), val, np.atleast_2d(u), y, id_method, max_iterations, st_m, st_c)
            
    # Set first-guess solution        
    w_0 = np.zeros((1,n_coeff))
    w_y = np.zeros((1,ylength))
    w_0 = np.hstack([w_0,w_y])
    if id_method == 'BJ' or id_method == 'GEN' or id_method == 'ARARX' or id_method == 'ARARMAX':
        w_0 = np.hstack([w_0,w_y,w_y])
           

    # Call the NLP solver
    sol = solver(lbx = w_lb,
         ubx = w_ub,
         x0  = w_0,
         lbg = g_lb,
         ubg = g_ub
         )

    # model output: info from the solver
    f_opt = sol["f"]                                # objective function
    x_opt = sol["x"]                                # optimization variables = model coefficients
    iterations = solver.stats()['iter_count']       # iteration number
    y_id = x_opt[-ylength:].full()[:,0]             # model output
    THETA = np.array(x_opt[:n_coeff])[:,0]
   
    # estimated error norm
    Vn = old_div((np.linalg.norm((y_id - y), 2) ** 2), (2 * ylength))
    
    # building TF coefficient vectors
    valH = max(nc, na + nd)
    valG = max(nb + theta, na + nf)   
     
    # G    
    # numG (B)
    if id_method == 'ARMA':
        NUM = 1.0     
    else: 
        NUM = np.zeros(valG)
        NUM[theta:nb + theta] = THETA[na:nb+na]
    # denG (A*F)
    A = cnt.tf(np.hstack((1, np.zeros((na)))), np.hstack((1, THETA[:na])),1)
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


def select_order_GEN(id_method, y, u, tsample=1., na_ord=[0, 5], nb_ord=[1, 5], nc_ord=[0, 5], nd_ord=[0, 5], nf_ord=[0, 5], delays=[0, 5], method='AIC', max_iterations = 200, st_m = 1.0, st_c = False):
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
    if ((np.issubdtype(sum_ords, np.signedinteger) or np.issubdtype(sum_ords, np.unsignedinteger)) and na_Min >= 0 and nb_Min > 0 and nc_Min >= 0 and nd_Min >= 0 and nf_Min >= 0 and theta_Min >= 0) is False:
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
                                useless1, useless2, useless3, useless4, Vn, y_id = GEN_id(id_method, y, u, i_a, i_b, i_c, i_d, i_f, i_t, max_iterations, st_m, st_c)
                                IC = information_criterion(i_a + i_b + i_c + i_d + i_f, y.size - max(i_a, i_b + i_t, i_c, i_d, i_f), Vn * 2, method)
                                # --> nota: non mi torna cosa scritto su ARMAX
                                if IC < IC_old:
                                    na_min, nb_min, nc_min, nd_min, nf_min, theta_min = i_a, i_b, i_c, i_d, i_f, i_t
                                    IC_old = IC
        print("suggested orders are: Na=", na_min, "; Nb=", nb_min, "; Nc=", nc_min, "; Nd=", nd_min, "; Nf=", nf_min, "; Delay: ", theta_min)
        
        # rerun identification
        NUM, DEN, NUMH, DENH, Vn, y_id = GEN_id(id_method, y, u, na_min, nb_min, nc_min, nd_min, nf_min, theta_min, max_iterations, st_m, st_c)
        Y_id = np.atleast_2d(y_id) * ystd
        
        # rescale NUM coeff
        if id_method != 'ARMA':
            NUM[theta_min:nb_min + theta_min] = NUM[theta_min:nb_min + theta_min] * ystd / Ustd
        
        # FdT
        G = cnt.tf(NUM, DEN, tsample)
        H = cnt.tf(NUMH, DENH, tsample)
        
        check_st_H = np.zeros(1) if id_method == 'OE' else np.abs(cnt.pole(H))
        if max(np.abs(cnt.pole(G))) > 1.0 or max(check_st_H) > 1.0:
            print("Warning: One of the identified system is not stable")
            if st_c is True:
                print(f"Infeasible solution: the stability constraint has been violated, since the maximum pole is {max(max(np.abs(cnt.pole(H))),max(np.abs(cnt.pole(G))))} \
                          ... against the imposed stability margin {st_m}")
            else:
                print(f"Consider activating the stability constraint. The maximum pole is {max(max(np.abs(cnt.pole(H))),max(np.abs(cnt.pole(G))))}  ")
         
        return na_min, nb_min, nc_min, nd_min, nf_min, theta_min, G, H, NUM, DEN, Vn, Y_id


# creating object GEN model
class GEN_model(object):
    def __init__(self, na, nb, nc, nd, nf, theta, ts, NUMERATOR, DENOMINATOR, G, H, Vn, Yid):
        self.na = na
        self.nb = nb
        self.nc = nc
        self.nd = nd
        self.nf = nf
        self.theta = theta
        self.ts = ts
        self.NUMERATOR = NUMERATOR
        self.DENOMINATOR = DENOMINATOR
        self.G = G
        self.H = H
        self.Vn = Vn
        self.Yid = Yid
