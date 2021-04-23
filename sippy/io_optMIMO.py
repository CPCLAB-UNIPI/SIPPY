# -*- coding: utf-8 -*-
"""
Created on 2021

@author: RBdC
"""
from __future__ import absolute_import, division, print_function
import control.matlab as cnt
import sys
from builtins import object

from .functionset import *
from .functionset_OPT import *

# from functionset import *


def GEN_MISO_id(id_method, y, u, na, nb, nc, nd, nf, theta, max_iterations, st_m, st_c):
    #nb = np.array(nb)
    #theta = np.array(theta)
    u = 1. * np.atleast_2d(u)
    ylength = y.size
    ystd, y = rescale(y)
    [udim, ulength] = u.shape
    eps = np.zeros(y.size)
    Reached_max = False
    # checking dimension
    if nb.size != udim:
        sys.exit("Error! nb must be a matrix, whose dimensions must be equal to yxu")
    #        return np.array([[1.]]),np.array([[0.]]),np.array([[0.]]),np.inf,Reached_max
    elif theta.size != udim:
        sys.exit("Error! theta matrix must have yxu dimensions")
    #        return np.array([[1.]]),np.array([[0.]]),np.array([[0.]]),np.inf,Reached_max
    else:
        nbth = nb + theta
        Ustd = np.zeros(udim)
        for j in range(udim):
            Ustd[j], u[j] = rescale(u[j])
        
        # max predictable dimension
        val = max(na, np.max(nbth), nc, nd, nf)
        
        # input/output number
        m = udim; p = 1
        
        # number of optimization variables
        n_coeff = na + np.sum(nb[:]) + nc + nd + nf
        
        
        # Build the optimization problem
        (solver, w_lb, w_ub, g_lb, g_ub) = opt_id(m, p, na, nb, nc, nd, nf, n_coeff, theta, val, np.atleast_2d(u), y, id_method, max_iterations, st_m, st_c)
        
        # Set first-guess solution        
        w_0 = np.zeros((1,n_coeff))
        w_0 = np.hstack([w_0,np.atleast_2d(y)])
        if id_method == 'BJ' or id_method == 'GEN' or id_method == 'ARARX' or id_method == 'ARARMAX':
            w_0 = np.hstack([w_0,np.atleast_2d(y),np.atleast_2d(y)])   
            
        
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
        y_id0 = x_opt[-ylength:].full()[:,0]            # model output
        THETA = np.array(x_opt[:n_coeff])[:,0]
        
        # Check iteration numbers
        if iterations >= max_iterations:
            print("Warning! Reached maximum iterations")
            Reached_max = True
        
             
        # estimated error norm
        Vn = old_div((np.linalg.norm((y_id0 - y), 2) ** 2), (2 * ylength))
        
        # rescaling Yid
        y_id = y_id0*ystd
        
        # building TF coefficient vectors
        valH = max(nc, na + nd)
        valG = max(np.max(nbth), na + nf)
        Nb = np.sum(nb[:])
        
        # H = (C/(A*D)) 
        if id_method == 'OE':
            NUMH = np.ones((1,1))
        else:   
            NUMH = np.zeros((1, valH + 1))
            NUMH[0, 0] = 1.
            NUMH[0, 1:nc + 1] = THETA[na+Nb:na+Nb+nc]
        #
        # DENH = np.zeros((1, val + 1))
        # DENH[0, 0] = 1.
        # DENH[0, 1:nd + 1] = THETA[Nb+na+nc:Nb+na+nc+nd]
        
        A = cnt.tf(np.hstack((1, np.zeros((na)))), np.hstack((1, THETA[:na])),1)
        D = cnt.tf(np.hstack((1, np.zeros((nd)))), np.hstack((1, THETA[na+Nb+nc:na+Nb+nc+nd])),1)
    
        _, denh = cnt.tfdata(A*D)
        denH = np.array(denh[0])
        DENH = np.zeros((1, valH + 1))
        DENH[0, 0:na+nd+1] = denH
        
         # G = (B/(A*F))
        F = cnt.tf(np.hstack((1, np.zeros((nf)))), np.hstack((1, THETA[na+Nb+nc+nd:na+Nb+nc+nd+nf])),1)
    
        _, deng = cnt.tfdata(A*F)      
        denG = np.array(deng[0])
        DEN = np.zeros((udim, valG + 1))    
        #DEN = np.zeros((udim, den.shape[1] + 1))
        #DEN = np.zeros((udim, den.shape[1]))
        #DEN[:, 0] = np.ones(udim)
        
        if id_method == 'ARMA':
            NUM = np.ones((udim, 1))     
        else:           
            NUM = np.zeros((udim, valG))
        #
        for k in range(udim):
            if id_method != 'ARMA':
                THETA[na + np.sum(nb[0:k]):na + np.sum(nb[0:k + 1])] = THETA[na + np.sum(nb[0:k]):na + np.sum(nb[0:k + 1])] * ystd / Ustd[k]
                NUM[k, theta[k]:theta[k] + nb[k]] = THETA[na + np.sum(nb[0:k]):na + np.sum(nb[0:k + 1])]
            #DEN[k, 1:den.shape[1] + 1] = den
            #DEN[k,:] = den
            DEN[k, 0:na+nf+1] = denG
        
        
        # check_stH = True if any(np.roots(DENH)>=1.0) else False
        # check_stG = True if any(np.roots(DEN)>=1.0) else False
        # if check_stH or check_stG:
        #     IDsys_unst = 
        # if st_c is True:
        #     if solver.stats()['return_status'] != 'Maximum_Iterations_Exceeded':
           
                
        return DEN, NUM, NUMH, DENH, Vn, y_id, Reached_max


# MIMO function
def GEN_MIMO_id(id_method, y, u, na, nb, nc, nd, nf, theta, tsample, max_iterations, st_m, st_c):
    na = np.array(na)
    nb = np.array(nb)
    nc = np.array(nc)
    nd = np.array(nd)
    nf = np.array(nf)
    theta = np.array(theta)
    [ydim, ylength] = y.shape
    [udim, ulength] = u.shape
    [th1, th2] = theta.shape
    # check dimension
    sum_ords = np.sum(nb) + np.sum(na) + np.sum(nc) + np.sum(nd) + np.sum(nf) + np.sum(theta)
    if na.size != ydim:
        sys.exit("Error! na must be a vector, whose length must be equal to y dimension")
    #        return 0.,0.,0.,0.,0.,0.,np.inf
    elif nb[:, 0].size != ydim:
        sys.exit("Error! nb must be a matrix, whose dimensions must be equal to yxu")
    #        return 0.,0.,0.,0.,0.,0.,np.inf
    elif nc.size != ydim:
        sys.exit("Error! nc must be a vector, whose length must be equal to y dimension")
    #        return 0.,0.,0.,0.,0.,0.,np.inf
    elif nd.size != ydim:
        sys.exit("Error! nd must be a vector, whose length must be equal to y dimension")
    #        return 0.,0.,0.,0.,0.,0.,np.inf  
    elif nf.size != ydim:
        sys.exit("Error! nf must be a vector, whose length must be equal to y dimension")
    #        return 0.,0.,0.,0.,0.,0.,np.inf              
    elif th1 != ydim:
        sys.exit("Error! theta matrix must have yxu dimensions")
    #        return 0.,0.,0.,0.,0.,0.,np.inf
    elif ((np.issubdtype(sum_ords, np.signedinteger) or np.issubdtype(sum_ords, np.unsignedinteger))
          and np.min(nb) >= 0 and np.min(na) >= 0 and np.min(nc) >= 0 and np.min(nd) >= 0 and np.min(nf) >= 0 and np.min(theta) >= 0) == False:
        sys.exit("Error! nf, nb, nc, nd, theta must contain only positive integer elements")
    #        return 0.,0.,0.,0.,0.,0.,np.inf
    else:
        # preallocation
        Vn_tot = 0.
        NUMERATOR = []
        DENOMINATOR = []
        DENOMINATOR_H = []
        NUMERATOR_H = []
        Y_id = np.zeros((ydim, ylength))
        # identification in MISO approach
        for i in range(ydim):
            DEN, NUM, NUMH, DENH, Vn, y_id, Reached_max = GEN_MISO_id(id_method, y[i, :], u, na[i], nb[i, :], nc[i], nd[i], nf[i],
                                                            theta[i, :], max_iterations, st_m, st_c)
            
            if Reached_max == True:
                print("at ", (i + 1), "° output")
                print("-------------------------------------")
            
            # append values to vectors    
            DENOMINATOR.append(DEN.tolist())
            NUMERATOR.append(NUM.tolist())
            NUMERATOR_H.append(NUMH.tolist())
            DENOMINATOR_H.append(DENH.tolist())
            #DENOMINATOR_H.append([DEN.tolist()[0]])
            Vn_tot = Vn + Vn_tot
            Y_id[i,:] = y_id
        # FdT
        G = cnt.tf(NUMERATOR, DENOMINATOR, tsample)
        H = cnt.tf(NUMERATOR_H, DENOMINATOR_H, tsample)
        
        check_st_H = np.zeros(1) if id_method == 'OE' else np.abs(cnt.pole(H))
        if max(np.abs(cnt.pole(G))) > 1.0 or max(check_st_H) > 1.0:
            print("Warning: One of the identified system is not stable")
            if st_c is True:
                print(f"Infeasible solution: the stability constraint has been violated, since the maximum pole is {max(max(np.abs(cnt.pole(H))),max(np.abs(cnt.pole(G))))} \
                      ... against the imposed stability margin {st_m}")
                           
        return DENOMINATOR, NUMERATOR, DENOMINATOR_H, NUMERATOR_H, G, H, Vn_tot, Y_id


# creating object GEN MIMO model
class GEN_MIMO_model(object):
    def __init__(self, na, nb, nc, nd, nf, theta, ts, NUMERATOR, DENOMINATOR, NUMERATOR_H, DENOMINATOR_H, G, H, Vn, Yid):
        self.na = na 
        self.nb = nb
        self.nc = nc
        self.nd = nd
        self.nf = nf
        self.theta = theta
        self.ts = ts
        self.NUMERATOR = NUMERATOR
        self.DENOMINATOR = DENOMINATOR
        self.NUMERATOR_H = NUMERATOR_H
        self.DENOMINATOR_H = DENOMINATOR_H
        self.G = G
        self.H = H
        self.Vn = Vn
        self.Yid = Yid
