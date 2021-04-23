# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 2017

@author: Giuseppe Armenise
"""
from __future__ import absolute_import, division, print_function

import sys
from builtins import object
import control.matlab as cnt
from .functionset import *
# from functionset import *


def ARX_id(y, u, na, nb, theta):
    # max predictable order 
    val = max(na, nb + theta)
    N = y.size - val
    phi = np.zeros(na + nb)
    PHI = np.zeros((N, na + nb))
    for i in range(N):
        phi[0:na] = -y[i + val - 1::-1][0:na]
        phi[na:na + nb] = u[val + i - 1::-1][theta:nb + theta]
        PHI[i, :] = phi
    # coeffiecients
    THETA = np.dot(np.linalg.pinv(PHI), y[val::])
    # model Output
    y_id0 = np.dot(PHI, THETA)
    # estimated error norm
    Vn = old_div((np.linalg.norm((y_id0 - y[val::]), 2) ** 2), (2 * N))
    # adding non-identified outputs
    y_id = np.hstack((y[:val], y_id0))
    NUM = np.zeros(val)
    # numerator
    NUM[theta:nb + theta] = THETA[na::]
    DEN = np.zeros(val + 1)
    DEN[0] = 1.
    # denominator
    DEN[1:na + 1] = THETA[0:na]
    NUMH = np.zeros(val + 1)
    NUMH[0] = 1.
    
    return NUM, DEN, NUMH, Vn, y_id


def select_order_ARX(y, u, tsample=1., na_ord=[0, 5], nb_ord=[1, 5], delays=[0, 5], method='AIC'):
    # order ranges
    na_Min = min(na_ord)
    na_MAX = max(na_ord) + 1
    nb_Min = min(nb_ord)
    nb_MAX = max(nb_ord) + 1
    theta_Min = min(delays)
    theta_Max = max(delays) + 1
    # check orders
    sum_ords = np.sum(na_Min + na_MAX + nb_Min + nb_MAX + theta_Min + theta_Max)
    if ((np.issubdtype(sum_ords, np.signedinteger) or np.issubdtype(sum_ords, np.unsignedinteger)) 
        and na_Min >= 0 and nb_Min > 0 and theta_Min >= 0) == False:
        sys.exit("Error! na, theta must be positive integers, nb must be strictly positive integer")
    #        return 0.,0.,0.,0.,0.,0.,0.,np.inf
    elif y.size != u.size:
        sys.exit("Error! y and u must have tha same length")
    #        return 0.,0.,0.,0.,0.,0.,0.,np.inf
    else:
        ystd, y = rescale(y)
        Ustd, u = rescale(u)
        IC_old = np.inf
        for i in range(na_Min, na_MAX):
            for j in range(nb_Min, nb_MAX):
                for k in range(theta_Min, theta_Max):
                    useless1, useless2, useless3, Vn, y_id = ARX_id(y, u, i, j, k)
                    IC = information_criterion(i + j, y.size - max(i, j + k), Vn * 2, method)
                    if IC < IC_old:
                        na_min, nb_min, theta_min = i, j, k
                        IC_old = IC
        print("suggested orders are: Na=", na_min, "; Nb=", nb_min, "Delay: ", theta_min)
        # rerun identification
        NUM, DEN, NUMH, Vn, y_id = ARX_id(y, u, na_min, nb_min, theta_min)
        Y_id = np.atleast_2d(y_id) * ystd
        NUM[theta_min:nb_min + theta_min] = NUM[theta_min:nb_min + theta_min] * ystd / Ustd
        # FdT
        g_identif = cnt.tf(NUM, DEN, tsample)
        h_identif = cnt.tf(NUMH, DEN, tsample)
        return na_min, nb_min, theta_min, g_identif, h_identif, NUM, DEN, Vn, Y_id


# creating object ARX model
class ARX_model(object):
    def __init__(self, na, nb, theta, ts, NUMERATOR, DENOMINATOR, G, H, Vn, Yid):
        self.na = na
        self.nb = nb
        self.theta = theta
        self.ts = ts
        self.NUMERATOR = NUMERATOR
        self.DENOMINATOR = DENOMINATOR
        self.G = G
        self.H = H
        self.Vn = Vn
        self.Yid = Yid
