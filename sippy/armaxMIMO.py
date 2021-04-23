# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 2017

@author: Giuseppe Armenise
"""
from __future__ import absolute_import, division, print_function
import control.matlab as cnt
import sys
from builtins import object

from .functionset import *
# from functionset import *


def ARMAX_MISO_id(y, u, na, nb, nc, theta, max_iterations):
    nb = np.array(nb)
    theta = np.array(theta)
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
        val = max(na, np.max(nbth), nc)
        # max predictable dimension 
        N = ylength - val
        # regressor matrix
        phi = np.zeros(na + np.sum(nb[:]) + nc)
        PHI = np.zeros((N, na + np.sum(nb[:]) + nc))
        for k in range(N):
            phi[0:na] = -y[k + val - 1::-1][0:na]
            for nb_i in range(udim):
                phi[na + np.sum(nb[0:nb_i]):na + np.sum(nb[0:nb_i + 1])] = \
                    u[nb_i, :][val + k - 1::-1][theta[nb_i]:nb[nb_i] + theta[nb_i]]
            PHI[k, :] = phi
        Vn = np.inf
        Vn_old = np.inf
        # coefficient vector
        THETA = np.zeros(na + np.sum(nb[:]) + nc)
        ID_THETA = np.identity(THETA.size)
        lambdak = 0.5
        iterations = 0
        while (Vn_old > Vn or iterations == 0) and iterations < max_iterations:
            THETA_old = THETA
            Vn_old = Vn
            iterations = iterations + 1
            for i in range(N):
                PHI[i, na + np.sum(nb[:]):na + np.sum(nb[:]) + nc] = eps[val + i - 1::-1][0:nc]
            THETA = np.dot(np.linalg.pinv(PHI), y[val::])
            Vn = old_div((np.linalg.norm(y[val::] - np.dot(PHI, THETA), 2) ** 2), (2 * N))
            THETA_new = THETA
            lambdak = 0.5
            while Vn > Vn_old:
                THETA = np.dot(ID_THETA * lambdak, THETA_new) + np.dot(ID_THETA * (1 - lambdak),
                                                                       THETA_old)
                # Model Output
                # y_id0 = np.dot(PHI,THETA)
                Vn = old_div((np.linalg.norm(y[val::] - np.dot(PHI,THETA), 2) ** 2), (2 * N))
                if lambdak < np.finfo(np.float32).eps:
                    THETA = THETA_old
                    Vn = Vn_old
                lambdak = old_div(lambdak, 2.)
            eps[val::] = y[val::] - np.dot(PHI, THETA)
            # adding non-identified outputs
            y_id = np.hstack((y[:val], np.dot(PHI,THETA)))*ystd  
            
        if iterations >= max_iterations:
            print("Warning! Reached maximum iterations")
            Reached_max = True
        DEN = np.zeros((udim, val + 1))
        NUMH = np.zeros((1, val + 1))
        NUMH[0, 0] = 1.
        NUMH[0, 1:nc + 1] = THETA[na + np.sum(nb[:])::]
        DEN[:, 0] = np.ones(udim)
        NUM = np.zeros((udim, val))
        for k in range(udim):
            THETA[na + np.sum(nb[0:k]):na + np.sum(nb[0:k + 1])] = THETA[
                                                                   na + np.sum(nb[0:k]):na + np.sum(
                                                                           nb[0:k + 1])] * ystd / \
                                                                   Ustd[k]
            NUM[k, theta[k]:theta[k] + nb[k]] = THETA[na + np.sum(nb[0:k]):na + np.sum(nb[0:k + 1])]
            DEN[k, 1:na + 1] = THETA[0:na]
        return DEN, NUM, NUMH, Vn, y_id, Reached_max


# MIMO function
def ARMAX_MIMO_id(y, u, na, nb, nc, theta, tsample=1., max_iterations=100):
    na = np.array(na)
    nb = np.array(nb)
    nc = np.array(nc)
    theta = np.array(theta)
    [ydim, ylength] = y.shape
    [udim, ulength] = u.shape
    [th1, th2] = theta.shape
    # check dimension
    sum_ords = np.sum(nb) + np.sum(na) + np.sum(nc) + np.sum(theta)
    if na.size != ydim:
        sys.exit("Error! na must be a vector, whose length must be equal to y dimension")
    #        return 0.,0.,0.,0.,0.,0.,np.inf
    elif nc.size != ydim:
        sys.exit("Error! nc must be a vector, whose length must be equal to y dimension")
    #        return 0.,0.,0.,0.,0.,0.,np.inf
    elif nb[:, 0].size != ydim:
        sys.exit("Error! nb must be a matrix, whose dimensions must be equal to yxu")
    #        return 0.,0.,0.,0.,0.,0.,np.inf
    elif th1 != ydim:
        sys.exit("Error! theta matrix must have yxu dimensions")
    #        return 0.,0.,0.,0.,0.,0.,np.inf
    elif ((np.issubdtype(sum_ords, np.signedinteger) or np.issubdtype(sum_ords, np.unsignedinteger)) 
          and np.min(nb) >= 0 and np.min(na) >= 0 and np.min(nc) >= 0 and np.min(theta) >= 0) == False:
        sys.exit("Error! na, nb, nc, theta must contain only positive integer elements")
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
            DEN, NUM, NUMH, Vn, y_id, Reached_max = ARMAX_MISO_id(y[i, :], u, na[i], nb[i, :], nc[i],
                                                            theta[i, :], max_iterations)
            if Reached_max == True:
                print("at ", (i + 1), "° output")
                print("-------------------------------------")
            # append values to vectors    
            DENOMINATOR.append(DEN.tolist())
            NUMERATOR.append(NUM.tolist())
            NUMERATOR_H.append(NUMH.tolist())
            DENOMINATOR_H.append([DEN.tolist()[0]])
            Vn_tot = Vn + Vn_tot
            Y_id[i,:] = y_id
        # FdT
        G = cnt.tf(NUMERATOR, DENOMINATOR, tsample)
        H = cnt.tf(NUMERATOR_H, DENOMINATOR_H, tsample)
        return DENOMINATOR, NUMERATOR, DENOMINATOR_H, NUMERATOR_H, G, H, Vn_tot, Y_id


# creating object ARMAX MIMO model
class ARMAX_MIMO_model(object):
    def __init__(self, na, nb, nc, theta, ts, NUMERATOR, DENOMINATOR, NUMERATOR_H, DENOMINATOR_H, G, H, Vn, Yid):
        self.na = na
        self.nb = nb
        self.nc = nc
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
