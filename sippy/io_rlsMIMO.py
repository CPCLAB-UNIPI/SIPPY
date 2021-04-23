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
# from functionset import *


def GEN_RLS_MISO_id(id_method, y, u, na, nb, nc, nd, nf, theta, max_iterations):
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
            
        # max number of non predictable data
        val = max(na, np.max(nbth), nc, nd, nf)
        # whole data
        N = ylength
        
        # Total Order: both LTI and time varying part
        nt = na + np.sum(nb[:]) + nc + nd + nf + 1
        nh = max([na,nc,nf])

        
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
        #eta = np.zeros(N)
        # Forgetting factors
        L_t = 1
        l_t = L_t*np.ones(N)
        #
        Yp = y.copy()
        E = np.zeros(N)
        fi = np.zeros((1,nt-1,N))
       
        ## Propagation
        for k in range(N):
            if k > val:
                ## Step 1: Regressor vector
                vecY = y[k-na:k][::-1]                 # Y vector
                vecYp = Yp[k-nf:k][::-1]               # Yp vector
                #
                vecU = []
                for nb_i in range(udim):               # U vector   
                    vecu = u[nb_i, :][k-nb[nb_i]-theta[nb_i]:k-theta[nb_i]][::-1]
                    vecU = np.hstack((vecU,vecu))
                # 
                #vecE = E[k-nh:k][::-1]                   # E vector
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
                Yp[k] = np.dot(phi.T,teta[:,k]) #+ eta[k]
                E[k] = y[k] - Yp[k]

                ## Step 5. Parameter estimate covariance update
                P_t[:,:,k] = (1/l_t[k-1])*(np.dot(np.eye(nt-1) - np.dot(K_t[:,k:k+1],phi.T),P_t[:,:,k-1]))

                ## Step 6: Forgetting factor update
                l_t[k] = 1.0
        
        
        # Error Norm
        Vn = old_div((np.linalg.norm(y - Yp, 2) ** 2), (2 * (N-val)))
        
        # Model Output
        y_id = Yp*ystd 
        
        # Parameters
        THETA = teta[:,-1]
            
        #if iterations >= max_iterations:
         #   print("Warning! Reached maximum iterations")
          #  Reached_max = True
        
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
        if id_method == 'OE':
            F = cnt.tf(np.hstack((1, np.zeros((nf)))), np.hstack((1, THETA[:nf])),1)
        else:
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
        ng = nf if id_method == 'OE' else na
        for k in range(udim):
            if id_method != 'ARMA':
                THETA[ng + np.sum(nb[0:k]):ng + np.sum(nb[0:k + 1])] = THETA[ng + np.sum(nb[0:k]):ng + np.sum(nb[0:k + 1])] * ystd / Ustd[k]
                NUM[k, theta[k]:theta[k] + nb[k]] = THETA[ng + np.sum(nb[0:k]):ng + np.sum(nb[0:k + 1])]
            #DEN[k, 1:den.shape[1] + 1] = den
            #DEN[k,:] = den
            DEN[k, 0:na+nf+1] = denG
        
        return DEN, NUM, NUMH, DENH, Vn, y_id, Reached_max


# MIMO function
def GEN_MIMO_id(id_method, y, u, na, nb, nc, nd, nf, theta, tsample=1., max_iterations=100):
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
            DEN, NUM, NUMH, DENH, Vn, y_id, Reached_max = GEN_RLS_MISO_id(id_method, y[i, :], u, na[i], nb[i, :], nc[i], nd[i], nf[i], theta[i, :], max_iterations)
            if Reached_max == True:
                print("at ", (i + 1), "° output")
                print("-------------------------------------")
            # append values to vectors    
            DENOMINATOR.append(DEN.tolist())
            NUMERATOR.append(NUM.tolist())
            NUMERATOR_H.append(NUMH.tolist())
            DENOMINATOR_H.append([DENH.tolist()[0]])
            Vn_tot = Vn + Vn_tot
            Y_id[i,:] = y_id
        # FdT
        G = cnt.tf(NUMERATOR, DENOMINATOR, tsample)
        H = cnt.tf(NUMERATOR_H, DENOMINATOR_H, tsample)
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
