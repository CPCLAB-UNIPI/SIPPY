"""Created on Sat Aug 12 2017

@author: Giuseppe Armenise
"""

import sys
import control.matlab as cnt
import numpy as np
from datetime import datetime
from sippy_unipi import functionset as fset


def ARX_MISO_id(y, u, ystd, ustd, na, nb, theta):
    nb = np.array(nb)
    theta = np.array(theta)
    u = 1.0 * np.atleast_2d(u)
    ylength = y.size
    [udim, ulength] = u.shape
    # checking dimension
    if nb.size != udim:
        sys.exit(
            "Error! nb must be a matrix, whose dimensions must be equal to yxu"
        )
    #        return np.array([[1.]]),np.array([[0.]]),np.array([[0.]]),np.inf
    elif theta.size != udim:
        sys.exit("Error! theta matrix must have yxu dimensions")
    #        return np.array([[1.]]),np.array([[0.]]),np.array([[0.]]),np.inf
    else:
        nbth = nb + theta
        # max predictable dimension
        val = max(na, np.max(nbth))
        N = ylength - val
        # regressor matrix
        phi = np.zeros(na + np.sum(nb[:]))
        PHI = np.zeros((N, na + np.sum(nb[:])))
        for k in range(N):
            phi[0:na] = -y[k + val - 1 :: -1][0:na]
            for nb_i in range(udim):
                phi[
                    na + np.sum(nb[0:nb_i]) : na + np.sum(nb[0 : nb_i + 1])
                ] = u[nb_i, :][val + k - 1 :: -1][
                    theta[nb_i] : nb[nb_i] + theta[nb_i]
                ]
            PHI[k, :] = phi
        # coefficient vector
        THETA = np.dot(np.linalg.pinv(PHI), y[val::])
        # model output
        y_id0 = np.dot(PHI, THETA)
        # estimated error norm
        Vn = (np.linalg.norm((y_id0 - y[val::]), 2) ** 2) / (2 * N)
        # adding non-identified outputs
        y_id = np.hstack((y[:val], y_id0)) * ystd
        DEN = np.zeros((udim, val + 1))
        NUMH = np.zeros((1, val + 1))
        NUMH[0, 0] = 1.0
        DEN[:, 0] = np.ones(udim)
        NUM = np.zeros((udim, val))
        for k in range(udim):
            THETA[na + np.sum(nb[0:k]) : na + np.sum(nb[0 : k + 1])] = (
                THETA[na + np.sum(nb[0:k]) : na + np.sum(nb[0 : k + 1])]
                * ystd
                / ustd[k]
            )
            NUM[k, theta[k] : theta[k] + nb[k]] = THETA[
                na + np.sum(nb[0:k]) : na + np.sum(nb[0 : k + 1])
            ]
            DEN[k, 1 : na + 1] = THETA[0:na]
        return DEN, NUM, NUMH, Vn, y_id


# MIMO function
def ARX_MIMO_id(y, u, ystd, ustd, na, nb, theta, tsample=1.0):
    na = np.array(na)
    nb = np.array(nb)
    theta = np.array(theta)
    [ydim, ylength] = y.shape
    [udim, ulength] = u.shape
    [th1, th2] = theta.shape
    # check dimensions
    sum_ords = np.sum(nb) + np.sum(na) + np.sum(theta)
    if na.size != ydim:
        sys.exit(
            "Error! na must be a vector, whose length must be equal to y dimension"
        )
    #        return 0.,0.,0.,0.,np.inf
    elif nb[:, 0].size != ydim:
        sys.exit(
            "Error! nb must be a matrix, whose dimensions must be equal to yxu"
        )
    #        return 0.,0.,0.,0.,np.inf
    elif th1 != ydim:
        sys.exit("Error! theta matrix must have yxu dimensions")
    #        return 0.,0.,0.,0.,np.inf
    elif not (
        (
            np.issubdtype(sum_ords, np.signedinteger)
            or np.issubdtype(sum_ords, np.unsignedinteger)
        )
        and np.min(nb) >= 0
        and np.min(na) >= 0
        and np.min(theta) >= 0
    ):
        sys.exit(
            "Error! na, nb, theta must contain only positive integer elements"
        )
    #        return 0.,0.,0.,0.,np.inf
    else:
        # preallocation
        Vn_tot = 0.0
        NUMERATOR = []
        DENOMINATOR = []
        DENOMINATOR_H = []
        NUMERATOR_H = []
        Y_id = np.zeros((ydim, ylength))
        # identification in MISO approach
        for i in range(ydim):
            DEN, NUM, NUMH, Vn, y_id = ARX_MISO_id(
                y[i, :], u, ystd[i,0], ustd, na[i], nb[i, :], theta[i, :]
            )
            # append values to vectors
            DENOMINATOR.append(DEN.tolist())
            NUMERATOR.append(NUM.tolist())
            NUMERATOR_H.append(NUMH.tolist())
            DENOMINATOR_H.append([DEN.tolist()[0]])
            Vn_tot = Vn_tot + Vn
            Y_id[i, :] = y_id
            
        # explained variance percentage on identification data
        EV = fset.compute_EV(y*ystd, Y_id)
        # fit percentage on identification data
        FIT = fset.compute_FIT(y*ystd, Y_id)
        # FdT
        G = cnt.tf(NUMERATOR, DENOMINATOR, tsample)
        H = cnt.tf(NUMERATOR_H, DENOMINATOR_H, tsample)
        
        return DENOMINATOR, NUMERATOR, G, H, Vn_tot, Y_id, EV, FIT


# class ARX MIMO and corresponding report class
class ARX_MIMO_model(object):
    def __init__(self,
                 na,
                 nb,
                 theta,
                 ts,
                 NUMERATOR,
                 DENOMINATOR,
                 G,
                 H,
                 Vn,
                 Yid,
                 centering,
                 y_cent,
                 u_cent,
                 N,
                 EV,
                 FIT
                 ):
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
        
        self.Report = ARX_MIMO_Report(centering, "ARX MIMO", EV, FIT)
        self.Report.set_data_used(y_cent, u_cent, ts, N)
    
class ARX_MIMO_Report:
    def __init__(self, centering, method, EV, FIT):
        self.Centering = centering        # 'None', 'InitVal', 'MeanVal'
        self.Method = method            
        self.Status = "Estimated using MIMO ARX"
        self.OptionsUsed = {}             # futuro
        self.DataUsed = {}                # Used Data informations 
        self.EV  = EV                     # Explained Variance in percentage
        self.FIT = FIT                    # Fit percentage
        self.Timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def set_data_used(self, y_cent, u_cent, ts, data_length):
        Nu = len(u_cent)
        Ny = len(y_cent)

        self.DataUsed = {
            "Name": {
                "Inputs":  [f"u{i+1}" for i in range(Nu)],
                "Outputs": [f"y{i+1}" for i in range(Ny)],
            },
            "Length": data_length,   
            "Ts": ts,
            "InputCentering": u_cent,
            "OutputCentering": y_cent
        }
