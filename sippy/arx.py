"""
Created on Wed Jul 26 2017

@author: Giuseppe Armenise
"""

import numpy as np


def ARX_id(y, u, na, nb, theta):
    # max predictable order
    val = max(na, nb + theta)
    N = y.size - val
    phi = np.zeros(na + nb)
    PHI = np.zeros((N, na + nb))
    for i in range(N):
        phi[0:na] = -y[i + val - 1 :: -1][0:na]
        phi[na : na + nb] = u[val + i - 1 :: -1][theta : nb + theta]
        PHI[i, :] = phi
    # coeffiecients
    THETA = np.dot(np.linalg.pinv(PHI), y[val::])
    # model Output
    y_id0 = np.dot(PHI, THETA)
    # estimated error norm
    Vn = (np.linalg.norm((y_id0 - y[val::]), 2) ** 2) / (2 * N)
    # adding non-identified outputs
    y_id = np.hstack((y[:val], y_id0))
    NUM = np.zeros(val)
    # numerator
    NUM[theta : nb + theta] = THETA[na::]
    DEN = np.zeros(val + 1)
    DEN[0] = 1.0
    # denominator
    DEN[1 : na + 1] = THETA[0:na]
    NUMH = np.zeros(val + 1)
    NUMH[0] = 1.0

    return NUM, DEN, NUMH, DEN, Vn, y_id
