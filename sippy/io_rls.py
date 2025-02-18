"""
Created on 2021

@author: RBdC & MV
"""

import control.matlab as cnt
import numpy as np


def GEN_RLS_id(id_method, y, u, na, nb, nc, nd, nf, theta):
    ylength = y.size

    # max number of non predictable data
    nbth = nb + theta
    val = max(na, nbth, nc, nd, nf)
    # whole data
    N = ylength

    # Total Order: both LTI and time varying part
    nt = na + nb + nc + nd + nf + 1

    # Iterative Identification Algorithm

    # Parameters Initialization
    # Confidence Parameter
    Beta = 1e4
    # Covariance matrix of parameter teta
    p_t = Beta * np.eye(nt - 1, nt - 1)
    P_t = np.zeros((nt - 1, nt - 1, N))
    for i in range(N):
        P_t[:, :, i] = p_t
    # Gain
    K_t = np.zeros((nt - 1, N))

    # First estimatate
    teta = np.zeros((nt - 1, N))
    eta = np.zeros(N)
    # Forgetting factors
    L_t = 1
    l_t = L_t * np.ones(N)
    #
    Yp = np.zeros(N)
    E = np.zeros(N)
    fi = np.zeros((1, nt - 1, N))

    # Propagation
    for k in range(N):
        if k > val:
            # Step 1: Regressor vector
            vecY = y[k - na : k][::-1]  # Y vector
            vecYp = Yp[k - nf : k][::-1]  # Yp vector
            #
            vecU = u[k - nb - theta : k - theta][::-1]  # U vector
            #
            # vecE = E[k-nh:k][::-1]                     # E vector
            vecE = E[k - nc : k][::-1]

            # choose input-output model
            if id_method == "ARMAX":
                fi[:, :, k] = np.hstack((-vecY, vecU, vecE))
            elif id_method == "ARX":
                fi[:, :, k] = np.hstack((-vecY, vecU))
            elif id_method == "OE":
                fi[:, :, k] = np.hstack((-vecYp, vecU))
            elif id_method == "FIR":
                fi[:, :, k] = np.hstack(vecU)
            phi = fi[:, :, k].T

            # Step 2: Gain Update
            # Gain of parameter teta
            K_t[:, k : k + 1] = np.dot(
                np.dot(P_t[:, :, k - 1], phi),
                np.linalg.inv(
                    l_t[k - 1] + np.dot(np.dot(phi.T, P_t[:, :, k - 1]), phi)
                ),
            )

            # Step 3: Parameter Update
            teta[:, k] = teta[:, k - 1] + np.dot(
                K_t[:, k : k + 1], (y[k] - np.dot(phi.T, teta[:, k - 1]))
            )

            # Step 4: A posteriori prediction-error
            Yp[k] = np.dot(phi.T, teta[:, k]) + eta[k]
            E[k] = y[k] - Yp[k]

            # Step 5. Parameter estimate covariance update
            P_t[:, :, k] = (1 / l_t[k - 1]) * (
                np.dot(
                    np.eye(nt - 1) - np.dot(K_t[:, k : k + 1], phi.T),
                    P_t[:, :, k - 1],
                )
            )

            # Step 6: Forgetting factor update
            l_t[k] = 1.0

    # Error Norm
    Vn = (np.linalg.norm(y - Yp, 2) ** 2) / (2 * (N - val))

    # Model Output
    y_id = Yp

    # Parameters
    THETA = teta[:, -1]

    # building TF coefficient vectors
    valH = max(nc, na + nd)
    valG = max(nb + theta, na + nf)

    # G
    # numG (B)
    # TODO: find out if depreciated or not needed
    if id_method == "ARMA":
        NUM = np.array(1)
    else:
        NUM = np.zeros(valG)
        ng = nf if id_method == "OE" else na
        NUM[theta : nb + theta] = THETA[ng : nb + ng]
    # denG (A*F)
    A = cnt.tf(np.hstack((1, np.zeros(na))), np.hstack((1, THETA[:na])), 1)

    if id_method == "OE":
        F = cnt.tf(np.hstack((1, np.zeros(nf))), np.hstack((1, THETA[:nf])), 1)
    else:
        F = cnt.tf(
            np.hstack((1, np.zeros(nf))),
            np.hstack((1, THETA[na + nb + nc + nd : na + nb + nc + nd + nf])),
            1,
        )
    if A is not None:
        _, deng = cnt.tfdata(A * F)
    denG = np.array(deng[0])
    DEN = np.zeros(valG + 1)
    DEN[0 : na + nf + 1] = denG

    # H
    # numH (C)
    if id_method == "OE":
        NUMH = 1
    else:
        NUMH = np.zeros(valH + 1)
        NUMH[0] = 1.0
        NUMH[1 : nc + 1] = THETA[na + nb : na + nb + nc]
    # denH (A*D)
    D = cnt.tf(
        np.hstack((1, np.zeros(nd))),
        np.hstack((1, THETA[na + nb + nc : na + nb + nc + nd])),
        1,
    )
    if A is not None:
        _, denh = cnt.tfdata(A * D)
    denH = np.array(denh[0])
    DENH = np.zeros(valH + 1)
    DENH[0 : na + nd + 1] = denH

    return NUM, DEN, NUMH, DENH, Vn, y_id
