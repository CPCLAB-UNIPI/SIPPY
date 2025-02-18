"""
Created on 2021

@author: RBdC & MV
"""

import control.matlab as cnt
import numpy as np

from .functionset_OPT import opt_id


def GEN_id(id_method, y, u, na, nb, nc, nd, nf, theta, max_iterations, st_m, st_c):
    ylength = y.size

    # max predictable order
    val = max(nb + theta, na, nc, nd, nf)

    # input/output number
    m = 1
    p = 1

    # number of optimization variables
    n_coeff = na + nb + nc + nd + nf

    # Calling the optimization problem
    (solver, w_lb, w_ub, g_lb, g_ub) = opt_id(
        m,
        p,
        na,
        np.array([nb]),
        nc,
        nd,
        nf,
        n_coeff,
        np.array([theta]),
        val,
        np.atleast_2d(u),
        y,
        id_method,
        max_iterations,
        st_m,
        st_c,
    )

    # Set first-guess solution
    w_0 = np.zeros((1, n_coeff))
    w_y = np.zeros((1, ylength))
    w_0 = np.hstack([w_0, w_y])
    if (
        id_method == "BJ"
        or id_method == "GEN"
        or id_method == "ARARX"
        or id_method == "ARARMAX"
    ):
        w_0 = np.hstack([w_0, w_y, w_y])

    # Call the NLP solver
    sol = solver(lbx=w_lb, ubx=w_ub, x0=w_0, lbg=g_lb, ubg=g_ub)

    # model output: info from the solver
    # f_opt = sol["f"]  # objective function
    x_opt = sol["x"]  # optimization variables = model coefficients
    # iterations = solver.stats()["iter_count"]  # iteration number
    y_id = x_opt[-ylength:].full()[:, 0]  # model output
    THETA = np.array(x_opt[:n_coeff])[:, 0]

    # estimated error norm
    Vn = (np.linalg.norm((y_id - y), 2) ** 2) / (2 * ylength)

    # building TF coefficient vectors
    valH = max(nc, na + nd)
    valG = max(nb + theta, na + nf)

    # G
    # numG (B)
    if id_method == "ARMA":
        NUM = np.ones(1)
    else:
        NUM = np.zeros(valG)
        NUM[theta : nb + theta] = THETA[na : nb + na]
    # denG (A*F)
    A = cnt.tf(np.hstack((1, np.zeros(na))), np.hstack((1, THETA[:na])), 1)
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
