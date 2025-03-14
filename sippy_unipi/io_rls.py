"""
Created on 2021

@author: RBdC & MV
"""

import numpy as np

from ._typing import RLSMethods
from .functionset import rescale
from .utils import build_tfs, common_setup, validate_and_prepare_inputs

# ----------------- Helper Functions -----------------


def _initialize_parameters(
    N: int, nt: int, y: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Initialize P_t, teta, eta, and Yp."""
    Beta = 1e4
    p_t = Beta * np.eye(nt, nt)
    P_t = np.repeat(p_t[:, :, np.newaxis], N, axis=2)
    teta = np.zeros((nt, N))
    eta = np.zeros(N)
    Yp = y.copy() if y is not None else np.zeros(N)
    return P_t, teta, eta, Yp


def _compute_error_norm(y: np.ndarray, Yp: np.ndarray, val: int) -> float:
    """Calculate the normalized prediction error."""
    return float(np.linalg.norm(y - Yp, 2) ** 2) / (2 * (y.size - val))


def _propagate_parameters(
    y,
    u,
    na,
    nb,
    nc,
    nd,
    nf,
    theta,
    id_method,
    val,
    P_t,
    teta,
    eta,
    Yp,
    nt,
):
    N = y.size
    # Gain
    K_t = np.zeros((nt, N))

    # Forgetting factors
    L_t = 1
    l_t = L_t * np.ones(N)
    #
    E = np.zeros(N)
    fi = np.zeros((1, nt, N))

    # Propagation
    for k in range(N):
        if k > val:
            # Step 1: Regressor vector
            vecY = y[k - na : k][::-1]  # Y vector
            vecYp = Yp[k - nf : k][::-1]  # Yp vector
            #
            # vecE = E[k-nh:k][::-1]                     # E vector

            vecU = []
            for nb_i in range(nb.size):  # U vector
                vecu = u[nb_i][k - nb[nb_i] - theta[nb_i] : k - theta[nb_i]][
                    ::-1
                ]
                vecU = np.hstack((vecU, vecu))  # U vector

                vecE = E[k - nc : k][::-1]

            # choose input-output model
            if id_method == "ARMAX":
                fi[:, :, k] = np.hstack((-vecY, vecU, vecE))
            elif id_method == "ARX":
                fi[:, :, k] = np.hstack((-vecY, vecU))
            elif id_method == "OE":
                fi[:, :, k] = np.hstack((-vecYp, vecU))
            elif id_method == "FIR":
                fi[:, :, k] = vecU
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
            Yp[k] = np.dot(phi.T, teta[:, k]).item() + eta[k]
            E[k] = y[k] - Yp[k]

            # Step 5. Parameter estimate covariance update
            P_t[:, :, k] = (1 / l_t[k - 1]) * (
                np.dot(
                    np.eye(nt) - np.dot(K_t[:, k : k + 1], phi.T),
                    P_t[:, :, k - 1],
                )
            )

            # Step 6: Forgetting factor update
            l_t[k] = 1.0
    return teta, Yp


# ----------------- Main Functions -----------------


def GEN_RLS_id(
    y: np.ndarray,
    u: np.ndarray,
    id_method: RLSMethods,
    na: int,
    nb: int | np.ndarray,
    nc: int,
    nd: int,
    nf: int,
    theta: int | np.ndarray,
    **_,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    # Input handling and validation
    u, nb, theta, udim = validate_and_prepare_inputs(u, nb, theta)
    val, nt = common_setup(na, nb, nc, nd, nf, theta)

    # Parameter initialization
    P_t, teta, eta, Yp = _initialize_parameters(y.size, nt)

    # Propagate parameters
    teta, Yp = _propagate_parameters(
        y, u, na, nb, nc, nd, nf, theta, id_method, val, P_t, teta, eta, Yp, nt
    )

    # Compute results
    Vn = _compute_error_norm(y, Yp, val)
    THETA = teta[:, -1]
    NUM, DEN, NUMH, DENH = build_tfs(
        THETA, na, nb, nc, nd, nf, theta, id_method, udim
    )

    return NUM.squeeze(), DEN.squeeze(), NUMH.squeeze(), DENH.squeeze(), Vn, Yp


def GEN_RLS_MISO_id(
    y: np.ndarray,
    u: np.ndarray,
    id_method: RLSMethods,
    na: int,
    nb: np.ndarray,
    nc: int,
    nd: int,
    nf: int,
    theta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    # Input handling and scaling
    u, nb, theta, udim = validate_and_prepare_inputs(u, nb, theta)
    y_std, y = rescale(y)
    U_std = np.zeros(udim)
    for j in range(udim):
        U_std[j], u[j] = rescale(u[j])

    # Common setup
    val, nt = common_setup(na, nb, nc, nd, nf, theta)
    P_t, teta, eta, Yp = _initialize_parameters(y.size, nt, y)

    # Propagate parameters
    teta, Yp = _propagate_parameters(
        y,
        u,
        na,
        nb,
        nc,
        nd,
        nf,
        theta,
        id_method,
        val,
        P_t,
        teta,
        eta,
        Yp,
        nt,
    )

    # Compute results
    Vn = _compute_error_norm(y, Yp, val)
    THETA = teta[:, -1]

    # Adjust coefficients for scaling
    start_idx = na if id_method != "OE" else nf
    for k in range(udim):
        coeffs = THETA[
            start_idx + np.sum(nb[:k]) : start_idx + np.sum(nb[: k + 1])
        ]
        THETA[start_idx + np.sum(nb[:k]) : start_idx + np.sum(nb[: k + 1])] = (
            coeffs * y_std / U_std[k]
        )

    NUM, DEN, NUMH, DENH = build_tfs(
        THETA, na, nb, nc, nd, nf, theta, id_method, udim, y_std, U_std
    )
    y_id = Yp * y_std

    return NUM, DEN, NUMH, DENH, Vn, y_id
