"""
Created on 2021

@author: RBdC & MV
"""

from warnings import warn

import numpy as np

from .functionset import rescale
from .functionset_OPT import opt_id
from .typing import IOMethods, OptMethods
from .utils import (
    build_tfs,
    check_valid_orders,
    common_setup,
    validate_and_prepare_inputs,
)

# ----------------- Helper Functions -----------------


def _build_initial_guess(y, n_coeff: int, id_method: IOMethods) -> np.ndarray:
    w_0 = np.zeros((1, n_coeff))
    w_y = np.atleast_2d(y)
    w_0 = np.hstack([w_0, w_y])
    if id_method in ["BJ", "GEN", "ARARX", "ARARMAX"]:
        w_0 = np.hstack([w_0, w_y, w_y])
    return w_0


def _extract_results(
    sol, n_coeff: int, ylength: int, y_std: float = 1.0
) -> tuple:
    x_opt = sol["x"]
    THETA = np.array(x_opt[:n_coeff])[:, 0]
    y_id0 = x_opt[-ylength:].full()[:, 0]
    y_id = y_id0 * y_std
    return THETA, y_id


# ----------------- Main Functions -----------------


def GEN_id(
    y: np.ndarray,
    u: np.ndarray,
    id_method: OptMethods,
    na: int,
    nb: int | np.ndarray,
    nc: int,
    nd: int,
    nf: int,
    theta: int | np.ndarray,
    max_iter: int,
    stab_marg: float,
    stab_cons: bool,
    adjust_B: bool = False,
    y_std: float = 1.0,
    U_std: np.ndarray = np.array([1.0]),
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.floating, np.ndarray
]:
    u, nb, theta, udim = validate_and_prepare_inputs(u, nb, theta)
    val, n_coeff = common_setup(na, nb, nc, nd, nf, theta)
    solver, w_lb, w_ub, g_lb, g_ub = opt_id(
        y,
        u,
        id_method,
        na,
        nb,
        nc,
        nd,
        nf,
        theta,
        max_iter,
        stab_marg,
        stab_cons,
        n_coeff,
        udim,
        val,
    )
    iterations = solver.stats()["iter_count"]
    if iterations >= max_iter:
        warn("Reached maximum number of iterations")

    w_0 = _build_initial_guess(y, n_coeff, id_method)
    sol = solver(lbx=w_lb, ubx=w_ub, x0=w_0, lbg=g_lb, ubg=g_ub)

    THETA, y_id = _extract_results(sol, n_coeff, y.size)
    y_id = y_id * y_std
    # TODO: this is currently implemented within build_tf_G()
    # Adjust B coefficients with scaling
    # if adjust_B:
    #     start_B = na
    #     for k in range(udim):
    #         end_Bk = start_B + np.sum(nb[:k])
    #         THETA[start_B:end_Bk] *= y_std / U_std[k]
    NUM, DEN, NUMH, DENH = build_tfs(
        THETA, na, nb, nc, nd, nf, theta, id_method, udim, y_std, U_std
    )

    Vn = np.linalg.norm(y_id - y, 2) ** 2 / (2 * y.size)
    return (
        NUM.squeeze(),
        DEN.squeeze(),
        NUMH.squeeze(),
        DENH.squeeze(),
        Vn,
        y_id,
    )


def GEN_MISO_id(
    y: np.ndarray,
    u: np.ndarray,
    id_method: OptMethods,
    na: int,
    nb: np.ndarray,
    nc: int,
    nd: int,
    nf: int,
    theta: np.ndarray,
    max_iter: int,
    stab_marg: float,
    stab_cons: bool,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.floating, np.ndarray
]:
    # Rescale inputs
    y_std, y = rescale(y)
    U_std = np.zeros(u.shape[0])
    for j in range(u.shape[0]):
        U_std[j], u[j] = rescale(u[j])

    check_valid_orders(u.shape[0], nb, theta)

    return GEN_id(
        y,
        u,
        id_method,
        na,
        nb,
        nc,
        nd,
        nf,
        theta,
        max_iter,
        stab_marg,
        stab_cons,
        False,
        y_std,
        U_std,
    )
