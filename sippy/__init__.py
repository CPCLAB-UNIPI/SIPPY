"""
Created on 2017

@author: Giuseppe Armenise

@updates: Riccardo Bacci di Capaci, Marco Vaccari, Federico Pelagagge
"""

from typing import Literal
from warnings import warn

import numpy as np

from .functionset import data_recentering
from .model import IO_MIMO_Model, IO_SISO_Model, SS_Model


def areinstances(objs, class_or_tuple) -> bool:
    return all(isinstance(obj, class_or_tuple) for obj in objs)


def check_orders(method, orders, desired_order):
    if method == "FIR" and len(orders) != desired_order:
        raise RuntimeError(
            f"{method} identification takes {desired_order} arguments in {method}_orders"
        )


def check_fix_orders(
    method: str,
    orders: tuple[int | list, int | list],
    desired_order: int,
    desired_shapes: list[tuple[int, int]],
):
    check_orders(method, orders, desired_order)

    if areinstances(orders, int):
        orders_ = [
            (order * np.ones(shape, dtype=int))
            for order, shape in zip(orders, desired_shapes)
        ]
    elif areinstances(orders, list):
        orders_ = orders
    else:
        raise RuntimeError(
            f"{method}_orders must be a list containing {desired_order} lists or {desired_order} integers"
        )
    return orders_


def system_identification(
    y,
    u,
    id_method: Literal[
        "FIR",
        "ARX",
        "ARMA",
        "ARMAX",
        "OE",
        "ARARX",
        "ARARMAX",
        "BJ",
        "GEN",
        "EARMAX",
        "EOE",
        "N4SID",
        "MOESP",
        "CVA",
        "PARSIM-K",
        "PARSIM-S",
        "PARSIM-P",
    ],
    centering: str | None = None,
    IC: str | None = None,
    tsample: int = 1,
    FIR_orders: tuple[int | list, int | list] = (1, 0),
    ARX_orders: list[int] = [1, 1, 0],
    ARMA_orders: list[int] = [1, 1, 0],
    ARMAX_orders: list[int] = [1, 1, 1, 0],
    ARARX_orders: list[int] = [1, 1, 1, 0],
    ARARMAX_orders: list[int] = [1, 1, 1, 1, 0],
    OE_orders: list[int] = [1, 1, 0],
    BJ_orders: list[int] = [1, 1, 1, 1, 0],
    GEN_orders: list[int] = [1, 1, 1, 1, 1, 0],
    na_ord: tuple[int, int] = (0, 5),
    nb_ord: tuple[int, int] = (1, 5),
    nc_ord: tuple[int, int] = (0, 5),
    nd_ord: tuple[int, int] = (0, 5),
    nf_ord: tuple[int, int] = (0, 5),
    delays: tuple[int, int] = (0, 5),
    FIR_mod: Literal["LLS", "RLLS"] = "LLS",
    ARX_mod: Literal["LLS", "RLLS"] = "LLS",
    ARMAX_mod: Literal["ILLS", "RLLS", "OPT"] = "ILLS",
    OE_mod: Literal["EOE", "RLLS", "OPT"] = "OPT",
    id_mode: Literal["LLS", "ILLS", "EOE", "RLLS", "OPT"] = "OPT",
    max_iterations: int = 200,
    stab_marg: float = 1.0,
    stab_cons: bool = False,
    SS_f: int = 20,
    SS_p: int = 20,
    SS_threshold: float = 0.1,
    SS_max_order: float = np.nan,
    SS_fixed_order: float = np.nan,
    SS_orders: list[int] = [1, 10],
    SS_D_required: bool = False,
    SS_A_stability: bool = False,
    SS_PK_B_reval: bool = False,
):
    y = 1.0 * np.atleast_2d(y)
    u = 1.0 * np.atleast_2d(u)
    [n1, n2] = y.shape
    ydim = min(n1, n2)
    ylength = max(n1, n2)
    if ylength == n1:
        y = y.T
    [n1, n2] = u.shape
    ulength = max(n1, n2)
    udim = min(n1, n2)
    if ulength == n1:
        u = u.T

    # Checking data consinstency
    if ulength != ylength:
        warn(
            "y and u lengths are not the same. The minor value between the two lengths has been chosen. The perfomed indentification may be not correct, be sure to check your input and output data alignement"
        )
        # Recasting data cutting out the over numbered data
        minlength = min(ulength, ylength)
        y = y[:, :minlength]
        u = u[:, :minlength]

    # Data centering
    if centering == "InitVal":
        y_rif = 1.0 * y[:, 0]
        u_init = 1.0 * u[:, 0]
        for i in range(ylength):
            y[:, i] = y[:, i] - y_rif
            u[:, i] = u[:, i] - u_init
    elif centering == "MeanVal":
        y_rif = np.zeros(ydim)
        u_mean = np.zeros(udim)
        for i in range(ydim):
            y_rif[i] = np.mean(y[i, :])
        for i in range(udim):
            u_mean[i] = np.mean(u[i, :])
        for i in range(ylength):
            y[:, i] = y[:, i] - y_rif
            u[:, i] = u[:, i] - u_mean
    else:
        if centering is not None:
            warn(
                "'centering' argument is not valid, its value has been reset to 'None'"
            )
        y_rif = 0.0 * y[:, 0]

    # Defining default values for orders
    na = np.zeros((ydim,), dtype=int)
    nb = np.zeros((ydim, udim), dtype=int)
    nc = np.zeros((ydim,), dtype=int)
    nd = np.zeros((ydim,), dtype=int)
    nf = np.zeros((ydim,), dtype=int)
    theta = np.zeros((ydim, udim), dtype=int)

    model = None

    ##### Check Information Criterion #####

    # MODE 1) fixed orders
    if not IC == "AIC" or IC == "AICc" or IC == "BIC":
        if IC is not None:
            warn("no correct information criterion selected, using 'None'")

        # MODEL choice

        # INPUT-OUTPUT MODELS
        io_models = {
            "FIR": (FIR_orders, 2, [nb.shape, theta.shape]),
            "ARX": (ARX_orders, 3, [na.shape, nb.shape, theta.shape]),
            "ARMAX": (
                ARMAX_orders,
                4,
                [na.shape, nb.shape, nc.shape, theta.shape],
            ),
            "OE": (OE_orders, 3, [nb.shape, nf.shape, theta.shape]),
            "ARMA": (ARMA_orders, 3, [na.shape, nc.shape, theta.shape]),
            "ARARX": (
                ARARX_orders,
                4,
                [na.shape, nb.shape, nd.shape, theta.shape],
            ),
            "ARARMAX": (
                ARARMAX_orders,
                5,
                [na.shape, nb.shape, nc.shape, nd.shape, theta.shape],
            ),
            "BJ": (
                BJ_orders,
                5,
                [nb.shape, nc.shape, nd.shape, nf.shape, theta.shape],
            ),
            "GEN": (
                GEN_orders,
                6,
                [
                    na.shape,
                    nb.shape,
                    nc.shape,
                    nd.shape,
                    nf.shape,
                    theta.shape,
                ],
            ),
            "EARMAX": (
                ARMAX_orders,
                4,
                [na.shape, nb.shape, nc.shape, theta.shape],
            ),
            "EOE": (OE_orders, 3, [nb.shape, nf.shape, theta.shape]),
        }

        if id_method in io_models:
            orders, n_args, shapes = io_models[id_method]

            orders = check_fix_orders(id_method, orders, n_args, shapes)

            if id_method == "FIR":
                nb, theta = orders
            elif id_method == "ARX":
                na, nb, theta = orders
            elif id_method == "ARMAX":
                na, nb, nc, theta = orders
            elif id_method == "OE":
                nb, nf, theta = orders
            elif id_method == "ARMA":
                na, nc, theta = orders
            elif id_method == "ARARX":
                na, nb, nd, theta = orders
            elif id_method == "ARARMAX":
                na, nb, nc, nd, theta = orders
            elif id_method == "BJ":
                nb, nc, nd, nf, theta = orders
            elif id_method == "GEN":
                na, nb, nc, nd, nf, theta = orders
            elif id_method == "EARMAX":
                na, nb, nc, theta = orders
            elif id_method == "EOE":
                nb, nf, theta = orders

            # FIR or ARX
            if id_method in ["FIR", "ARX"]:
                # Standard Linear Least Square
                if ARX_mod == "LLS" or FIR_mod == "LLS":
                    from . import arxMIMO

                    DENOMINATOR, NUMERATOR, G, H, Vn_tot, Yid = arxMIMO.ARX_MIMO_id(
                        y, u, na, nb, theta, tsample
                    )
                    nc = None
                    nd = None
                    nf = None
                    DENOMINATOR_H = None
                    NUMERATOR_H = None

                # Recursive Least Square
                elif ARX_mod == "RLLS" or FIR_mod == "RLLS":
                    from . import io_rlsMIMO

                    (
                        DENOMINATOR,
                        NUMERATOR,
                        DENOMINATOR_H,
                        NUMERATOR_H,
                        G,
                        H,
                        Vn_tot,
                        Yid,
                    ) = io_rlsMIMO.GEN_MIMO_id(
                        id_method,
                        y,
                        u,
                        na,
                        nb,
                        nc,
                        nd,
                        nf,
                        theta,
                        tsample,
                        max_iterations,
                    )

            # ARMAX
            elif id_method == "ARMAX":
                # check identification method

                # Iterative Linear Least Squares
                if ARMAX_mod == "ILLS":
                    from . import armaxMIMO

                    (
                        DENOMINATOR,
                        NUMERATOR,
                        DENOMINATOR_H,
                        NUMERATOR_H,
                        G,
                        H,
                        Vn_tot,
                        Yid,
                    ) = armaxMIMO.ARMAX_MIMO_id(
                        y, u, na, nb, nc, theta, tsample, max_iterations
                    )
                    nd = None
                    nf = None

                # Recursive Least Square
                elif ARMAX_mod == "RLLS":
                    from . import io_rlsMIMO

                    (
                        DENOMINATOR,
                        NUMERATOR,
                        DENOMINATOR_H,
                        NUMERATOR_H,
                        G,
                        H,
                        Vn_tot,
                        Yid,
                    ) = io_rlsMIMO.GEN_MIMO_id(
                        id_method,
                        y,
                        u,
                        na,
                        nb,
                        nc,
                        nd,
                        nf,
                        theta,
                        tsample,
                        max_iterations,
                    )

                # OPTMIZATION-BASED
                elif ARMAX_mod == "OPT":
                    from . import io_optMIMO

                    (
                        DENOMINATOR,
                        NUMERATOR,
                        DENOMINATOR_H,
                        NUMERATOR_H,
                        G,
                        H,
                        Vn_tot,
                        Yid,
                    ) = io_optMIMO.GEN_MIMO_id(
                        id_method,
                        y,
                        u,
                        na,
                        nb,
                        nc,
                        nd,
                        nf,
                        theta,
                        tsample,
                        max_iterations,
                        stab_marg,
                        stab_cons,
                    )

            # (OE) Output-Error
            elif id_method == "OE":
                # check identification method

                # Iterative Linear Least Squares
                if OE_mod == "RLLS":
                    from . import io_rlsMIMO

                    (
                        DENOMINATOR,
                        NUMERATOR,
                        DENOMINATOR_H,
                        NUMERATOR_H,
                        G,
                        H,
                        Vn_tot,
                        Yid,
                    ) = io_rlsMIMO.GEN_MIMO_id(
                        id_method,
                        y,
                        u,
                        na,
                        nb,
                        nc,
                        nd,
                        nf,
                        theta,
                        tsample,
                        max_iterations,
                    )

                # OPTMIZATION-BASED
                elif OE_mod == "OPT":
                    from . import io_optMIMO

                    (
                        DENOMINATOR,
                        NUMERATOR,
                        DENOMINATOR_H,
                        NUMERATOR_H,
                        G,
                        H,
                        Vn_tot,
                        Yid,
                    ) = io_optMIMO.GEN_MIMO_id(
                        id_method,
                        y,
                        u,
                        na,
                        nb,
                        nc,
                        nd,
                        nf,
                        theta,
                        tsample,
                        max_iterations,
                        stab_marg,
                        stab_cons,
                    )

            elif (
                id_method == "ARMA"
                or id_method == "ARARX"
                or id_method == "ARARMAX"
                or id_method == "GEN"
                or id_method == "BJ"
                or id_method == "EOE"
            ):
                from . import io_optMIMO

                (
                    DENOMINATOR,
                    NUMERATOR,
                    DENOMINATOR_H,
                    NUMERATOR_H,
                    G,
                    H,
                    Vn_tot,
                    Yid,
                ) = io_optMIMO.GEN_MIMO_id(
                    id_method,
                    y,
                    u,
                    na,
                    nb,
                    nc,
                    nd,
                    nf,
                    theta,
                    tsample,
                    max_iterations,
                    stab_marg,
                    stab_cons,
                )

            Yid = data_recentering(Yid, y_rif, ylength)
            model = IO_MIMO_Model(
                na,
                nb,
                nc,
                nd,
                nf,
                theta,
                tsample,
                NUMERATOR,
                DENOMINATOR,
                NUMERATOR_H,
                DENOMINATOR_H,
                G,
                H,
                Vn_tot,
                Yid,
            )

        # SS MODELS
        ss_models = {
            "N4SID": "OLSims_methods",
            "MOESP": "OLSims_methods",
            "CVA": "OLSims_methods",
            "PARSIM-K": "Parsim_methods",
            "PARSIM-S": "Parsim_methods",
            "PARSIM-P": "Parsim_methods",
        }
        if id_method in ss_models:
            if id_method in ["N4SID", "MOESP", "CVA"]:
                from . import OLSims_methods

                A, B, C, D, Vn, Q, R, S, K = OLSims_methods.OLSims(
                    y,
                    u,
                    SS_f,
                    id_method,
                    SS_threshold,
                    SS_max_order,
                    SS_fixed_order,
                    SS_D_required,
                    SS_A_stability,
                )
                x0 = None
            else:
                from . import Parsim_methods

                A_K, C, B_K, D, K, A, B, x0, Vn = getattr(
                    Parsim_methods, id_method.replace("-", "_")
                )(
                    y,
                    u,
                    SS_f,
                    SS_p,
                    SS_threshold,
                    SS_max_order,
                    SS_fixed_order,
                    SS_D_required,
                    SS_PK_B_reval,
                )
                Q, R, S = None, None, None
            model = SS_Model(A, B, C, D, K, tsample, Vn, x0, Q, R, S)

        # NO method selected
        if model is None:
            raise RuntimeError("No identification method selected")

    # =========================================================================
    # MODE 2) order range
    # if an IC is selected
    else:
        if ydim != 1 or udim != 1:
            raise RuntimeError(
                "Information criteria are implemented ONLY in SISO case "
                "for INPUT-OUTPUT model sets. Use subspace methods instead"
                " for MIMO cases"
            )

        io_models = {
            "FIR": {
                "LLS": {"na_ord": (0, 0), "flag": "arx"},
                "RLLS": {
                    "na_ord": (0, 0),
                    "nc_ord": (0, 0),
                    "nd_ord": (0, 0),
                    "nf_ord": (0, 0),
                    "flag": "rls",
                },
            },
            "ARX": {
                "LLS": {"flag": "arx"},
                "RLLS": {
                    "nc_ord": (0, 0),
                    "nd_ord": (0, 0),
                    "nf_ord": (0, 0),
                    "flag": "rls",
                },
            },
            "ARMA": {"flag": "opt"},
            "ARMAX": {
                "ILLS": {"flag": "armax"},
                "RLLS": {"nd_ord": (0, 0), "nf_ord": (0, 0), "flag": "rls"},
                "OPT": {"nd_ord": (0, 0), "nf_ord": (0, 0), "flag": "opt"},
            },
            "ARARX": {"flag": "opt"},
            "ARARMAX": {"nf_ord": (0, 0), "flag": "opt"},
            "OE": {
                "RLLS": {
                    "na_ord": (0, 0),
                    "nc_ord": (0, 0),
                    "nd_ord": (0, 0),
                    "flag": "rls",
                },
                "OPT": {
                    "na_ord": (0, 0),
                    "nc_ord": (0, 0),
                    "nd_ord": (0, 0),
                    "flag": "opt",
                },
            },
            "BJ": {"na_ord": (0, 0), "flag": "opt"},
            "GEN": {"flag": "opt"},
            "EARMAX": {"flag": "opt"},
            "EOE": {"nc_ord": (0, 0), "flag": "opt"},
        }

        if id_method in io_models:
            method_config = io_models[id_method]
            if isinstance(method_config, dict):
                if id_mode in method_config:
                    config = method_config[id_mode]
                    na_ord = config.get("na_ord", na_ord)
                    nb_ord = config.get("nb_ord", nb_ord)
                    nc_ord = config.get("nc_ord", nc_ord)
                    nd_ord = config.get("nd_ord", nd_ord)
                    nf_ord = config.get("nf_ord", nf_ord)
                    flag = config.get("flag", "opt")
                else:
                    raise RuntimeError(
                        f"Method {id_mode} not available for {id_method}"
                    )
            else:
                raise RuntimeError(f"Method {id_method} not available")

            orders = na_ord, nb_ord, nc_ord, nd_ord, nf_ord, delays

            if flag != "armax":
                model = IO_SISO_Model._from_order(
                    "arx",
                    id_method,
                    y[0],
                    u[0],
                    tsample,
                    *orders,
                    IC,
                    max_iterations,
                    stab_marg,
                    stab_cons,
                )
                model.Yid = data_recentering(model.Yid, y_rif, ylength)

            # ARMAX
            else:
                # Iterative Linear Least Squares
                if ARMAX_mod == "ILLS":
                    from . import armax

                    model = armax.Armax(
                        na_ord,
                        nb_ord,
                        nc_ord,
                        delays,
                        tsample,
                        IC,
                        max_iterations,
                    )
                    #
                    model.find_best_estimate(y[0], u[0])
                    model.Yid = data_recentering(model.Yid, y_rif, ylength)

        # SS-MODELS
        ss_models = {
            "N4SID": "OLSims_methods",
            "MOESP": "OLSims_methods",
            "CVA": "OLSims_methods",
            "PARSIM-K": "Parsim_methods",
            "PARSIM-S": "Parsim_methods",
            "PARSIM-P": "Parsim_methods",
        }
        if id_method in ss_models:
            if id_method in ["N4SID", "MOESP", "CVA"]:
                from . import OLSims_methods

                A, B, C, D, Vn, Q, R, S, K = OLSims_methods.select_order_SIM(
                    y,
                    u,
                    SS_f,
                    id_method,
                    IC,
                    SS_orders,
                    SS_D_required,
                    SS_A_stability,
                )
                x0 = None
            else:
                from . import Parsim_methods

                A_K, C, B_K, D, K, A, B, x0, Vn = Parsim_methods.select_order(
                    id_method.split("-")[-1],
                    y,
                    u,
                    SS_f,
                    SS_p,
                    IC,
                    SS_orders,
                    SS_D_required,
                    SS_PK_B_reval,
                )
                Q, R, S = None, None, None
            model = SS_Model(A, B, C, D, K, tsample, Vn, x0, Q, R, S)

        # NO method selected
        if model is None:
            raise RuntimeError("No identification method selected")

    return model
