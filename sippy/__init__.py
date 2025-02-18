"""
Created on 2017

@author: Giuseppe Armenise

@updates: Riccardo Bacci di Capaci, Marco Vaccari, Federico Pelagagge
"""

from collections.abc import Mapping
from typing import Literal
from warnings import warn

import numpy as np

from .functionset import data_recentering
from .model import IO_MIMO_Model, IO_SISO_Model, SS_Model


def check_fix_orders(
    orders: Mapping[str, int | list | np.ndarray],
    orders_defaults: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """
    Ensure that the orders dictionary has the correct shape and type.

    Parameters:
    orders (Mapping[str, int | list | np.ndarray]): The orders to check.
    orders_defaults (Mapping[str, np.ndarray]): The default orders to use for shape reference.

    Returns:
    dict[str, np.ndarray]: The validated and fixed orders.

    Raises:
    RuntimeError: If the order is not an int, list, or np.ndarray, or if the list length does not match the default shape.

    Examples:
    >>> orders_defaults = {'na': np.zeros((1,)), 'nb': np.zeros((2,2)), 'nc': np.zeros((2,))}
    >>> orders = {'na': 2, 'nb': [[1, 2], [3, 4]], 'nc': np.array([3, 4])}
    >>> check_fix_orders(orders, orders_defaults)
    {'na': array([2]), 'nb': array([[1, 2], [3, 4]]), 'nc': array([3, 4])}

    >>> orders = {'na': 2, 'nb': [1, 2, 3], 'nc': np.array([3, 4])}
    >>> check_fix_orders(orders, orders_defaults)
    Traceback (most recent call last):
        ...
    RuntimeError: Order for nb must have 2 elements
    """
    orders_: dict[str, np.ndarray] = {}
    for name, order in orders.items():
        if isinstance(order, int):
            orders_[name] = order * np.ones(
                orders_defaults[name].shape, dtype=int
            )
        elif isinstance(order, list | np.ndarray):
            orders_[name] = np.array(order, dtype=int)
            if orders_[name].shape != orders_defaults[name].shape:
                raise RuntimeError(
                    f"Order for {name} must have {len(orders_defaults[name])} elements"
                )
        else:
            raise RuntimeError(
                f"Order for {name} must be a list of integers of shape {orders_defaults[name].shape}"
            )

    return orders_


def system_identification(
    y: np.ndarray,
    u: np.ndarray,
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
    ts: float = 1.0,
    FIR_orders: tuple[int | list, int | list] = (1, 0),
    ARX_orders: tuple[int | list, int | list, int | list] = (1, 1, 0),
    ARMA_orders: tuple[int | list, int | list, int | list] = (1, 1, 0),
    ARMAX_orders: tuple[int | list, int | list, int | list, int | list] = (
        1,
        1,
        1,
        0,
    ),
    ARARX_orders: tuple[int | list, int | list, int | list, int | list] = (
        1,
        1,
        1,
        0,
    ),
    ARARMAX_orders: tuple[
        int | list, int | list, int | list, int | list, int | list
    ] = (1, 1, 1, 1, 0),
    OE_orders: tuple[int | list, int | list, int | list] = (1, 1, 0),
    BJ_orders: tuple[
        int | list, int | list, int | list, int | list, int | list
    ] = (
        1,
        1,
        1,
        1,
        0,
    ),
    GEN_orders: tuple[
        int | list, int | list, int | list, int | list, int | list, int | list
    ] = (1, 1, 1, 1, 1, 0),
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
    max_iter: int = 200,
    stab_marg: float = 1.0,
    stab_cons: bool = False,
    SS_f: int = 20,
    SS_p: int = 20,
    SS_threshold: float = 0.1,
    SS_order: int = 0,
    SS_orders: tuple[int, int] = (1, 10),
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
    orders_defaults: Mapping[str, np.ndarray] = {
        "na": np.zeros((ydim,), dtype=int),
        "nb": np.zeros((ydim, udim), dtype=int),
        "nc": np.zeros((ydim,), dtype=int),
        "nd": np.zeros((ydim,), dtype=int),
        "nf": np.zeros((ydim,), dtype=int),
        "theta": np.zeros((ydim, udim), dtype=int),
    }

    model = None

    ##### Check Information Criterion #####

    # MODE 1) fixed orders
    if not IC == "AIC" or IC == "AICc" or IC == "BIC":
        if IC is not None:
            warn("no correct information criterion selected, using 'None'")

        # MODEL choice

        # INPUT-OUTPUT MODELS
        io_models = {
            "FIR": dict(zip(["na", "nb", "theta"], (0, *FIR_orders))),
            "ARX": dict(zip(["na", "nb", "theta"], ARX_orders)),
            "ARMAX": dict(zip(["na", "nb", "nc", "theta"], ARMAX_orders)),
            "OE": dict(zip(["nb", "nf", "theta"], OE_orders)),
            "ARMA": dict(zip(["na", "nc", "theta"], ARMA_orders)),
            "ARARX": dict(zip(["na", "nb", "nd", "theta"], ARARX_orders)),
            "ARARMAX": dict(
                zip(["na", "nb", "nc", "nd", "theta"], ARARMAX_orders)
            ),
            "BJ": dict(zip(["nb", "nc", "nd", "nf", "theta"], BJ_orders)),
            "GEN": dict(
                zip(["na", "nb", "nc", "nd", "nf", "theta"], GEN_orders)
            ),
            "EARMAX": dict(zip(["na", "nb", "nc", "theta"], ARMAX_orders)),
            "EOE": dict(zip(["nb", "nf", "theta"], OE_orders)),
        }

        if id_method in io_models:
            orders_: dict = io_models[id_method]
            orders = dict(orders_defaults)
            orders.update(orders_)
            orders = check_fix_orders(orders, orders_defaults)
            params = {**orders, "ts": ts, "max_iter": max_iter}

            # FIR or ARX
            if id_method in ["FIR", "ARX"]:
                # Standard Linear Least Square
                if ARX_mod == "LLS" or FIR_mod == "LLS":
                    from . import arxMIMO

                    denominator, numerator, G, H, Vn_tot, Yid = (
                        arxMIMO.ARX_MIMO_id(y, u, **params)
                    )
                    denominator_H = None
                    numerator_H = None

                # Recursive Least Square
                elif ARX_mod == "RLLS" or FIR_mod == "RLLS":
                    from . import io_rlsMIMO

                    (
                        denominator,
                        numerator,
                        denominator_H,
                        numerator_H,
                        G,
                        H,
                        Vn_tot,
                        Yid,
                    ) = io_rlsMIMO.GEN_MIMO_id(id_method, y, u, **params)

            # ARMAX
            elif id_method == "ARMAX":
                # check identification method

                # Iterative Linear Least Squares
                if ARMAX_mod == "ILLS":
                    from . import armaxMIMO

                    (
                        denominator,
                        numerator,
                        denominator_H,
                        numerator_H,
                        G,
                        H,
                        Vn_tot,
                        Yid,
                    ) = armaxMIMO.ARMAX_MIMO_id(y, u, **params)

                # Recursive Least Square
                elif ARMAX_mod == "RLLS":
                    from . import io_rlsMIMO

                    (
                        denominator,
                        numerator,
                        denominator_H,
                        numerator_H,
                        G,
                        H,
                        Vn_tot,
                        Yid,
                    ) = io_rlsMIMO.GEN_MIMO_id(id_method, y, u, **params)

                # OPTMIZATION-BASED
                elif ARMAX_mod == "OPT":
                    from . import io_optMIMO

                    (
                        denominator,
                        numerator,
                        denominator_H,
                        numerator_H,
                        G,
                        H,
                        Vn_tot,
                        Yid,
                    ) = io_optMIMO.GEN_MIMO_id(
                        id_method,
                        y,
                        u,
                        **params,
                        st_m=stab_marg,
                        st_c=stab_cons,
                    )

            # (OE) Output-Error
            elif id_method == "OE":
                # check identification method

                # Iterative Linear Least Squares
                if OE_mod == "RLLS":
                    from . import io_rlsMIMO

                    (
                        denominator,
                        numerator,
                        denominator_H,
                        numerator_H,
                        G,
                        H,
                        Vn_tot,
                        Yid,
                    ) = io_rlsMIMO.GEN_MIMO_id(id_method, y, u, **params)

                # OPTMIZATION-BASED
                elif OE_mod == "OPT":
                    from . import io_optMIMO

                    (
                        denominator,
                        numerator,
                        denominator_H,
                        numerator_H,
                        G,
                        H,
                        Vn_tot,
                        Yid,
                    ) = io_optMIMO.GEN_MIMO_id(
                        id_method,
                        y,
                        u,
                        **params,
                        st_m=stab_marg,
                        st_c=stab_cons,
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
                    denominator,
                    numerator,
                    denominator_H,
                    numerator_H,
                    G,
                    H,
                    Vn_tot,
                    Yid,
                ) = io_optMIMO.GEN_MIMO_id(
                    id_method,
                    y,
                    u,
                    **params,
                    st_m=stab_marg,
                    st_c=stab_cons,
                )

            Yid = data_recentering(Yid, y_rif, ylength)
            model = IO_MIMO_Model(
                **params,
                numerator=numerator,
                denominator=denominator,
                numerator_H=numerator_H,
                denominator_H=denominator_H,
                G=G,
                H=H,
                Vn=Vn_tot,
                Yid=Yid,
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
                    id_method,
                    y,
                    u,
                    SS_order,
                    SS_threshold,
                    SS_f,
                    SS_D_required,
                    SS_A_stability,
                )
                x0 = None
            else:
                from . import Parsim_methods

                A_K, C, B_K, D, K, A, B, x0, Vn = Parsim_methods.parsim(
                    id_method,
                    y,
                    u,
                    SS_order,
                    SS_threshold,
                    SS_f,
                    SS_p,
                    SS_D_required,
                    SS_PK_B_reval,
                )
                Q, R, S = None, None, None
            model = SS_Model(A, B, C, D, K, ts, Vn, x0, Q, R, S)

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
                        f"Method {id_mode} not available for {id_method}. Available: {method_config.keys()}"
                    )
            else:
                raise RuntimeError(
                    f"Method {id_method} not available. Available: {io_models.keys()}"
                )

            orders = na_ord, nb_ord, nc_ord, nd_ord, nf_ord, delays

            if flag != "armax":
                model = IO_SISO_Model._from_order(
                    "arx",
                    id_method,
                    y[0],
                    u[0],
                    ts,
                    *orders,
                    IC,
                    max_iter,
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
                        ts,
                        IC,
                        max_iter,
                    )
                    #
                    model.find_best_estimate(y[0], u[0])
                    model.Yid = data_recentering(model.Yid, y_rif, ylength)

        # SS-MODELS
        ss_models: dict[
            Literal[
                "N4SID",
                "MOESP",
                "CVA",
                "PARSIM-K",
                "PARSIM-S",
                "PARSIM-P",
            ],
            str,
        ] = {
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
                A_K, B_K, x0 = None, None, None
            else:
                from . import Parsim_methods

                A_K, C, B_K, D, K, A, B, x0, Vn = Parsim_methods.parsim(
                    id_method,
                    y,
                    u,
                    SS_orders,
                    f=SS_f,
                    p=SS_p,
                    D_required=SS_D_required,
                    B_recalc=SS_PK_B_reval,
                    ic_method=IC,
                )
                Q, R, S = None, None, None
            model = SS_Model(A, B, C, D, K, ts, Vn, x0, Q, R, S, A_K, B_K)

        # NO method selected
        if model is None:
            raise RuntimeError("No identification method selected")

    return model
