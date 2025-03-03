"""
Created on 2017

@author: Giuseppe Armenise

@updates: Riccardo Bacci di Capaci, Marco Vaccari, Federico Pelagagge
"""

from collections.abc import Mapping
from typing import cast, get_args
from warnings import warn

import numpy as np

from .model import IO_MIMO_Model, IO_SISO_Model, SS_Model
from .typing import (
    AvailableMethods,
    AvailableModes,
    CenteringMethods,
    Flags,
    ICMethods,
    IOMethods,
    OLSimMethods,
    PARSIMMethods,
    SSMethods,
)

ID_MODES: dict[AvailableModes, Flags] = {
    "LLS": "arx",
    "RLLS": "rls",
    "OPT": "opt",
    "ILLS": "armax",
}

METHOD_ORDERS: dict[AvailableMethods, list[str]] = {
    "FIR": ["na", "nb", "theta"],
    "ARX": ["na", "nb", "theta"],
    "ARMAX": ["na", "nb", "nc", "theta"],
    "OE": ["nb", "nf", "theta"],
    "ARMA": ["na", "nc", "theta"],
    "ARARX": ["na", "nb", "nd", "theta"],
    "ARARMAX": ["na", "nb", "nc", "nd", "theta"],
    "BJ": ["nb", "nc", "nd", "nf", "theta"],
    "GEN": ["na", "nb", "nc", "nd", "nf", "theta"],
    "EARMAX": ["na", "nb", "nc", "theta"],
    "EOE": ["nb", "nf", "theta"],
    "CVA": ["n"],
    "MOESP": ["n"],
    "N4SID": ["n"],
    "PARSIM_K": ["n"],
    "PARSIM_P": ["n"],
    "PARSIM_S": ["n"],
}


def _check_fix_orders(
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
    >>> _check_fix_orders(orders, orders_defaults)
    {'na': array([2]), 'nb': array([[1, 2], [3, 4]]), 'nc': array([3, 4])}

    >>> orders = {'na': 2, 'nb': [1, 2, 3], 'nc': np.array([3, 4])}
    >>> _check_fix_orders(orders, orders_defaults)
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


def _areinstances(*args, class_or_tuple):
    """
    Check if all arguments are instances of a given class or tuple of classes.
    Args:
        *args: Variable length argument list of objects to check.
        class_or_tuple (type or tuple of types): The class or tuple of classes to check against.
    Returns:
        bool: True if all arguments are instances of the given class or tuple of classes, False otherwise.
    Examples:
        >>> _areinstances(1, 2, 3, class_or_tuple=int)
        True
        >>> _areinstances(1, 'a', 3, class_or_tuple=int)
        False
        >>> _areinstances(1, 'a', 3, class_or_tuple=(int, str))
        True
    """

    return all(map(lambda x: isinstance(x, class_or_tuple), args))


def _consolidate_orders(orders_defaults, orders_dict):
    orders_: dict = orders_dict
    orders_up = dict(orders_defaults)
    orders_up.update(orders_)
    orders_up = _check_fix_orders(orders_up, orders_defaults)
    orders = tuple(orders_up.values())
    return orders


def _validate_orders(
    *orders: int | list | np.ndarray | tuple, IC: ICMethods | None = None
):
    """
    Validates the orders types.

    Args:
        orders (list): A list of orders which can be of type int, list, or tuple.
        IC (ICMethods | None): An instance of ICMethods or None.

    Raises:
        ValueError: If orders are tuples and IC is not one of the ICMethods.
        ValueError: If orders are not all int, list, or tuple.

    Examples:
        >>> _validate_orders(*[1, 2, 3])
        >>> _validate_orders(*[(1, 2), (3, 4)], IC="AIC")
        >>> _validate_orders(*[1, [2, 3], (4, 5)])
        Traceback (most recent call last):
        ...
        ValueError: All orders must be either int, list, or tuple.Got [<class 'int'>, <class 'list'>, <class 'tuple'>] instead.
        >>> _validate_orders(*[(1, 2), (3, 4)])
        Traceback (most recent call last):
        ...
        ValueError: IC must be one of ('AIC', 'AICc', 'BIC', ...) if orders are tuples.
    """
    if _areinstances(*orders, class_or_tuple=tuple):
        if IC is None or IC not in get_args(ICMethods):
            raise ValueError(
                f"IC must be one of {get_args(ICMethods)} if orders are tuples. Got {IC} with {orders} instead."
            )
    elif not (
        _areinstances(*orders, class_or_tuple=int)
        or _areinstances(*orders, class_or_tuple=list)
    ):
        raise ValueError(
            "All orders must be either int, list, or tuple."
            f"Got {[type(order) for order in orders]} instead."
        )


def _verify_n_orders(
    id_method: AvailableMethods,
    *orders: int | list | np.ndarray | tuple,
    ydim: int,
    IC: ICMethods | None,
):
    """
    Verify the number and values of orders for a given identification method.

    Args:
        id_method (AvailableMethods): The identification method to be used.
        *orders (int | list | np.ndarray | tuple): Variable length argument list of orders.
        ydim (int): The dimension of the output.
        IC (ICMethods | None): The information criterion method, if any.

    Raises:
        ValueError: If the number of orders does not match the required number of orders for the method.
        ValueError: If the order 'na' for FIR is not valid.

    Examples:
        No exception raised
        >>> _verify_n_orders("FIR", *[[0,0], [1, 2], [2,3]], ydim=2, IC=None)

        >>> _verify_n_orders("FIR", *[1, 0], ydim=2, IC=None)
        Traceback (most recent call last):
            ...
        ValueError: Order 'na' for FIR must be [0, 0]. Got [1, 0] instead.

        >>> _verify_n_orders("ARX", 1, 2, ydim=2, IC=None)
        Traceback (most recent call last):
            ...
        ValueError: Number of orders (2) does not match the number of required orders (3). Required are [na, nb, theta] got [1, 2]
    """
    method_orders = METHOD_ORDERS[id_method]
    # TODO: allow user to not define `na` for FIR.
    if id_method == "FIR":
        if orders[0] != [0] * ydim and orders[0] != (0, 0):
            raise ValueError(
                f"Order 'na' for FIR must be {[0] * ydim if IC is None or IC not in get_args(ICMethods) else (0, 0)}. Got {orders[0]} instead."
            )

    if len(orders) != len(method_orders):
        raise ValueError(
            f"Number of orders ({len(orders)}) does not match the number of required orders ({len(method_orders)})."
            f"Required are {method_orders} got [{', '.join(map(str, orders))}]"
        )


def _recentering_transform(y, y_rif):
    ylength = y.shape[1]
    for i in range(ylength):
        y[:, i] = y[:, i] + y_rif
    return y


def _recentering_fit_transform(y, u, centering: CenteringMethods = None):
    ydim, ylength = y.shape
    udim, ulength = u.shape
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

    return y, u, y_rif


def system_identification(
    y: np.ndarray,
    u: np.ndarray,
    id_method: AvailableMethods,
    *orders: int | list[int] | list[list[int]] | np.ndarray | tuple[int, int],
    ts: float = 1.0,
    centering: CenteringMethods | None = None,
    IC: ICMethods | None = None,
    id_mode: AvailableModes = "OPT",  # TODO: figure out whether to remove default
    max_iter: int = 200,
    stab_marg: float = 1.0,
    stab_cons: bool = False,
    SS_f: int = 20,
    SS_p: int = 20,
    SS_threshold: float = 0.1,
    SS_D_required: bool = False,
    SS_A_stability: bool = False,
    SS_PK_B_reval: bool = False,
):
    # Verify y and u
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

    # Defining default values for orders
    orders_defaults: Mapping[str, np.ndarray] = {
        "na": np.zeros((ydim,), dtype=int),
        "nb": np.zeros((ydim, udim), dtype=int),
        "nc": np.zeros((ydim,), dtype=int),
        "nd": np.zeros((ydim,), dtype=int),
        "nf": np.zeros((ydim,), dtype=int),
        "theta": np.zeros((ydim, udim), dtype=int),
    }

    _validate_orders(*orders, IC=IC)

    _verify_n_orders(id_method, *orders, ydim=ydim, IC=IC)

    orders_dict = dict(zip(METHOD_ORDERS[id_method], orders, strict=True))

    # Data centering
    y, u, y_rif = _recentering_fit_transform(y, u, centering)

    ##### Check Information Criterion #####

    # MODE 1) fixed orders
    if not _areinstances(*orders, class_or_tuple=tuple):
        if IC is not None:
            warn("Ignoring argument 'IC' as fixed orders are provided.")

        # IO Models
        if id_method in get_args(IOMethods):
            id_method = cast(IOMethods, id_method)
            if id_mode in get_args(AvailableModes):
                flag = ID_MODES[id_mode]
            else:
                raise RuntimeError(
                    f"Method {id_mode} not available for {id_method}. Available: {get_args(AvailableModes)}"
                )

            orders = _consolidate_orders(orders_defaults, orders_dict)

            model = IO_MIMO_Model._identify(
                y,
                u,
                flag,
                id_method,
                *orders,
                ts=ts,
                max_iter=max_iter,
                stab_marg=stab_marg,
                stab_cons=stab_cons,
            )

            model.Yid = _recentering_transform(model.Yid, y_rif)

        # SS MODELS
        elif id_method in get_args(SSMethods):
            if len(orders) == 1 and isinstance(orders[0], int):
                order = orders[0]
            else:
                raise RuntimeError()
            if id_method in get_args(OLSimMethods):
                id_method = cast(OLSimMethods, id_method)
                from . import OLSims_methods

                A, B, C, D, Vn, Q, R, S, K = OLSims_methods.OLSims(
                    y,
                    u,
                    id_method,
                    order,
                    threshold=SS_threshold,
                    f=SS_f,
                    D_required=SS_D_required,
                    A_stability=SS_A_stability,
                )
                x0 = None
            else:
                id_method = cast(PARSIMMethods, id_method)
                from . import Parsim_methods

                A_K, C, B_K, D, K, A, B, x0, Vn = Parsim_methods.parsim(
                    y,
                    u,
                    id_method,
                    order,
                    SS_threshold,
                    SS_f,
                    SS_p,
                    SS_D_required,
                    SS_PK_B_reval,
                )
                Q, R, S = None, None, None
            model = SS_Model(A, B, C, D, K, ts, Vn, x0, Q, R, S)

        # NO method selected
        else:
            raise RuntimeError(
                f"Wrong identification method selected. Got {id_method}"
                f"expected one of {get_args(AvailableMethods)}"
            )

    # =========================================================================
    # MODE 2) order range
    # if an IC is selected
    else:
        IC = cast(ICMethods, IC)
        if ydim != 1 or udim != 1:
            raise RuntimeError(
                "Information criteria are implemented ONLY in SISO case "
                "for INPUT-OUTPUT model sets. Use subspace methods instead"
                " for MIMO cases"
            )

        if id_method in get_args(IOMethods):
            id_method = cast(IOMethods, id_method)
            if id_mode in get_args(AvailableModes):
                flag = ID_MODES[id_mode]
            else:
                raise RuntimeError(
                    f"Method {id_mode} not available for {id_method}. Available: {get_args(AvailableModes)}"
                )

            if not _areinstances(*orders, class_or_tuple=tuple):
                raise RuntimeError(
                    f"Orders ranges must be tuples. Got {[type(order) for order in orders]} instead."
                )
            else:
                orders = cast(tuple[tuple[int, int]], orders)
            model = IO_SISO_Model._from_order(
                y[0],
                u[0],
                *orders,
                flag=flag,
                id_method=id_method,
                ts=ts,
                ic_method=IC,
                max_iter=max_iter,
                stab_marg=stab_marg,
                stab_cons=stab_cons,
            )
            model.Yid = _recentering_transform(model.Yid, y_rif)

        # SS-MODELS
        ss_models: dict[
            SSMethods,
            str,
        ] = {
            "N4SID": "OLSims_methods",
            "MOESP": "OLSims_methods",
            "CVA": "OLSims_methods",
            "PARSIM_K": "Parsim_methods",
            "PARSIM_S": "Parsim_methods",
            "PARSIM_P": "Parsim_methods",
        }
        if id_method in ss_models:
            if len(orders) == 1 and isinstance(orders[0], tuple):
                order = orders[0]
            else:
                raise RuntimeError()
            if id_method in get_args(OLSimMethods):
                id_method = cast(OLSimMethods, id_method)
                from . import OLSims_methods

                A, B, C, D, Vn, Q, R, S, K = OLSims_methods.select_order_SIM(
                    y,
                    u,
                    id_method,
                    order,
                    IC,
                    SS_f,
                    SS_D_required,
                    SS_A_stability,
                )
                A_K, B_K, x0 = None, None, None
            else:
                id_method = cast(PARSIMMethods, id_method)
                from . import Parsim_methods

                A_K, C, B_K, D, K, A, B, x0, Vn = Parsim_methods.parsim(
                    y,
                    u,
                    id_method,
                    order,
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
