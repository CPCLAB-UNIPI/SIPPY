"""
Created on 2017

@author: Giuseppe Armenise

@updates: Riccardo Bacci di Capaci, Marco Vaccari, Federico Pelagagge
"""

from collections.abc import Mapping
from typing import cast, get_args, overload
from warnings import warn

import numpy as np

from ._typing import (
    AvailableMethods,
    AvailableModes,
    CenteringMethods,
    Flags,
    ICMethods,
    IOMethods,
    SSMethods,
)
from .model import IO_MIMO_Model, IO_SISO_Model, SS_Model

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
    "CVA": ["na"],
    "MOESP": ["na"],
    "N4SID": ["na"],
    "PARSIM_K": ["na"],
    "PARSIM_P": ["na"],
    "PARSIM_S": ["na"],
}


@overload
def _as_orders_defaults(
    orders_dict: Mapping[str, int | list | np.ndarray],
    orders_defaults: Mapping[str, np.ndarray],
) -> Mapping[str, np.ndarray]: ...
@overload
def _as_orders_defaults(
    orders_dict: Mapping[str, tuple[int, int]],
    orders_defaults: Mapping[str, tuple[int, int]],
) -> Mapping[str, tuple[int, int]]: ...


def _as_orders_defaults(
    orders_dict: Mapping[str, int | list | np.ndarray | tuple[int, int]],
    orders_defaults: Mapping[str, np.ndarray | tuple[int, int]],
) -> Mapping[str, np.ndarray | tuple[int, int]]:
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
    >>> _as_orders_defaults(orders, orders_defaults)
    {'na': array([2]), 'nb': array([[1, 2], [3, 4]]), 'nc': array([3, 4])}

    >>> orders = {'na': 2, 'nb': [1, 2, 3], 'nc': np.array([3, 4])}
    >>> _as_orders_defaults(orders, orders_defaults)
    Traceback (most recent call last):
        ...
    RuntimeError: Order for nb must have 2 elements

    >>> orders_defaults = {'na': (0, 0), 'nb': (0, 0), 'nc': (0, 0)}
    >>> orders = {'na': (0, 0), 'nb': (0, 0), 'nc': (0, 0)}
    >>> _as_orders_defaults(orders, orders_defaults)
    {'na': (0, 0), 'nb': (0, 0), 'nc': (0, 0)}
    >>> orders = {'na': 0, 'nb': (0, 0), 'nc': (0, 0)}
    >>> _as_orders_defaults(orders, orders_defaults)
    Traceback (most recent call last):
        ...
    RuntimeError: Order for na must be convertible to (0, 0). Got 0 instead.
    """
    orders_: dict[str, np.ndarray | tuple[int, int]] = {}
    for name, order in orders_dict.items():
        order_defaults = orders_defaults[name]
        if isinstance(order_defaults, np.ndarray):
            shape = order_defaults.shape
            if isinstance(order, int):
                order_ = order * np.ones(shape, dtype=int)
            elif isinstance(order, list | np.ndarray):
                order_ = np.array(order, dtype=int)
                if order_.shape != shape:
                    raise RuntimeError(
                        f"Order for {name} must be of shape {shape}. Got {order_.shape} instead."
                    )
            orders_[name] = order_
        elif isinstance(order, tuple):
            if len(order) != 2:
                raise RuntimeError(
                    f"Order for {name} must have 2 elements. Got {len(order)} instead."
                )
            orders_[name] = order

        else:
            raise RuntimeError(
                f"Order for {name} must be convertible to {order_defaults}. Got {order} instead."
            )

    return orders_


def _areinstances(args: tuple, class_or_tuple):
    """
    Check if all arguments are instances of a given class or tuple of classes.
    Args:
        *args: Variable length argument list of objects to check.
        class_or_tuple (type or tuple of types): The class or tuple of classes to check against.
    Returns:
        bool: True if all arguments are instances of the given class or tuple of classes, False otherwise.
    Examples:
        >>> _areinstances((1, 2, 3), int)
        True
        >>> _areinstances((1, 'a', 3), int)
        False
        >>> _areinstances((1, 'a', 3), (int, str))
        True
    """

    return all(isinstance(x, class_or_tuple) for x in args)


def _verify_orders_types(
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
        >>> _verify_orders_types(*[1, 2, 3])
        >>> _verify_orders_types(*[(1, 2), (3, 4)], IC="AIC")
        >>> _verify_orders_types(*[1, [2, 3], (4, 5)])
        Traceback (most recent call last):
        ...
        ValueError: All orders must be either int, list, or tuple.Got [<class 'int'>, <class 'list'>, <class 'tuple'>] instead.
        >>> _verify_orders_types(*[(1, 2), (3, 4)])
        Traceback (most recent call last):
        ...
        ValueError: IC must be one of ('AIC', 'AICc', ...) if orders are tuples.
    """
    if _areinstances(orders, tuple):
        if IC is None or IC not in get_args(ICMethods):
            raise ValueError(
                f"IC must be one of {get_args(ICMethods)} if orders are tuples. Got {IC} with {orders} instead."
            )
    elif not (_areinstances(orders, int) or _areinstances(orders, list)):
        raise ValueError(
            "All orders must be either int, list, or tuple."
            f"Got {[type(order) for order in orders]} instead."
        )


def _verify_orders_len(
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
        >>> _verify_orders_len("FIR", *[[0,0], [1, 2], [2,3]], ydim=2, IC=None)

        >>> _verify_orders_len("FIR", *[1, 0], ydim=2, IC=None)
        Traceback (most recent call last):
            ...
        ValueError: Order 'na' for FIR must be [0, 0]. Got [1, 0] instead.

        >>> _verify_orders_len("ARX", 1, 2, ydim=2, IC=None)
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


@overload
def _update_orders(
    orders: tuple[int | list[int] | list[list[int]] | np.ndarray, ...],
    orders_defaults: Mapping[str, np.ndarray],
    id_method: AvailableMethods,
) -> tuple[np.ndarray, ...]: ...
@overload
def _update_orders(
    orders: tuple[tuple[int, int], ...],
    orders_defaults: Mapping[str, tuple[int, int]],
    id_method: AvailableMethods,
) -> tuple[tuple[int, int], ...]: ...


def _update_orders(
    orders: tuple[
        int | list[int] | list[list[int]] | np.ndarray | tuple[int, int], ...
    ],
    orders_defaults: Mapping[str, np.ndarray | tuple[int, int]],
    id_method: AvailableMethods,
) -> tuple[np.ndarray | tuple[int, int], ...]:
    """
    Consolidates two dictionaries of orders, giving precedence to the values in `orders`.
    This function merges `orders_defaults` and `orders`, with `orders` values
    taking precedence over `orders_defaults`. It then checks and fixes the consolidated
    orders using the `_as_orders_defaults` function and returns the final orders as a tuple.
    Args:
        orders: The orders dictionary with updated values.
        orders_defaults: The default orders dictionary.
    Returns:
        tuple: A tuple containing the consolidated orders.
    Examples:
        >>> orders = ([0], [[1, 2], [3, 4]], np.array([[1, 2], [3, 4]]))
        >>> orders_defaults = {'na': np.zeros((1,)), 'nb': np.zeros((2,2)), 'nc': np.zeros((2,)), 'nd': np.zeros((2,)), 'nf': np.zeros((2,)), 'theta': np.zeros((2,2))}
        >>> _update_orders(orders, orders_defaults, id_method="FIR")
        (array([0]), array([[1, 2], [3, 4]]), array([0, 0]), array([0, 0]), array([0, 0]), array([[1, 2], [3, 4]]))

        >>> orders = (2, [1, 2, 3], np.array([3, 4]))
        >>> _update_orders(orders, orders_defaults, id_method="FIR")
        Traceback (most recent call last):
            ...
        RuntimeError: Order for nb must be of shape (2, 2). Got (3,) instead.
    """
    orders_dict = dict(zip(METHOD_ORDERS[id_method], orders, strict=True))
    orders_up = dict(orders_defaults)
    orders_up.update(orders_dict)
    orders_up = _as_orders_defaults(orders_up, orders_defaults)
    return tuple(orders_up.values())


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


# TODO: learn how to provide overloads without extensively typing out irrelevant args
# @overload
# def system_identification(
#     y: np.ndarray,
#     u: np.ndarray,
#     id_method: AvailableMethods,
#     *orders: int | list[int] | list[list[int]] | np.ndarray,
#     IC: None = None,
# ): ...
# @overload
# def system_identification(
#     y: np.ndarray,
#     u: np.ndarray,
#     id_method: AvailableMethods,
#     *orders: tuple[int, int],
#     IC: ICMethods,
# ): ...


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
) -> IO_SISO_Model | IO_MIMO_Model | SS_Model:
    # Verify y and u
    y = np.atleast_2d(y)
    u = np.atleast_2d(u)
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

    _verify_orders_types(*orders, IC=IC)

    _verify_orders_len(id_method, *orders, ydim=ydim, IC=IC)

    # Data centering
    y, u, y_rif = _recentering_fit_transform(y, u, centering)

    ##### Check Information Criterion #####

    # MODE 1) fixed orders
    if not _areinstances(orders, tuple):
        if IC is not None:
            warn("Ignoring argument 'IC' as fixed orders are provided.")

        orders = cast(
            tuple[int | list[int] | list[list[int]] | np.ndarray, ...], orders
        )
        orders_defaults: Mapping[str, np.ndarray] = {
            "na": np.zeros((ydim,), dtype=int),
            "nb": np.zeros((ydim, udim), dtype=int),
            "nc": np.zeros((ydim,), dtype=int),
            "nd": np.zeros((ydim,), dtype=int),
            "nf": np.zeros((ydim,), dtype=int),
            "theta": np.zeros((ydim, udim), dtype=int),
        }
        orders = _update_orders(orders, orders_defaults, id_method=id_method)

        # IO Models
        if id_method in get_args(IOMethods):
            id_method = cast(IOMethods, id_method)
            if id_mode in get_args(AvailableModes):
                flag = ID_MODES[id_mode]
            else:
                raise RuntimeError(
                    f"Method {id_mode} not available for {id_method}. Available: {get_args(AvailableModes)}"
                )

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
            id_method = cast(SSMethods, id_method)
            order = orders[0]
            model = SS_Model._identify(
                y,
                u,
                id_method,
                order,
                SS_f,
                SS_p,
                SS_threshold,
                SS_D_required,
                SS_PK_B_reval,
            )

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

        orders = cast(tuple[tuple[int, int], ...], orders)
        orders_ranges_defaults: Mapping[str, tuple[int, int]] = {
            "na": (0, 0),
            "nb": (0, 0),
            "nc": (0, 0),
            "nd": (0, 0),
            "nf": (0, 0),
            "theta": (0, 0),
        }
        orders = _update_orders(
            orders, orders_ranges_defaults, id_method=id_method
        )

        if id_method in get_args(IOMethods):
            id_method = cast(IOMethods, id_method)
            if id_mode in get_args(AvailableModes):
                flag = ID_MODES[id_mode]
            else:
                raise RuntimeError(
                    f"Method {id_mode} not available for {id_method}. Available: {get_args(AvailableModes)}"
                )

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
        elif id_method in get_args(SSMethods):
            id_method = cast(SSMethods, id_method)
            order = orders[0]
            model = SS_Model._from_order(
                y,
                u,
                id_method,
                order,
                IC,
                SS_f,
                SS_p,
                SS_D_required,
                SS_PK_B_reval,
            )

        # NO method selected
        else:
            raise RuntimeError(
                f"Wrong identification method selected. Got {id_method}"
                f"expected one of {get_args(AvailableMethods)}"
            )

    return model
