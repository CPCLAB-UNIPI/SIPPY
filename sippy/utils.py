from warnings import warn

import control as cnt
import numpy as np


def atleast_3d(arr: list | np.ndarray) -> np.ndarray:
    arr = np.array(arr)
    if arr.ndim == 1:
        return arr.reshape(1, 1, -1)
    elif arr.ndim == 2:
        return arr.reshape(1, *arr.shape)
    else:
        return arr


def check_valid_orders(dim: int, *orders: np.ndarray):
    for i, arg in enumerate(orders):
        if isinstance(arg, int) or arg.shape == ():
            continue

        if arg.shape[0] != dim:
            arg_is_vec = len(arg.shape) == 1
            raise RuntimeError(
                f"Argument {i} must be a {'vector' if arg_is_vec else 'matrix'}, whose dimensions must be equal to {dim}"
            )
        if not np.issubdtype(arg.dtype, np.integer) or np.min(arg) < 0:
            raise RuntimeError(
                f"Arguments must contain only positive int elements. Arg {i} violates this rule."
            )


def check_feasibility(G, H, id_method: str, stab_marg: float, stab_cons: bool):
    poles_G = np.abs(cnt.poles(G))
    poles_H = np.abs(cnt.poles(H))

    if len(poles_G) != 0 and len(poles_H) != 0:
        poles_G = max(poles_G)
        poles_H = max(poles_H)
        # TODO: verify with RBdC if correct setting this to zero. Raises warnings.
        # check_st_H = poles_H
        if poles_G > 1.0 or poles_H > 1.0:
            warn("One of the identified system is not stable")
            if stab_cons is True:
                raise RuntimeError(
                    f"Infeasible solution: the stability constraint has been violated, since the maximum pole is {max(poles_H, poles_G)} \
                        ... against the imposed stability margin {stab_marg}"
                )
            else:
                warn(
                    f"Consider activating the stability constraint. The maximum pole is {max(poles_H, poles_G)}  "
                )


def get_val_range(order_range: int | tuple[int, int]):
    if isinstance(order_range, int):
        order_range = (order_range, order_range + 1)
    min_val, max_val = order_range
    if min_val < 0:
        raise ValueError("Minimum value must be non-negative")
    return range(min_val, max_val + 1)


def validate_and_prepare_inputs(
    u: np.ndarray, nb: int | np.ndarray, theta: int | np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Check input dimensions and ensure nb/theta are arrays."""
    u = np.atleast_2d(u)
    udim = u.shape[0]
    nb = np.atleast_1d(nb)
    theta = np.atleast_1d(theta)
    check_valid_orders(udim, nb, theta)
    return u, nb, theta, udim


def common_setup(
    na: int,
    nb: int | np.ndarray,
    nc: int,
    nd: int,
    nf: int,
    theta: int | np.ndarray,
) -> tuple[int, int]:
    nbth = nb + theta
    val = max(na, np.max(nbth), nc, nd, nf)
    n_coeff = na + np.sum(nb) + nc + nd + nf
    return val, n_coeff


def build_polynomial(order: int, coeffs: np.ndarray) -> cnt.TransferFunction:
    """
    Build a polynomial transfer function.
    This function constructs a transfer function of the form:
    H(s) = (s^order + 0*s^(order-1) + ... + 0*s + 1) / (s^order + coeffs[0]*s^(order-1) + ... + coeffs[order-1])
    Args:
        order (int): The order of the polynomial.
        coeffs (np.ndarray): The coefficients of the polynomial.
    Returns:
        cnt.TransferFunction: The resulting transfer function.
    Raises:
        RuntimeError: If the transfer function could not be obtained.
    Example:
        >>> import numpy as np
        >>> import control as cnt
        >>> coeffs = np.array([1, 2, 3])
        >>> tf = build_polynomial(3, coeffs)
        >>> tf
        TransferFunction(array([1, 0, 0, 0]), array([1, 1, 2, 3]), 1)
    """

    tf = cnt.tf([1] + [0] * order, [1] + list(coeffs), 1)
    if tf is None:
        raise RuntimeError("TF could not be obtained")
    return tf


def build_tf_G(
    THETA: np.ndarray,
    na: int,
    nb: np.ndarray,
    nc: int,
    nd: int,
    nf: int,
    theta: np.ndarray,
    id_method: str,
    udim: int,
    y_std: float = 1.0,
    U_std: np.ndarray = np.array([1.0]),
) -> tuple[np.ndarray, np.ndarray]:
    """Construct NUM, DEN, NUMH, DENH from parameters."""
    ng = na if id_method != "OE" else nf
    nb_total = np.sum(nb)
    nf_start = 0 if id_method == "OE" else na + nb_total + nc + nd
    indices = {
        "A": (0, na),
        "B": (ng, ng + nb_total),
        "F": (nf_start, nf_start + nf),
    }

    # Denominator polynomials
    A = build_polynomial(na, THETA[indices["A"][0] : indices["A"][1]])
    F = build_polynomial(nf, THETA[indices["F"][0] : indices["F"][1]])

    # Denominator calculations
    denG = np.array(cnt.tfdata(A * F)[1][0]) if A and F else np.array([1])

    # Numerator handling
    valG = max(np.max(nb + theta), na + nf)

    if id_method == "ARMA":
        NUM = np.ones((udim, 1))
    else:
        NUM = np.zeros((udim, valG))

    DEN = np.zeros((udim, valG + 1))

    for k in range(udim):
        if id_method != "ARMA":
            # TODO: verify whether this adjustment should be done prior to using THETA for polynomial calculations
            #  actual implementation is consistent with version 0.*.* of SIPPY
            b_slice = (
                THETA[
                    indices["B"][0] + np.sum(nb[:k]) : indices["B"][0]
                    + np.sum(nb[: k + 1])
                ]
                * y_std
                / U_std[k]
            )
            NUM[k, theta[k] : theta[k] + nb[k]] = b_slice

        DEN[k, : na + nf + 1] = denG

    return NUM, DEN


def build_tf_H(
    THETA: np.ndarray,
    na: int,
    nb: np.ndarray,
    nc: int,
    nd: int,
    _: int,
    __: np.ndarray,
    id_method: str,
    ___: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct NUM, DEN, NUMH, DENH from parameters."""
    nb_total = np.sum(nb)
    indices = {
        "A": (0, na),
        "C": (na + nb_total, na + nb_total + nc),
        "D": (na + nb_total + nc, na + nb_total + nc + nd),
    }

    # Denominator polynomials
    A = build_polynomial(na, THETA[indices["A"][0] : indices["A"][1]])
    D = build_polynomial(nd, THETA[indices["D"][0] : indices["D"][1]])

    # Denominator calculations
    denH = np.array(cnt.tfdata(A * D)[1][0]) if A and D else np.array([1])

    # Numerator handling
    valH = max(nc, na + nd)

    if id_method == "OE":
        NUMH = np.ones((1, 1))
    else:
        NUMH = np.zeros((1, valH + 1))
        NUMH[0, 0] = 1.0
        NUMH[0, 1 : nc + 1] = THETA[indices["C"][0] : indices["C"][1]]

    DENH = np.zeros((1, valH + 1))
    DENH[0, : na + nd + 1] = denH

    return NUMH, DENH


def build_tfs(
    THETA: np.ndarray,
    na: int,
    nb: np.ndarray,
    nc: int,
    nd: int,
    nf: int,
    theta: np.ndarray,
    id_method: str,
    udim: int,
    y_std: float = 1.0,
    U_std: np.ndarray = np.array([1.0]),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Construct NUM, DEN, NUMH, DENH from parameters."""
    NUM, DEN = build_tf_G(
        THETA, na, nb, nc, nd, nf, theta, id_method, udim, y_std, U_std
    )
    NUMH, DENH = build_tf_H(THETA, na, nb, nc, nd, nf, theta, id_method, udim)

    return NUM, DEN, NUMH, DENH
