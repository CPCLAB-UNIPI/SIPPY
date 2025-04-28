from math import gcd
from typing import Literal, overload

import numpy as np
import sympy as sp
from scipy.signal import tf2ss as tf2ss_siso


def _pad_numerators(num_list):
    """
    Pad the numerator matrix of polynomials with zeros to have all equal length, padding from the left.

    Parameters:
        num_list: List of lists of numerators for each (output, input) transfer function.

    Returns:
        Padded numerator list with all numerators having the same length.

    >>> _pad_numerators([[[1], [1, 2]], [[1, 2, 3], [1]]])
    [[[0, 0, 1], [0, 1, 2]], [[1, 2, 3], [0, 0, 1]]]
    """
    max_len = max(len(num) for row in num_list for num in row)
    padded_num_list = [
        [
            [0 if all(isinstance(r, int) for r in num) else 0.0]
            * (max_len - len(num))
            + num
            for num in row
        ]
        for row in num_list
    ]
    return padded_num_list


def list_to_poly(coefs, s=sp.Symbol("s")):
    """
    Convert a list of coefficients (in descending order) into a sympy Poly.

    Parameters:
      coefs: List of coefficients (descending order).
             For example, [1, 3, 2] represents 1*s**2 + 3*s + 2.
      s: sympy symbol (default is s).

    Returns:
      A sympy Poly object.

    >>> p = list_to_poly([1, 3, 2])
    >>> sp.expand(p.as_expr())
    s**2 + 3*s + 2
    >>> sp.expand(p.as_expr()) == sp.expand(sp.Symbol('s')**2 + 3*sp.Symbol('s') + 2)
    True
    """
    poly_expr = sum(
        coef * s ** (len(coefs) - i - 1) for i, coef in enumerate(coefs)
    )
    return sp.Poly(poly_expr, s)


def compute_lcd_from_den_list(den_list, s=sp.Symbol("s")):
    """
    Compute the least common denominator (LCD) of a MIMO system's denominators.

    Parameters:
      den_list: A list of lists of denominators. Each denominator is a list of coefficients
                in descending order.
      s: sympy symbol (default is s).

    Returns:
      A sympy Poly representing the LCD.

    >>> lcd = compute_lcd_from_den_list([[[1, 2]], [[1, 2]]])
    >>> sp.expand(lcd.as_expr())
    s + 2
    """
    # Start with the first denominator in the list
    first_poly = list_to_poly(den_list[0][0], s)
    lcd = first_poly
    for row in den_list:
        for den in row:
            poly = list_to_poly(den, s)
            lcd = sp.lcm(lcd, poly)
    return lcd


def compute_adjusted_num(num, lcd, den, s=sp.Symbol("s")):
    """
    Compute the adjusted numerator polynomial coefficients given the numerator and denominator
    of a transfer function and the common LCD.

    This function multiplies the original numerator by the LCD and divides by the original
    denominator. The resulting quotient (assumed to be exact) gives the adjusted numerator.

    Parameters:
      num: List of numerator coefficients (in descending order).
      lcd: A sympy Poly representing the least common denominator.
      den: List of denominator coefficients (in descending order).
      s: sympy symbol (default is s).

    Returns:
      A numpy array of adjusted numerator coefficients (in descending order).

    >>> coeffs = compute_adjusted_num([1, 1], list_to_poly([1, 3, 2]), [1, 3, 2])
    >>> coeffs.tolist()
    [1.0, 1.0]
    """
    num_poly = list_to_poly(num, s)
    den_poly = list_to_poly(den, s)
    # Multiply numerator polynomial by LCD
    new_expr = sp.expand(num_poly.as_expr() * lcd.as_expr())
    new_poly = sp.Poly(new_expr, s)
    quotient, remainder = sp.div(new_poly, den_poly)
    if remainder.as_expr() != 0:
        raise ValueError("Adjusted numerator division has non-zero remainder")
    return np.array(quotient.all_coeffs(), dtype=np.float64)


def transpose(matrix):
    """
    Transpose a list of lists (matrix).

    Parameters:
        matrix: List of lists to be transposed.

    Returns:
        Transposed list of lists.

    >>> transpose([[1, 2, 3], [4, 5, 6]])
    [[1, 4], [2, 5], [3, 6]]
    """
    return [list(row) for row in zip(*matrix)]


def state_space_from_poly(poly):
    """
    Compute the state-space representation (A, B) from the denominator polynomial using tf2ss.

    Parameters:
      poly: A sympy Poly representing the denominator.

    Returns:
      A, B: Matrices from the tf2ss representation of the system with transfer function 1/poly.

    >>> A, B = state_space_from_poly(list_to_poly([1, 2]))
    >>> A.shape
    (1, 1)
    >>> B.shape
    (1, 1)
    """
    lcd_coeffs = np.array(poly.all_coeffs(), dtype=np.float64)
    A, B, _, _ = tf2ss_siso([1], lcd_coeffs)
    return A, B


@overload
def _get_lcm_norm_coeffs(
    den_list: list[list[list[float]]],
    mode: Literal["global", "local"] = "global",
) -> list[float]: ...
@overload
def _get_lcm_norm_coeffs(
    den_list: list[list[list[float]]],
    mode: Literal["global", "local"] = "local",
) -> list[list[float]]: ...
def _get_lcm_norm_coeffs(
    den_list: list[list[list[float]]],
    mode: Literal["global", "local"] = "global",
) -> list[float] | list[list[float]]:
    """
    Compute the least common multiple (LCM) of a list of floating-point polynomials.

    Parameters:
      den_list: A list of lists of lists of floating-point coefficients representing the denominators.
      mode: A string indicating the mode of LCM computation. Can be "global" or "local".
            "global" computes a single LCM for all denominators.
            "local" computes LCMs for each input of denominators.

    Returns:
      If mode is "global", returns a list of floating-point coefficients representing the LCM.
      If mode is "local", returns a list of lists of floating-point coefficients representing the LCMs for each column.

    Examples:
    >>> den_list = [[[1.0, 2.0], [1.0, 1.0]], [[1.0, 2.0], [1.0, 3.0]]]
    >>> _get_lcm_norm_coeffs(den_list, mode="global")
    [1.0, 6.0, 11.0, 6.0]

    >>> den_list = [[[1.0, 2.0], [1.0, 1.0]], [[1.0, 2.0], [1.0, 3.0]]]
    >>> _get_lcm_norm_coeffs(den_list, mode="local")
    [[1.0, 2.0, 0.0], [1.0, 4.0, 3.0]]
    """
    if mode == "local":
        # TransferFunction.common_den() right-pads the denominators with zeros
        normalized_coeffs_list = [
            _get_lcm_norm_coeffs([col], "global")
            for col in transpose(den_list)
        ]
        max_len = max(len(coeffs) for coeffs in normalized_coeffs_list)
        return [
            coeffs + [0.0] * (max_len - len(coeffs))
            for coeffs in normalized_coeffs_list
        ]
    else:
        lcm_poly = compute_lcd_from_den_list(den_list)
        # Extract coefficients as floats
        lcd_coeffs = [float(c) for c in lcm_poly.all_coeffs()]

        # Normalize by the greatest common divisor of integer coefficients
        coeff_gcd = gcd(*(int(c) for c in lcd_coeffs if c != 0))
        normalized_coeffs = [c / coeff_gcd for c in lcd_coeffs]
    return normalized_coeffs


def rjust(list_, width):
    """
    Examples:
    >>> rjust([1, 2, 3], 4)
    [1, 2, 3, 0]
    >>> rjust([1, 2, 3, 4, 5], 4)
    [1, 2, 3, 4]
    """

    return list(list_[:width]) + [0] * max(width - len(list_), 0)


def controllable_canonical_form(denominator: list[float] | sp.Poly) -> tuple:
    """
    Compute the controllable canonical form (A, B) matrices for a given common denominator polynomial.

    Parameters:
    denominator (list): Coefficients of the denominator polynomial [a_n, ..., a_1, a_0],
                        where the highest-degree term is first.

    Returns:
    tuple: (A, B) state-space representation in controllable canonical form.

    Examples:
    >>> controllable_canonical_form([1, 2, 3])
    (array([[ 0.,  1.],
           [-2., -3.]]), array([[0.],
           [1.]]))

    """
    if isinstance(denominator, np.ndarray):
        denominator = list(denominator)
    elif isinstance(denominator, sp.Poly):
        denominator = list(
            np.array(denominator.all_coeffs(), dtype=np.float64)
        )

    n = len(denominator) - 1  # System order
    if n < 1:
        return np.array([0.0]), np.array([0.0])
    # Construct the A matrix in controllable canonical form
    A = np.zeros((n, n))
    A[:-1, 1:] = np.eye(n - 1)  # Upper diagonal ones
    A[-1, :] = -np.array(
        denominator[1:]
    )  # Last row is negative denominator coefficients

    # Construct the B matrix (last column is 1)
    B = np.zeros((n, 1))
    B[-1, 0] = 1

    return A, B


def tf2ss(num_list, den_list, minreal: bool = False):
    """
    Convert a MIMO transfer function to a minimal state-space realization.

    The method first computes a common denominator (LCD) for all channels, obtains the
    corresponding state-space representation (A, B) from the LCD, and then computes the
    output matrix C and feedthrough matrix D by adjusting each numerator accordingly.

    Parameters:
      num_list: List of lists of numerators for each (output, input) transfer function.
                Each numerator is a list of coefficients in descending order.
      den_list: List of lists of denominators for each (output, input) transfer function.
                Each denominator is a list of coefficients in descending order.

    Returns:
      A, B, C, D: Minimal state-space matrices of the MIMO system.

    >>> num_list = [[[1]], [[1]]]
    >>> den_list = [[[1, 2]], [[1, 2]]]
    >>> A, B, C, D = tf2ss(num_list, den_list)
    >>> A.shape[0] > 0
    True
    """
    # TODO: implement minimal realization. The functions used randomly change the shapes of the matrices...
    # sys = tf(num_list, den_list)
    # num_list, den_list, _ = sys.minreal()._common_den()
    # den_list = np.expand_dims(den_list, axis=0)
    # den_list = np.tile(den_list, (num_list.shape[0], 1, 1))
    num_list = np.vectorize(lambda x: sp.Rational(str(x)))(num_list)
    den_list = np.vectorize(lambda x: sp.Rational(str(x)))(den_list)
    n_outputs = len(num_list)
    n_inputs = len(num_list[0])
    s = sp.Symbol("s")
    # Step 1: Compute the LCD of all denominators.
    lcd = compute_lcd_from_den_list(den_list, s)
    # Step 2: Get state-space representation (A, B) from the LCD.
    A, B_scalar = controllable_canonical_form(lcd)
    n_states = A.shape[0]
    C = np.zeros((n_outputs, n_states))
    D = np.zeros((n_outputs, n_inputs))
    # Step 3: For each transfer function, compute the adjusted numerator and fill C and D.
    for i_out in range(n_outputs):
        for j_in in range(n_inputs):
            adjusted_num_coeffs = compute_adjusted_num(
                num_list[i_out][j_in], lcd, den_list[i_out][j_in], s
            )
            # In the controllable canonical (companion) form, the output matrix uses the reversed order.
            C[i_out, :] = rjust(adjusted_num_coeffs[::-1], n_states)
            # If the degree is less than n_states, assign the constant term to D.
            if len(adjusted_num_coeffs) < n_states:
                D[i_out, j_in] = adjusted_num_coeffs[-1]
            else:
                D[i_out, j_in] = 0
    B = np.tile(B_scalar, (1, n_inputs))
    return A, B, C, D


if __name__ == "__main__":
    import doctest

    doctest.testmod()
