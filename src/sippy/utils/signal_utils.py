"""
Signal generation and processing utilities.
"""

import warnings

import numpy as np

# Import compiled utilities for performance
try:
    from .compiled_utils import (
        NUMBA_AVAILABLE,
        GBN_seq_advanced_compiled,
        GBN_seq_compiled,
        RW_seq_compiled,
        signal_rescale_advanced_compiled,
        white_noise_compiled_advanced,
        white_noise_var_compiled,
    )
except ImportError:
    GBN_seq_compiled = None
    GBN_seq_advanced_compiled = None
    RW_seq_compiled = None
    white_noise_var_compiled = None
    white_noise_compiled_advanced = None
    signal_rescale_advanced_compiled = None
    NUMBA_AVAILABLE = False


def GBN_seq(N, p_swd, Nmin=1, Range=[-1.0, 1.0], Tol=0.01, nit_max=30):
    """
    Generate a Generalized Binary Noise (GBN) sequence.

    Parameters:
    -----------
    N : int
        Sequence length (total number of samples)
    p_swd : float
        Desired probability of switching (0<x<1, no switch: 0, always switch: 1)
    Nmin : int, optional
        Minimum number of samples between two switches
    Range : list, optional
        Input range [min, max]
    Tol : float, optional
        Tolerance on switching probability relative error
    nit_max : int, optional
        Maximum number of iterations

    Returns:
    --------
    gbn_b : ndarray
        Generated GBN sequence
    p_sw_b : float
        Actual switching probability achieved
    Nswb : int
        Number of switches in the sequence
    """
    min_Range = min(Range)
    max_Range = max(Range)
    prob = np.random.random()

    # Use compiled versions when available
    if NUMBA_AVAILABLE:
        # Try advanced compiled version first (with tolerance checking)
        if GBN_seq_advanced_compiled is not None:
            try:
                return GBN_seq_advanced_compiled(
                    N, p_swd, Nmin, (min_Range, max_Range), Tol, nit_max
                )
            except Exception:
                # Fall back to simplified version if advanced fails
                pass

        # Use basic compiled version for simple cases
        if GBN_seq_compiled is not None and Tol == 0.01:
            try:
                gbn_b = GBN_seq_compiled(N, p_swd, Nmin, (min_Range, max_Range))
                # Calculate actual switching probability for basic case
                switches = np.sum(np.abs(np.diff(gbn_b)) > 0)
                p_sw_b = min(Nmin * (switches + 1) / N, N)  # cap at 1.0
                Nswb = switches
                return gbn_b, p_sw_b, Nswb
            except Exception:
                # Fall back to original implementation
                pass

    # Use original implementation when compiled versions not available
    # Set first value
    if prob < 0.5:
        gbn = -1.0 * np.ones(N)
    else:
        gbn = 1.0 * np.ones(N)

    # Initialize variables
    p_sw = p_sw_b = 2.0  # actual switch probability
    nit = 0

    while (np.abs(p_sw - p_swd)) / p_swd > Tol and nit <= nit_max:
        i_fl = 0
        Nsw = 0
        for i in range(N - 1):
            gbn[i + 1] = gbn[i]
            # Test switch probability
            if i - i_fl >= Nmin:
                prob = np.random.random()
                # Track last test of p_sw
                i_fl = i
                if prob < p_swd:
                    # Switch and then count it
                    gbn[i + 1] = -gbn[i + 1]
                    Nsw = Nsw + 1

        # Check actual switch probability
        p_sw = Nmin * (Nsw + 1) / N

        # Set best iteration
        if np.abs(p_sw - p_swd) < np.abs(p_sw_b - p_swd):
            p_sw_b = p_sw
            Nswb = Nsw
            gbn_b = gbn.copy()

        # Increase iteration number
        nit = nit + 1

    # Rescale GBN
    gbn_b = np.where(gbn_b > 0, max_Range, min_Range)
    return gbn_b, p_sw_b, Nswb


def RW_seq(N, rw0, sigma=1):
    """
    Generate a Random Walk sequence.

    This function automatically uses the Numba-compiled version when available
    for improved performance.

    Parameters:
    -----------
    N : int
        Sequence length (total number of samples)
    rw0 : float
        Initial value
    sigma : float, optional
        Standard deviation (mobility) of random walk

    Returns:
    --------
    rw : ndarray
        Generated random walk sequence
    """
    if NUMBA_AVAILABLE and RW_seq_compiled is not None:
        try:
            return RW_seq_compiled(N, rw0, sigma)
        except Exception:
            # Fallback to original implementation
            pass

    # Original implementation
    rw = rw0 * np.ones(N)
    for i in range(N - 1):
        # Return random sample from a normal distribution
        delta = np.random.normal(0.0, sigma, 1)
        # Refresh input
        rw[i + 1] = rw[i] + delta
    return rw


def white_noise(y, A_rel):
    """
    Add white noise to a signal.

    This function automatically uses the Numba-compiled version when available
    for improved performance.

    Parameters:
    -----------
    y : ndarray
        Clean signal
    A_rel : float
        Relative amplitude (0<x<1) to the standard deviation of y

    Returns:
    --------
    errors : ndarray
        Generated noise
    y_err : ndarray
        Signal with added noise
    """
    if NUMBA_AVAILABLE and white_noise_compiled_advanced is not None:
        try:
            return white_noise_compiled_advanced(y, A_rel)
        except Exception:
            # Fallback to original implementation
            pass

    # Original implementation
    num = y.size
    y_err = np.zeros(num)
    Ystd = np.std(y)
    scale = np.abs(A_rel * Ystd)

    if scale < np.finfo(np.float32).eps:
        scale = np.finfo(np.float32).eps
        warnings.warn("A_rel may be too small, its value set to the lowest default one")

    errors = np.random.normal(0.0, scale, num)
    y_err = y + errors
    return errors, y_err


def white_noise_var(L, Var):
    """
    Generate white noise matrix with specified variances.

    This function automatically uses the Numba-compiled version when available
    for improved performance.

    Parameters:
    -----------
    L : int
        Number of samples (columns)
    Var : array_like
        Variance vector for each row

    Returns:
    --------
    noise : ndarray
        Noise matrix with shape (len(Var), L)
    """
    if NUMBA_AVAILABLE and white_noise_var_compiled is not None:
        try:
            return white_noise_var_compiled(L, Var)
        except Exception:
            # Fallback to original implementation
            pass

    # Original implementation
    Var = np.array(Var)
    n = Var.size
    noise = np.zeros((n, L))

    for i in range(n):
        if Var[i] < np.finfo(np.float32).eps:
            Var[i] = np.finfo(np.float32).eps
            warnings.warn(
                f"Var[{i}] may be too small, its value set to the lowest default one"
            )
        noise[i, :] = np.random.normal(0.0, Var[i] ** 0.5, L)

    return noise


def rescale(y):
    """
    Rescale array to its standard deviation.

    Parameters:
    -----------
    y : ndarray
        Input signal

    Returns:
    --------
    ystd : float
        Standard deviation of y
    y_scaled : ndarray
        y rescaled by its standard deviation
    """
    ystd = np.std(y)
    y_scaled = y / ystd
    return ystd, y_scaled


def information_criterion(K, N, Variance, method="AIC"):
    """
    Calculate information criterion for model selection.

    Parameters:
    -----------
    K : int
        Number of parameters
    N : int
        Number of data points
    Variance : float
        Model residual variance
    method : str, optional
        Information criterion type ('AIC', 'AICc', 'BIC')

    Returns:
    --------
    IC : float
        Information criterion value
    """
    if method == "AIC":
        IC = N * np.log(Variance) + 2 * K
    elif method == "AICc":
        if N - K - 1 > 0:
            IC = N * np.log(Variance) + 2 * K + 2 * K * (K + 1) / (N - K - 1)
        else:
            raise ValueError(
                "Number of data is less than the number of parameters, AICc cannot be applied"
            )
    elif method == "BIC":
        IC = N * np.log(Variance) + K * np.log(N)
    else:
        raise ValueError(f"Unknown method: {method}")
    return IC


def mean_square_error(predictions, targets):
    """
    Calculate mean square error.

    Parameters:
    -----------
    predictions : ndarray
        Predicted values
    targets : ndarray
        Target values

    Returns:
    --------
    mse : float
        Mean square error
    """
    return ((predictions - targets) ** 2).mean()
