"""
Numba-compiled utility functions for high-performance numerical computations.

This module contains JIT-compiled versions of frequently used computational
functions that are performance bottlenecks in system identification algorithms.
"""
import warnings

import numpy as np

try:
    import numba
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    warnings.warn("Numba not available. Using slower pure Python implementations.")
    NUMBA_AVAILABLE = False


# Fallback decorator when numba is not available
def fallback_njit(*args, **kwargs):
    """Fallback decorator when numba is not available."""
    if len(args) == 1 and callable(args[0]):
        # Called as @fallback_njit
        func = args[0]
        return func
    else:
        # Called as @fallback_njit(...)
        def decorator(func):
            return func
        return decorator


# Use numba if available, otherwise use fallback
jit = njit if NUMBA_AVAILABLE else fallback_njit


@jit
def ordinate_sequence_compiled(y, f, p):
    """
    Compiled version of ordinate sequence creation for subspace identification.
    
    Parameters:
    -----------
    y : ndarray
        Output data with shape (outputs, time_steps)
    f : int
        Future horizon
    p : int
        Past horizon
    
    Returns:
    --------
    Yf : ndarray
        Future output ordinate sequence
    Yp : ndarray
        Past output ordinate sequence
    """
    l, L = y.shape
    N = L - p - f + 1
    Yp = np.zeros((l * f, N))
    Yf = np.zeros((l * f, N))

    for i in range(f):
        for j in range(l):
            # Fill future sequence
            start_idx_f = p + i
            end_idx_f = L - f + i + 1
            Yf[l * i + j, :] = y[j, start_idx_f:end_idx_f]
            
            # Fill past sequence
            start_idx_p = i
            end_idx_p = L - f - p + i + 1
            Yp[l * i + j, :] = y[j, start_idx_p:end_idx_p]

    return Yf, Yp


@jit
def simulate_ss_system_compiled(A, B, C, D, u, x0=None):
    """
    Compiled version of state-space system simulation.
    
    Parameters:
    -----------
    A, B, C, D : ndarray
        State-space matrices
    u : ndarray
        Input signals (inputs x time_steps)
    x0 : ndarray, optional
        Initial state
    
    Returns:
    --------
    x : ndarray
        State trajectory
    y : ndarray
        Output signals
    """
    m, L = u.shape
    l, n = C.shape
    y = np.zeros((l, L))
    x = np.zeros((n, L))

    if x0 is not None:
        x[:, 0] = x0[:, 0]

    # First time step
    for i in range(l):
        for j in range(n):
            y[i, 0] += C[i, j] * x[j, 0]
    for i in range(l):
        for j in range(m):
            y[i, 0] += D[i, j] * u[j, 0]

    # Remaining time steps
    for t in range(1, L):
        # State update: x[:, t] = A @ x[:, t-1] + B @ u[:, t-1]
        for i in range(n):
            x[i, t] = 0.0
            for j in range(n):
                x[i, t] += A[i, j] * x[j, t-1]
            for j in range(m):
                x[i, t] += B[i, j] * u[j, t-1]
        
        # Output update: y[:, t] = C @ x[:, t] + D @ u[:, t]
        for i in range(l):
            y[i, t] = 0.0
            for j in range(n):
                y[i, t] += C[i, j] * x[j, t]
            for j in range(m):
                y[i, t] += D[i, j] * u[j, t]

    return x, y


@jit
def impile_compiled(M1, M2):
    """
    Compiled version of matrix vertical stacking.
    
    Parameters:
    -----------
    M1, M2 : ndarray
        Matrices to stack
    
    Returns:
    --------
    M : ndarray
        Vertically stacked matrix
    """
    rows1, cols = M1.shape
    rows2, _ = M2.shape
    M = np.zeros((rows1 + rows2, cols))
    
    for i in range(rows1):
        for j in range(cols):
            M[i, j] = M1[i, j]
    
    for i in range(rows2):
        for j in range(cols):
            M[rows1 + i, j] = M2[i, j]
    
    return M


@jit
def reducingOrder_compiled(U_n, S_n, V_n, threshold=0.1, max_order=10):
    """
    Compiled version of model order reduction based on singular values.
    
    Parameters:
    -----------
    U_n, S_n, V_n : ndarray
        SVD components
    threshold : float
        Threshold for truncation
    max_order : int
        Maximum order to keep
    
    Returns:
    --------
    U_n, S_n, V_n : ndarray
        Truncated SVD components
    """
    s0 = S_n[0]
    index = S_n.size
    for i in range(S_n.size):
        if S_n[i] < threshold * s0 or i >= max_order:
            index = i
            break
    
    return U_n[:, 0:index], S_n[0:index], V_n[0:index, :]


@jit
def Vn_mat_compiled(y, yest):
    """
    Compiled version of residual variance computation.
    
    Parameters:
    -----------
    y : ndarray
        Process output
    yest : ndarray
        Estimated model output
    
    Returns:
    --------
    Vn : float
        Residual variance
    """
    eps = y - yest
    Vn = np.dot(eps, eps) / max(y.size, 1)
    return Vn


@jit
def rescale_compiled(y):
    """
    Compiled version of array rescaling to standard deviation.
    
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
    # Compute standard deviation
    y_mean = np.mean(y)
    diff = y - y_mean
    ystd = np.sqrt(np.dot(diff, diff) / max(y.size - 1, 1))
    
    if ystd < 1e-15:  # Avoid division by very small numbers
        ystd = 1.0
    
    y_scaled = y / ystd
    return ystd, y_scaled


@jit
def white_noise_compiled(L, Var):
    """
    Compiled version of white noise generation.
    
    Parameters:
    -----------
    L : int
        Number of samples (columns)
    Var : ndarray
        Variance vector for each row
    
    Returns:
    --------
    noise : ndarray
        Noise matrix with shape (len(Var), L)
    """
    n = Var.size
    noise = np.zeros((n, L))

    for i in range(n):
        if Var[i] < 1e-15:
            Var[i] = 1e-15
        std_dev = np.sqrt(Var[i])
        # Simple Box-Muller transform for normal distribution
        for j in range(L):
            # Generate uniform random numbers
            u1 = np.random.rand()
            u2 = np.random.rand()
            # Box-Muller transform
            z0 = np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)
            noise[i, j] = std_dev * z0

    return noise


@jit
def GBN_seq_compiled(N, p_swd, Nmin=1, Range_val=(-1.0, 1.0)):
    """
    Compiled version of Generalized Binary Noise sequence generation.
    
    Parameters:
    -----------
    N : int
        Sequence length
    p_swd : float
        Desired probability of switching
    Nmin : int
        Minimum number of samples between switches
    Range_val : tuple
        Range values (min, max)
    
    Returns:
    --------
    gbn : ndarray
        Generated GBN sequence
    """
    min_Range, max_Range = Range_val
    
    # Initialize sequence
    gbn = np.ones(N)
    
    # Random initial value
    if np.random.rand() < 0.5:
        gbn = -gbn
    
    # Generate sequence
    for i in range(1, N):
        gbn[i] = gbn[i-1]  # Default: no switch
        if i - Nmin >= 0:  # Check minimum switch constraint
            if np.random.rand() < p_swd:  # Switch with probability
                gbn[i] = -gbn[i-1]
    
    # Rescale to range
    gbn = np.where(gbn > 0, max_Range, min_Range)
    return gbn


@jit
def information_criterion_compiled(K, N, Variance, method='AIC'):
    """
    Compiled version of information criterion calculation.
    
    Parameters:
    -----------
    K : int
        Number of parameters
    N : int
        Number of data points
    Variance : float
        Model residual variance
    method : int
        Information criterion type (0=AIC, 1=AICc, 2=BIC)
    
    Returns:
    --------
    IC : float
        Information criterion value
    """
    if method == 0:  # AIC
        IC = N * np.log(Variance) + 2 * K
    elif method == 1:  # AICc
        if N - K - 1 > 0:
            IC = N * np.log(Variance) + 2 * K + 2 * K * (K + 1) / (N - K - 1)
        else:
            IC = 1e15  # Large value when calculation is not possible
    elif method == 2:  # BIC
        IC = N * np.log(Variance) + K * np.log(N)
    else:
        IC = 1e15  # Default large value
    
    return IC


@jit
def create_regression_matrix_arx_compiled(u, y, na, nb, nk, ny, nu, N):
    """
    Compiled version of ARX regression matrix creation.
    
    Parameters:
    -----------
    u, y : ndarray
        Input and output data
    na, nb, nk : int
        Model orders and delay
    ny, nu : int
        Number of outputs and inputs
    N : int
        Number of data points
    
    Returns:
    --------
    Phi : ndarray
        Regression matrix
    y_matrix : ndarray
        Output matrix
    """
    max_lag = max(na, nb + nk - 1)
    N_eff = N - max_lag
    
    if N_eff <= 0:
        # Return empty matrices if not enough data
        return np.zeros((1, 1)), np.zeros((1, 1))
    
    n_params = na * ny + nb * ny * nu
    Phi = np.zeros((N_eff, n_params))
    
    # Fill AR part (lagged outputs)
    for i in range(na):
        for j in range(ny):
            col_idx = i * ny + j
            start_idx = max_lag - 1 - i
            end_idx = max_lag - 1 - i + N_eff
            Phi[:, col_idx] = y[j, start_idx:end_idx]
    
    # Fill X part (lagged inputs)
    for k in range(nb):
        for i in range(nu):
            for j in range(ny):
                col_idx = na * ny + k * ny * nu + i * ny + j
                delay_idx = max_lag - 1 - (k + nk - 1)
                if delay_idx >= 0 and delay_idx + N_eff <= N:
                    Phi[:, col_idx] = u[i, delay_idx:delay_idx + N_eff]
    
    # Output matrix - need to handle MIMO case properly
    y_matrix = y[:, max_lag:N]
    
    # For MIMO, flatten the output properly for the least squares
    # This is handled in the calling code now
    
    return Phi, y_matrix


# Export available functions
__all__ = [
    'ordinate_sequence_compiled',
    'simulate_ss_system_compiled', 
    'impile_compiled',
    'reducingOrder_compiled',
    'Vn_mat_compiled',
    'rescale_compiled',
    'white_noise_compiled',
    'GBN_seq_compiled',
    'information_criterion_compiled',
    'create_regression_matrix_arx_compiled',
    'NUMBA_AVAILABLE'
]
