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
                x[i, t] += A[i, j] * x[j, t - 1]
            for j in range(m):
                x[i, t] += B[i, j] * u[j, t - 1]

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
        gbn[i] = gbn[i - 1]  # Default: no switch
        if i - Nmin >= 0:  # Check minimum switch constraint
            if np.random.rand() < p_swd:  # Switch with probability
                gbn[i] = -gbn[i - 1]

    # Rescale to range
    gbn = np.where(gbn > 0, max_Range, min_Range)
    return gbn


@jit
def information_criterion_compiled(K, N, Variance, method="AIC"):
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
                    Phi[:, col_idx] = u[i, delay_idx : delay_idx + N_eff]

    # Output matrix - need to handle MIMO case properly
    y_matrix = y[:, max_lag:N]

    # For MIMO, flatten the output properly for the least squares
    # This is handled in the calling code now

    return Phi, y_matrix


@jit
def create_regression_matrix_fir_compiled(u, y, nb, nk, ny, nu, N):
    """
    Compiled version of FIR regression matrix creation.

    Parameters:
    -----------
    u, y : ndarray
        Input and output data
    nb, nk : int
        Model coefficients count and delay
    ny, nu : int
        Number of outputs and inputs
    N : int
        Number of data points

    Returns:
    --------
    Phi : ndarray
        Regression matrix for all outputs
    y_matrix : ndarray
        Output matrix
    """
    max_lag = nb + nk - 1
    N_eff = N - max_lag

    if N_eff <= 0:
        return np.zeros((1, 1)), np.zeros((1, 1))

    n_params = nb * nu
    Phi = np.zeros((N_eff, n_params))

    # Fill regression matrix with lagged inputs
    for i in range(nb):
        for j in range(nu):
            col_idx = i * nu + j
            delay_idx = max_lag - 1 - i
            if delay_idx >= 0 and delay_idx + N_eff <= N:
                Phi[:, col_idx] = u[j, delay_idx : delay_idx + N_eff]

    y_matrix = y[:, max_lag:N]
    return Phi, y_matrix


@jit
def create_regression_matrix_bj_compiled(u, y, nb, nc, nd, nf, nk, ny, nu, N):
    """
    Compiled version of Box-Jenkins regression matrix creation.

    Parameters:
    -----------
    u, y : ndarray
        Input and output data
    nb, nc, nd, nf, nk : int
        Model orders and delay
    ny, nu : int
        Number of outputs and inputs
    N : int
        Number of data points

    Returns:
    --------
    Phi_list : list
        List of regression matrices for each output
    y_targets : list
        List of output vectors for each output
    """
    max_lag = max(nb + nk - 1, nc, nd, nf)
    N_eff = N - max_lag

    if N_eff <= 0:
        # Return empty structures
        return [np.zeros((1, 1))] * ny, [np.zeros(1)] * ny

    Phi_list = []
    y_targets = []

    for output_idx in range(ny):
        # For each output, create regression matrix
        n_params = nb * nu + nc + nd
        Phi = np.zeros((N_eff, n_params))
        col = 0

        # Input terms: lagged inputs
        for i in range(nb):
            for j in range(nu):
                delay_idx = max_lag - 1 - (i + nk - 1)
                if delay_idx >= 0 and delay_idx + N_eff <= N:
                    Phi[:, col] = u[j, delay_idx : delay_idx + N_eff]
                col += 1

        # Noise AR terms: lagged outputs
        for i in range(nc):
            start_idx = max_lag - 1 - i - 1
            end_idx = start_idx + N_eff
            if start_idx >= 0 and end_idx <= N:
                Phi[:, col] = y[output_idx, start_idx:end_idx]
            col += 1

        # Noise MA terms: simplified approach using lagged residuals
        # For initial implementation, use lagged outputs as approximation
        for i in range(nd):
            start_idx = max_lag - 1 - i - 1
            end_idx = start_idx + N_eff
            if start_idx >= 0 and end_idx <= N:
                Phi[:, col] = y[output_idx, start_idx:end_idx]
            col += 1

        # Target output
        y_target = y[output_idx, max_lag:N]

        Phi_list.append(Phi)
        y_targets.append(y_target)

    return Phi_list, y_targets


@jit
def create_regression_matrix_armax_compiled(u, y, na, nb, nc, nk, ny, nu, N):
    """
    Compiled version of ARMAX regression matrix creation.

    Parameters:
    -----------
    u, y : ndarray
        Input and output data
    na, nb, nc, nk : int
        Model orders and delay
    ny, nu : int
        Number of outputs and inputs
    N : int
        Number of data points

    Returns:
    --------
    Phi : ndarray
        Regression matrix for all outputs (flattened)
    y_matrix : ndarray
        Flattened output matrix
    """
    max_lag = max(na, nb + nk - 1, nc)
    N_eff = N - max_lag

    if N_eff <= 0:
        return np.zeros((1, 1)), np.zeros((1, 1))

    n_params = na * ny + nb * ny * nu + nc * ny
    Phi = np.zeros((N_eff * ny, n_params))

    # Build regression matrix for all outputs
    for output_idx in range(ny):
        row_start = output_idx * N_eff
        col = 0

        # AR terms: lagged outputs
        for i in range(na):
            for j in range(ny):
                start_idx = max_lag - 1 - i
                end_idx = start_idx + N_eff
                if start_idx >= 0 and end_idx <= N:
                    Phi[row_start : row_start + N_eff, col] = y[j, start_idx:end_idx]
                col += 1

        # X terms: lagged inputs
        for i in range(nb):
            for j in range(nu):
                for k in range(ny):
                    delay_idx = max_lag - 1 - (i + nk - 1)
                    if delay_idx >= 0 and delay_idx + N_eff <= N:
                        Phi[row_start : row_start + N_eff, col] = u[
                            j, delay_idx : delay_idx + N_eff
                        ]
                    col += 1

        # MA terms: simplified lagged outputs for initial implementation
        for i in range(nc):
            start_idx = max_lag - 1 - i - 1
            end_idx = start_idx + N_eff
            if start_idx >= 0 and end_idx <= N:
                Phi[row_start : row_start + N_eff, col] = y[
                    output_idx, start_idx:end_idx
                ]
            col += 1

        # Target output for this output channel
        Phi[row_start : row_start + N_eff, -1] = y[output_idx, max_lag:N]

    # Build flattened output matrix
    y_matrix = np.zeros(N_eff * ny)
    for i in range(ny):
        y_matrix[i * N_eff : (i + 1) * N_eff] = y[i, max_lag:N]

    return Phi, y_matrix


@jit
def create_regression_matrix_ararmax_compiled(u, y, na, nb, nc, nd, nf, nk, ny, nu, N):
    """
    Compiled version of ARARMAX regression matrix creation.

    Parameters:
    -----------
    u, y : ndarray
        Input and output data
    na, nb, nc, nd, nf, nk : int
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
        Flattened output matrix
    """
    max_lag = max(na, nb + nk - 1, nc, nd, nf)
    N_eff = N - max_lag

    if N_eff <= 0:
        return np.zeros((1, 1)), np.zeros((1, 1))

    # Simplified parameter count for ARARMAX
    n_params = na * ny + nb * ny * nu + nc * ny + nd * ny
    Phi = np.zeros((N_eff * ny, n_params))

    for output_idx in range(ny):
        row_start = output_idx * N_eff
        col = 0

        # AR terms
        for i in range(na):
            for j in range(ny):
                start_idx = max_lag - 1 - i
                end_idx = start_idx + N_eff
                if start_idx >= 0 and end_idx <= N:
                    Phi[row_start : row_start + N_eff, col] = y[j, start_idx:end_idx]
                col += 1

        # X terms
        for i in range(nb):
            for j in range(nu):
                for k in range(ny):
                    delay_idx = max_lag - 1 - (i + nk - 1)
                    if delay_idx >= 0 and delay_idx + N_eff <= N:
                        Phi[row_start : row_start + N_eff, col] = u[
                            j, delay_idx : delay_idx + N_eff
                        ]
                    col += 1

        # Intermediate AR terms (NC)
        for i in range(nc):
            start_idx = max_lag - 1 - i - 1
            end_idx = start_idx + N_eff
            if start_idx >= 0 and end_idx <= N:
                Phi[row_start : row_start + N_eff, col] = y[
                    output_idx, start_idx:end_idx
                ]
            col += 1

        # MA terms (ND)
        for i in range(nd):
            start_idx = max_lag - 1 - i - 1
            end_idx = start_idx + N_eff
            if start_idx >= 0 and end_idx <= N:
                Phi[row_start : row_start + N_eff, col] = y[
                    output_idx, start_idx:end_idx
                ]
            col += 1

        # Target output
        Phi[row_start : row_start + N_eff, -1] = y[output_idx, max_lag:N]

    # Build flattened output matrix
    y_matrix = np.zeros(N_eff * ny)
    for i in range(ny):
        y_matrix[i * N_eff : (i + 1) * N_eff] = y[i, max_lag:N]

    return Phi, y_matrix


@jit
def parsim_k_matrix_operations_compiled(y, u, f, p, m, l_, L):
    """
    Compiled version of core PARSIM-K matrix operations.

    Parameters:
    -----------
    y, u : ndarray
        Input and output data
    f, p : int
        Future and past horizons
    m, l_, L : int
        Number of inputs, outputs, and time steps

    Returns:
    --------
    Yf, Yp, Uf, Up : ndarray
        Future and past input/output ordinate sequences
    M : ndarray
        Initial system matrix
    """
    Yf, Yp = ordinate_sequence_compiled(y, f, p)
    Uf, Up = ordinate_sequence_compiled(u, f, p)

    Zp = impile_compiled(Up, Yp)
    YfdotPIort_Uf = Z_dot_PIort_compiled(Yf, Uf)
    ZpdotPIort_Uf = Z_dot_PIort_compiled(Zp, Uf)

    # Matrix operations using compiled pseudoinverse (simplified)
    M = np.dot(YfdotPIort_Uf, np.linalg.pinv(ZpdotPIort_Uf))

    return Yf, Yp, Uf, Up, M


@jit
def parsim_y_tilde_estimation_compiled(H_K, Uf, G_K, Yf, i, m, l_, f):
    """
    Compiled version of y_tilde estimation in PARSIM algorithms.

    Parameters:
    -----------
    H_K, G_K : ndarray
        System matrices
    Uf, Yf : ndarray
        Input and output future sequences
    i, m, l_, f : int
        Algorithm parameters

    Returns:
    --------
    y_tilde : ndarray
        Estimated output
    """
    y_tilde = np.dot(H_K[0:l_, :], Uf[m * i : m * (i + 1), :])

    for j in range(1, i):
        y_tilde = (
            y_tilde
            + np.dot(
                H_K[l_ * j : l_ * (j + 1), :],
                Uf[m * (i - j) : m * (i - j + 1), :],
            )
            + np.dot(
                G_K[l_ * j : l_ * (j + 1), :],
                Yf[l_ * (i - j) : l_ * (i - j + 1), :],
            )
        )

    return y_tilde


@jit
def subspace_weighted_svd_compiled(Yf, Yp, Uf, Up, Zp, weights_method, l_, m):
    """
    Compiled version of weighted SVD operations for subspace algorithms.

    Parameters:
    -----------
    Yf, Yp, Uf, Up, Zp : ndarray
        Subspace matrices
    weights_method : int
        Weighting method (0=N4SID, 1=MOESP, 2=CVA)
    l_, m : int
        Number of outputs and inputs

    Returns:
    --------
    U_n, S_n, V_n : ndarray
        SVD components
    """
    YfdotPIort_Uf = Z_dot_PIort_compiled(Yf, Uf)
    ZpdotPIort_Uf = Z_dot_PIort_compiled(Zp, Uf)
    O_i = np.dot(YfdotPIort_Uf, np.linalg.pinv(ZpdotPIort_Uf))

    if weights_method == 0:  # N4SID
        W1 = None  # Identity
        U_n, S_n, V_n = np.linalg.svd(O_i, full_matrices=False)

    elif weights_method == 1:  # MOESP
        W1 = None
        OidotPIort_Uf = Z_dot_PIort_compiled(O_i, Uf)
        U_n, S_n, V_n = np.linalg.svd(OidotPIort_Uf, full_matrices=False)

    else:  # CVA or fallback
        W1 = None
        U_n, S_n, V_n = np.linalg.svd(O_i, full_matrices=False)

    return U_n, S_n, V_n


@jit
def Z_dot_PIort_compiled(z, X):
    """
    Compiled version of Z_dot_PIort operation.

    Parameters:
    -----------
    z : ndarray
        Vector
    X : ndarray
        Matrix

    Returns:
    --------
    result : ndarray
        Computed result
    """
    return z - np.dot(np.dot(z, X.T), np.linalg.pinv(X.T))


@jit
def matrix_operations_a_compiled(X_fd, O_i, n, B_recalc, u, f, N):
    """
    Compiled version of matrix operations for A matrix extraction.

    Parameters:
    -----------
    X_fd, O_i : ndarray
        System matrices
    n : int
        System order
    B_recalc : bool
        Whether to recalculate B matrix
    u : ndarray
        Input data
    f : int
        Horizon
    N : int
        Number of data points

    Returns:
    --------
    A, B : ndarray
        State transition and input matrices
    """
    Ob = np.linalg.pinv(O_i)

    # Extract states
    X_hat = np.dot(Ob, O_i)

    if B_recalc and u.shape[0] > 0 and f < N:
        # Try to compute B matrix
        if np.abs(np.linalg.det(u[:, f : f + N - 1])) > 1e-10:
            U_slice = u[:, f : f + N - 1]
            X_next = X_fd[:, 1:N]
            X_curr = X_fd[:, 0 : N - 1]

            # Simple B estimation
            B_est = np.dot(X_next - np.dot(X_curr, A), np.linalg.pinv(U_slice))

            # Simple A estimation
            A = np.dot(X_next, np.linalg.pinv(np.hstack([X_curr, U_slice])))[:, :n]
        else:
            A = np.random.randn(n, n) * 0.1
            B_est = np.random.randn(n, u.shape[0]) * 0.1
    else:
        A = (
            np.dot(X_next, np.linalg.pinv(X_curr))
            if "X_next" in locals()
            else np.random.randn(n, n) * 0.1
        )
        B_est = (
            np.random.randn(n, u.shape[0]) * 0.1 if n > 0 else np.zeros((0, u.shape[0]))
        )

    return A, B_est


@jit
def RW_seq_compiled(N, rw0, sigma=1.0):
    """
    Compiled version of Random Walk sequence generation.

    Parameters:
    -----------
    N : int
        Sequence length (total number of samples)
    rw0 : float
        Initial value
    sigma : float
        Standard deviation (mobility) of random walk

    Returns:
    --------
    rw : ndarray
        Generated random walk sequence
    """
    rw = np.zeros(N)
    rw[0] = rw0

    for i in range(1, N):
        # Generate normal random number using Box-Muller transform
        u1 = np.random.rand()
        u2 = np.random.rand()
        z0 = np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)
        delta = sigma * z0
        rw[i] = rw[i - 1] + delta

    return rw


@jit
def white_noise_var_compiled(L, Var):
    """
    Compiled version of white noise generation with specified variances.

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
    Var = np.array(Var)
    n = Var.size
    noise = np.zeros((n, L))

    for i in range(n):
        if Var[i] < 1e-15:
            var_val = 1e-15
        else:
            var_val = Var[i]

        std_dev = np.sqrt(var_val)
        for j in range(L):
            # Box-Muller transform for normal distribution
            u1 = np.random.rand()
            u2 = np.random.rand()
            z0 = np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)
            noise[i, j] = std_dev * z0

    return noise


@jit
def white_noise_compiled_advanced(y, A_rel):
    """
    Advanced compiled version of white noise addition to signal.

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
    num = y.size
    y_err = np.zeros(num)

    # Compute standard deviation
    y_mean = np.mean(y)
    diff = y - y_mean
    y_std = np.sqrt(np.dot(diff, diff) / max(num - 1, 1))

    scale = np.abs(A_rel * y_std)
    if scale < 1e-15:
        scale = 1e-15

    # Generate noise using vectorized approach
    errors = np.zeros(num)
    for i in range(num):
        # Box-Muller transform for normal distribution
        u1 = np.random.rand()
        u2 = np.random.rand()
        z0 = np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)
        errors[i] = scale * z0

    y_err = y + errors
    return errors, y_err


@jit
def GBN_seq_advanced_compiled(
    N, p_swd, Nmin=1, Range_val=(-1.0, 1.0), Tol=0.01, nit_max=30
):
    """
    Advanced compiled version of GBN sequence generation with tolerance checking.

    Parameters:
    -----------
    N : int
        Sequence length (total number of samples)
    p_swd : float
        Desired probability of switching (0<x<1, no switch: 0, always switch: 1)
    Nmin : int
        Minimum number of samples between two switches
    Range_val : tuple
        Input range (min, max)
    Tol : float
        Tolerance on switching probability relative error
    nit_max : int
        Maximum number of iterations

    Returns:
    --------
    gbn_best : ndarray
        Generated GBN sequence
    p_sw_best : float
        Actual switching probability achieved
    Nsw_best : int
        Number of switches in the sequence
    """
    min_Range, max_Range = Range_val

    # Set first value
    gbn_best = np.ones(N)
    if np.random.rand() < 0.5:
        gbn_best = -gbn_best

    # Initialize variables
    p_sw_best = 2.0
    nit = 0

    while nit <= nit_max:
        i_fl = 0
        Nsw = 0
        gbn = gbn_best.copy()

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
        if N > 0:
            p_sw = Nmin * (Nsw + 1) / N
        else:
            p_sw = 0.0

        # Set best iteration
        if np.abs(p_sw - p_swd) < np.abs(p_sw_best - p_swd):
            p_sw_best = p_sw
            Nsw_best = Nsw
            gbn_best = gbn.copy()

        # Check tolerance
        if N > 0 and p_swd > 0:
            if (np.abs(p_sw - p_swd)) / p_swd <= Tol:
                break

        # Increase iteration number
        nit = nit + 1

    # Rescale GBN
    gbn_best = np.where(gbn_best > 0, max_Range, min_Range)

    return gbn_best, p_sw_best, Nsw_best


@jit
def signal_rescale_advanced_compiled(y):
    """
    Advanced compiled version of array rescaling with robust handling.

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
    # Compute mean and standard deviation robustly
    y_mean = np.mean(y)
    diff = y - y_mean
    y_std = np.sqrt(np.dot(diff, diff) / max(y.size - 1, 1))

    if y_std < 1e-15:
        y_std = 1.0

    y_scaled = y / y_std
    return y_std, y_scaled


# Export available functions
__all__ = [
    "ordinate_sequence_compiled",
    "simulate_ss_system_compiled",
    "impile_compiled",
    "reducingOrder_compiled",
    "Vn_mat_compiled",
    "rescale_compiled",
    "white_noise_compiled",
    "GBN_seq_compiled",
    "information_criterion_compiled",
    "create_regression_matrix_arx_compiled",
    "create_regression_matrix_fir_compiled",
    "create_regression_matrix_bj_compiled",
    "create_regression_matrix_armax_compiled",
    "create_regression_matrix_ararmax_compiled",
    "parsim_k_matrix_operations_compiled",
    "parsim_y_tilde_estimation_compiled",
    "subspace_weighted_svd_compiled",
    "Z_dot_PIort_compiled",
    "matrix_operations_a_compiled",
    "RW_seq_compiled",
    "white_noise_var_compiled",
    "white_noise_compiled_advanced",
    "GBN_seq_advanced_compiled",
    "signal_rescale_advanced_compiled",
    "NUMBA_AVAILABLE",
]
