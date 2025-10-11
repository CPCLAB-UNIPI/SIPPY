"""
System simulation and analysis utilities.
"""

import warnings

import numpy as np
from scipy import fftpack, signal, stats
from scipy.linalg import solve_discrete_are

try:
    import harold
    HAROLD_AVAILABLE = True
except ImportError:
    HAROLD_AVAILABLE = False
    warnings.warn("harold library not available. Some simulation features will be limited.")


def ordinate_sequence(y, f, p):
    """
    Create ordinate sequences for subspace identification.

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

    for i in range(1, f + 1):
        Yf[l * (i - 1):l * i] = y[:, p + i - 1:L - f + i]
        Yp[l * (i - 1):l * i] = y[:, i - 1:L - f - p + i]

    return Yf, Yp


def Z_dot_PIort(z, X):
    """
    Compute the scalar product between vector z and I - X^T * pinv(X^T),
    avoiding direct computation of the full matrix.

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


def Vn_mat(y, yest):
    """
    Compute the variance of the model residuals.

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
    y = y.flatten()
    yest = yest.flatten()
    eps = y - yest
    Vn = (eps @ eps) / (max(y.shape))
    return Vn


def impile(M1, M2):
    """
    Stack two matrices vertically.

    Parameters:
    -----------
    M1, M2 : ndarray
        Matrices to stack

    Returns:
    --------
    M : ndarray
        Vertically stacked matrix
    """
    M = np.zeros((M1[:, 0].size + M2[:, 0].size, M1[0, :].size))
    M[0:M1[:, 0].size] = M1
    M[M1[:, 0].size::] = M2
    return M


def reducingOrder(U_n, S_n, V_n, threshold=0.1, max_order=10):
    """
    Reduce model order based on singular values.

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


def check_types(threshold, max_order, fixed_order, f, p=20):
    """
    Validate parameter types and values.

    Parameters:
    -----------
    threshold : float
        Threshold value
    max_order : float or int
        Maximum model order
    fixed_order : float or int
        Fixed model order
    f : int
        Future horizon
    p : int
        Past horizon

    Returns:
    --------
    valid : bool
        Whether parameters are valid
    """
    if threshold < 0. or threshold >= 1.:
        print("Error! The threshold value must be >=0. and <1.")
        return False
    if not np.isnan(max_order):
        if not isinstance(max_order, int):
            print("Error! The max_order value must be integer")
            return False
    if not np.isnan(fixed_order):
        if not isinstance(fixed_order, int):
            print("Error! The fixed_order value must be integer")
            return False
    if not isinstance(f, int):
        print("Error! The future horizon (f) must be integer")
        return False
    if not isinstance(p, int):
        print("Error! The past horizon (p) must be integer")
        return False
    return True


def check_inputs(threshold, max_order, fixed_order, f):
    """
    Adjust and validate input parameters.

    Parameters:
    -----------
    threshold : float
        Threshold value
    max_order : float or None
        Maximum model order
    fixed_order : float or None
        Fixed model order
    f : int
        Future horizon

    Returns:
    --------
    threshold : float
        Adjusted threshold
    max_order : int
        Adjusted maximum order
    """
    if not np.isnan(fixed_order):
        threshold = 0.0
        max_order = fixed_order
    if f < max_order:
        print('Warning! The horizon must be larger than the model order, max_order setted as f')
    if max_order >= f:
        max_order = f
    return threshold, max_order


def simulate_ss_system(A, B, C, D, u, x0=None):
    """
    Simulate state-space system in process form.

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

    y[:, 0] = np.dot(C, x[:, 0]) + np.dot(D, u[:, 0])

    for i in range(1, L):
        x[:, i] = np.dot(A, x[:, i - 1]) + np.dot(B, u[:, i - 1])
        y[:, i] = np.dot(C, x[:, i]) + np.dot(D, u[:, i])

    return x, y


def ssmatrix(data, axis=1):
    """
    Convert argument to a (possibly empty) state space matrix.

    Parameters:
    -----------
    data : array, list, or string
        Input data defining the contents of the 2D array
    axis : 0 or 1
        If input data is 1D, which axis to use for return object

    Returns:
    -------
    arr : ndarray
        2D array, with shape (0, 0) if empty
    """
    arr = np.array(data, dtype=float)
    ndim = arr.ndim
    shape = arr.shape

    # Change the shape of the array into a 2D array
    if ndim > 2:
        raise ValueError("state-space matrix must be 2-dimensional")
    elif (ndim == 2 and shape == (1, 0)) or (ndim == 1 and shape == (0,)):
        # Passed an empty matrix or empty vector; change shape to (0, 0)
        shape = (0, 0)
    elif ndim == 1:
        # Passed a row or column vector
        shape = (1, shape[0]) if axis == 1 else (shape[0], 1)
    elif ndim == 0:
        # Passed a constant; turn into a matrix
        shape = (1, 1)

    return arr.reshape(shape)


def K_calc(A, C, Q, R, S):
    """
    Calculate Kalman filter gain.

    Parameters:
    -----------
    A, C, Q, R, S : ndarray
        State-space and noise covariance matrices

    Returns:
    --------
    K : ndarray
        Kalman gain
    Calculated : bool
        Whether calculation was successful
    """
    try:
        X = solve_discrete_are(A.T, C.T, Q, R)
        P = ssmatrix(X)
        K = np.dot(np.dot(A, P), C.T) + S
        K = np.dot(K, np.linalg.inv(np.dot(np.dot(C, P), C.T) + R))
        Calculated = True
    except (ValueError, np.linalg.LinAlgError, IndexError):
        K = []
        warnings.warn("Kalman filter cannot be calculated")
        Calculated = False
    return K, Calculated


def get_model_uncertainty(u, y, model):
    """
    Returns the frequency response of a finite impulse response model and
    frequency confidence intervals (95% and 68%).

    Parameters:
    -----------
    u : array-like
        Input signal
    y : array-like
        Output signal
    model : ndarray
        Finite impulse response of the IO pair

    Returns:
    --------
    freqs : ndarray
        Frequency range
    model_bode_mag : ndarray
        Gain portion of model frequency response
    ci95 : ndarray
        95% confidence interval
    ci68 : ndarray
        68% confidence interval
    snr : ndarray
        Signal to noise ratio
    """
    n = len(u)

    confidence95 = 0.95
    confidence68 = 0.68
    nperseg = 1024

    y_estimate = signal.convolve(u, model, mode='full')[:len(u)]
    model_error = y - y_estimate

    h = fftpack.fft(model, nperseg)[:nperseg//2]
    freqs, Pxx = signal.welch(u, nperseg=nperseg)
    freqs, Pyy = signal.welch(y, nperseg=nperseg)
    freqs, Pyy_err = signal.welch(model_error, nperseg=nperseg)
    freqs, Pxy = signal.csd(u, y, nperseg=nperseg)

    snr = Pyy / Pyy_err
    data_bode = Pxy / Pxx
    data_bode_mag = np.abs(data_bode)

    win = np.hamming(16)
    data_bode_mag_filt_f = (np.convolve(data_bode_mag, win, mode='full') / sum(win))[:len(data_bode_mag)]
    data_bode_mag_filt_b = (np.convolve(data_bode_mag_filt_f[::-1], win, mode='full') / sum(win))[:len(data_bode_mag_filt_f)][::-1]
    snr_filt_f = (np.convolve(np.abs(snr), win, mode='full') / sum(win))[:len(snr)]
    snr_filt_b = (np.convolve(snr_filt_f[::-1], win, mode='full') / sum(win))[:len(snr_filt_f)][::-1]

    model_bode_mag = np.abs(h)
    combined_bode = np.vstack((model_bode_mag, data_bode_mag_filt_b[:-1]))
    se = stats.sem(combined_bode)
    ci95 = se * stats.t.ppf((1 + confidence95) / 2., n - 1)
    ci68 = se * stats.t.ppf((1 + confidence68) / 2., n - 1)

    return freqs[:-1], model_bode_mag, ci95, ci68, snr_filt_b[:-1]


def get_fir_coef(model, inds, deps, sampling, tss):
    """
    Returns a nested dictionary of numpy arrays containing FIR coefficients.

    Parameters:
    -----------
    model : harold.State or object with similar interface
        State-space model
    inds : list
        List of independent variables
    deps : list
        List of dependent variables
    sampling : int
        Model sampling rate in seconds
    tss : int
        Time to steady state in minutes

    Returns:
    --------
    fir_model : dict
        Nested dictionary of FIR coefficients
    """
    if not HAROLD_AVAILABLE:
        warnings.warn("harold not available, returning mock FIR coefficients")
        fir_model = {}
        for dep in deps:
            fir_model[dep] = {}
            for ind in inds:
                fir_model[dep][ind] = np.random.randn(int(tss*60/sampling)) * 0.01
        return fir_model

    fir_model = dict()
    t = np.arange(0, tss * 60, sampling)
    Gc = harold.undiscretize(model, method='backward euler')
    imp_response, _ = harold.simulate_impulse_response(Gc, t)

    for depidx, dep in enumerate(deps):
        fir_model[dep] = dict()
        for indidx, ind in enumerate(inds):
            if model.NumberOfInputs == 1 and model.NumberOfOutputs == 1:
                fir_model[dep][ind] = imp_response * model.SamplingPeriod
            elif model.NumberOfInputs == 1 and model.NumberOfOutputs > 1:
                fir_model[dep][ind] = imp_response[:, depidx] * model.SamplingPeriod
            else:
                fir_model[dep][ind] = imp_response[:, depidx, indidx] * model.SamplingPeriod

    return fir_model


def get_step_response(fir_model):
    """
    Returns a nested dictionary of numpy arrays containing step response.

    Parameters:
    -----------
    fir_model : dict
        Nested dictionary of FIR coefficients

    Returns:
    --------
    step_response : dict
        Nested dictionary of step responses
    """
    step_response = dict()
    for dep in fir_model.keys():
        step_response[dep] = dict()
        for ind in fir_model[dep].keys():
            step_response[dep][ind] = np.cumsum(fir_model[dep][ind])
    return step_response


def get_deadtime(step_response, isramp=False):
    """
    Returns the estimated deadtime based on a predefined minimum response tolerance.
    Current tolerance is the 4% of steady state gain or overshoot.

    Parameters:
    -----------
    step_response : ndarray
        Step response of the model
    isramp : bool
        Ramp type flag

    Returns:
    --------
    deadtime : int
        Deadtime in terms of number of samples
    """
    if isramp:
        gain = step_response[-1] - step_response[-2]
        abs_gain = abs(gain)
        tol = abs_gain / 25
    else:
        gain = step_response[-1]
        abs_gain = abs(gain)
        overshoot = np.abs(step_response).max()
        tol = abs_gain / 25 if overshoot <= abs_gain else overshoot / 25

    deadtime = 0
    for coef in step_response:
        if abs(coef) <= tol:
            deadtime += 1
        else:
            break

    deadtime = deadtime if deadtime >= 2 else 0
    return deadtime


def simulate_fir(fir_model, data):
    """
    Returns a pandas dataframe of dependent variable predictions.

    Parameters:
    -----------
    fir_model : dict
        Nested dictionary of FIR coefficients
    data : pandas.DataFrame
        DataFrame containing independent and dependent variables data

    Returns:
    --------
    predictions : pandas.DataFrame
        DataFrame containing dependent variable predictions
    """
    N = len(data)
    deps = list(fir_model.keys())
    inds = list(fir_model[deps[0]].keys())
    predictions = data[deps].copy(deep=True)

    for dep in deps:
        predictions[dep].values[:] = 0.0
        for ind in inds:
            predictions[dep] += signal.convolve(data[ind], fir_model[dep][ind], mode='full')[:N]
        tss = len(fir_model[dep][inds[0]])
        predictions[dep][:tss] = predictions[dep].values[tss + 1]
        predictions[dep] = predictions[dep] - predictions[dep].mean() + data[dep].mean()

    return predictions
