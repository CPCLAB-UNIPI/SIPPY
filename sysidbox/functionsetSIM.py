# -*- coding: utf-8 -*-
"""
Created on Sun Oct 08 2017

@author: Giuseppe Armenise
"""
from __future__ import absolute_import, print_function
from scipy.linalg import solve_discrete_are
from scipy import stats, signal, fftpack
import math
import harold
from .functionset import *
# from functionset import *



def ordinate_sequence(y, f, p):
    [l, L] = y.shape
    N = L - p - f + 1
    Yp = np.zeros((l * f, N))
    Yf = np.zeros((l * f, N))
    for i in range(1, f + 1):
        Yf[l * (i - 1):l * i] = y[:, p + i - 1:L - f + i]
        Yp[l * (i - 1):l * i] = y[:, i - 1:L - f - p + i]
    return Yf, Yp

def Z_dot_PIort(z, X):
    """
    Compute the scalar product between a vector z and $I - x^T \cdot pinv(X^T)$, avoiding the direct computation of the matrix
    
    PI = np.dot(X.T, np.linalg.pinv(X.T)), causing high memory usage
    
    
    Parameters
    ----------
    z : (...) vector array_like
    
    X : (...) matrix array_like
  
    """
        
    Z_dot_PIort = (z - np.dot(np.dot(z, X.T), np.linalg.pinv(X.T)))
    return Z_dot_PIort

def Vn_mat(y,yest):
    """
    Compute the variance of the model residuals
    
    Parameters
    ----------
    y : (L*l,1) vectorized matrix of output of the process
    
    yest : (L*l,1) vectorized matrix of output of the estimated model
  
    """ 
    y = y.flatten()    
    yest = yest.flatten()
    eps = y - yest
    Vn = (eps@eps)/(max(y.shape))   # @ is dot
    return Vn
    

def impile(M1, M2):
    M = np.zeros((M1[:, 0].size + M2[:, 0].size, M1[0, :].size))
    M[0:M1[:, 0].size] = M1
    M[M1[:, 0].size::] = M2
    return M


def reducingOrder(U_n, S_n, V_n, threshold=0.1, max_order=10):
    s0 = S_n[0]
    index = S_n.size
    for i in range(S_n.size):
        if S_n[i] < threshold * s0 or i >= max_order:
            index = i
            break
    return U_n[:, 0:index], S_n[0:index], V_n[0:index, :]


def check_types(threshold, max_order, fixed_order, f, p=20):
    if threshold < 0. or threshold >= 1.:
        print("Error! The threshold value must be >=0. and <1.")
        return False
    if (np.isnan(max_order)) == False:
        if type(max_order) != int:
            print("Error! The max_order value must be integer")
            return False
    if (np.isnan(fixed_order)) == False:
        if type(fixed_order) != int:
            print("Error! The fixed_order value must be integer")
            return False
    if type(f) != int:
        print("Error! The future horizon (f) must be integer")
        return False
    if type(p) != int:
        print("Error! The past horizon (p) must be integer")
        return False
    return True


def check_inputs(threshold, max_order, fixed_order, f):
    if (math.isnan(fixed_order)) == False:
        threshold = 0.0
        max_order = fixed_order
    if f < max_order:
        print('Warning! The horizon must be larger than the model order, max_order setted as f')
    if (max_order < f) == False:
        max_order = f
    return threshold, max_order


def SS_lsim_process_form(A, B, C, D, u, x0='None'):
    m, L = u.shape
    l, n = C.shape
    y = np.zeros((l, L))
    x = np.zeros((n, L))
    if type(x0) != str:
        x[:, 0] = x0[:, 0]
    y[:, 0] = np.dot(C, x[:, 0]) + np.dot(D, u[:, 0])
    for i in range(1, L):
        x[:, i] = np.dot(A, x[:, i - 1]) + np.dot(B, u[:, i - 1])
        y[:, i] = np.dot(C, x[:, i]) + np.dot(D, u[:, i])
    return x, y


def SS_lsim_predictor_form(A_K, B_K, C, D, K, y, u, x0='None'):
    m, L = u.shape
    l, n = C.shape
    y_hat = np.zeros((l, L))
    x = np.zeros((n, L + 1))
    if type(x0) != str:
        x[:, 0] = x0[:, 0]
    for i in range(0, L):
        x[:, i + 1] = np.dot(A_K, x[:, i]) + np.dot(B_K, u[:, i]) + np.dot(K, y[:, i])
        y_hat[:, i] = np.dot(C, x[:, i]) + np.dot(D, u[:, i])
    return x, y_hat


def SS_lsim_innovation_form(A, B, C, D, K, y, u, x0='None'):
    m, L = u.shape
    l, n = C.shape
    y_hat = np.zeros((l, L))
    x = np.zeros((n, L + 1))
    if type(x0) != str:
        x[:, 0] = x0[:, 0]
    for i in range(0, L):
        y_hat[:, i] = np.dot(C, x[:, i]) + np.dot(D, u[:, i])
        x[:, i + 1] = np.dot(A, x[:, i]) + np.dot(B, u[:, i]) + np.dot(K, y[:, i] - y_hat[:, i])
    return x, y_hat


def ssmatrix(data, axis=1):
    """Convert argument to a (possibly empty) state space matrix.

    Parameters
    ----------
    data : array, list, or string
        Input data defining the contents of the 2D array
    axis : 0 or 1
        If input data is 1D, which axis to use for return object.  The default
        is 1, corresponding to a row matrix.

    Returns
    -------
    arr : 2D array, with shape (0, 0) if a is empty

    """
  
    arr = np.array(data, dtype=float)
    ndim = arr.ndim
    shape = arr.shape

    # Change the shape of the array into a 2D array
    if (ndim > 2):
        raise ValueError("state-space matrix must be 2-dimensional")

    elif (ndim == 2 and shape == (1, 0)) or \
         (ndim == 1 and shape == (0, )):
        # Passed an empty matrix or empty vector; change shape to (0, 0)
        shape = (0, 0)

    elif ndim == 1:
        # Passed a row or column vector
        shape = (1, shape[0]) if axis == 1 else (shape[0], 1)

    elif ndim == 0:
        # Passed a constant; turn into a matrix
        shape = (1, 1)

    #  Create the actual object used to store the result
    return arr.reshape(shape)

def K_calc(A, C, Q, R, S):
    try:
        X = solve_discrete_are(A.T, C.T, Q, R)
        P = ssmatrix(X)
        K = np.dot(np.dot(A, P), C.T) + S
        K = np.dot(K, np.linalg.inv(np.dot(np.dot(C, P), C.T) + R))
        Calculated = True
    except:
        K = []
        print("Kalman filter cannot be calculated")
        Calculated = False
    return K, Calculated

def get_model_uncertainty(u, y,model):
    """
    Returns the frequency rsponse of a finite impulse response model and frequency confidance intervals (95 and 68).
        
        Parameters
        ----------
        u (pandas.Series or Numpy 1D array): Input siganal
        y (pandas.Series or Numpy 1D array): Output siganal
        model(Numpy 1D array): Finite impulse response of the IO pair

        Returns
        -------
        freqs (Numpy 1D array): frequency range
        model_bode_mag (Numpy 1D array): Gain portion of model frequency response
        model_bode_mag (Numpy 1D array): Gain portion of model frequency response
        ci95 (Numpy 1D array): 95% confidance interval
        ci68 (Numpy 1D array): 68% confidance interval
        snr (Numpy 1D array): signal to noise ratio
    """
    n = len(u)
    
    confidence95 = 0.95
    confidence68 = 0.68
    nperseg = 1024
    y_estimate = signal.convolve(u, model, mode='full')[:len(u)]
    model_error = y - y_estimate
    # Pxx = fftpack.fft(signal.fftconvolve(u, u[::-1], 'full')[-len(u)//2:], nperseg)[:nperseg//2]
    # Pyy = fftpack.fft(signal.fftconvolve(y, y[::-1], 'full')[-len(u)//2:], nperseg)[:nperseg//2]
    # Pxy = fftpack.fft(signal.fftconvolve(u, y[::-1], 'full')[-len(u)//2:], nperseg)[:nperseg//2]
    # Pyy_err = fftpack.fft(signal.fftconvolve(model_error, y[::-1], 'full')[-len(u)//2:], nperseg)[:nperseg//2]
    # freqs = fftpack.fftfreq(nperseg)[:nperseg//2]
    h = fftpack.fft(model, nperseg)[:nperseg//2]
    freqs, Pxx = signal.welch(u, nperseg=nperseg)
    freqs, Pyy = signal.welch(y, nperseg=nperseg)
    freqs, Pyy_err = signal.welch(model_error, nperseg=nperseg)
    freqs, Pxy = signal.csd(u, y, nperseg=nperseg)
    snr = Pyy / Pyy_err
    data_bode =  Pxy / Pxx
    data_bode_mag = np.abs(data_bode)
    win = np.hamming(16)
    data_bode_mag_filt_f = (np.convolve(data_bode_mag, win, mode='full') / sum(win))[:len(data_bode_mag)]
    data_bode_mag_filt_b = (np.convolve(data_bode_mag_filt_f[::-1], win, mode='full') / sum(win))[:len(data_bode_mag_filt_f)][::-1]
    snr_filt_f = (np.convolve(np.abs(snr), win, mode='full') / sum(win))[:len(snr)]
    snr_filt_b = (np.convolve(snr_filt_f[::-1], win, mode='full') / sum(win))[:len(snr_filt_f)][::-1]
    model_bode_mag = np.abs(h)
    combined_bode = np.vstack((model_bode_mag, data_bode_mag_filt_b[:-1]))
    se = stats.sem(combined_bode)
    ci95 = se * stats.t.ppf((1 + confidence95) / 2., n-1)
    ci68 = se * stats.t.ppf((1 + confidence68) / 2., n-1)
    return freqs[:-1], model_bode_mag, ci95, ci68, snr_filt_b[:-1]


def get_deadtime(step_response, isramp=False):
    """
    Returns the estimated deadtime based on a predifined minimum response tolerance.
    Current tollarance is the 4% of steady sate gain or overshoot.
        
        Parameters
        ----------
        step (Numpy 1D array): Step response of the model.
        isramp (bool): Ramp type flag.

        Returns
        -------
        deadtime (int): deadtime in terms of number of samples.
    """
    if isramp:
        gain = step_response[-1] - step_response[-2]
        abs_gain = abs(gain)
        tol = abs_gain/25
    else:
        gain = step_response[-1]
        abs_gain = abs(gain)
        overshoot  = np.abs(step_response).max()
        tol = abs_gain/25 if  overshoot <= abs_gain else overshoot/25
    deadtime = 0
    for coef in step_response:
        if abs(coef)<= tol:
            deadtime += 1
        else:
            break
    deadtime = deadtime if deadtime >= 2 else 0
    return deadtime
def get_fir_coef(model, inds, deps, sampling, tss):
    """
    Returns a nested dictionary of numpy ayyay containig FIR coeficiants.
        
        Parameters
        ----------
        model (harold.State): Statespace model.
        inds (list): List of independant variables.
        deps (list): List of dependant variables.
        sampling (int): Model sampling rate in seconds.
        tss (int): Time to steady steate in minutes.
        Returns
        -------
        fir_model (dict(dict(numpy.array))): nested dictionary of numpy ayyay containig FIR coeficiants.
    """
    fir_model = dict()
    t = np.arange(0, tss*60, sampling)
    Gc = harold.undiscretize(model)
    Gd = harold.discretize(G=Gc, dt=sampling, method='backward euler')
    imp_response, _ = harold.simulate_impulse_response(Gd, t)
    for depidx, dep in enumerate(deps):
        fir_model[dep] = dict()
        for indidx, ind in enumerate(inds):
            if model.NumberOfInputs == 1 and model.NumberOfOutputs == 1:
                fir_model[dep][ind] = imp_response * model.SamplingPeriod
            elif  model.NumberOfInputs == 1 and model.NumberOfOutputs > 1:
                fir_model[dep][ind] = imp_response[:,depidx] * model.SamplingPeriod
            else:
                fir_model[dep][ind] = imp_response[:,depidx,indidx] * model.SamplingPeriod
    return fir_model

def get_step_response(fir_model):
    """
    Returns a nested dictionary of numpy array containig stap responce.
        
        Parameters
        ----------
        fir_model (dict(dict(numpy.array))): nested dictionary of numpy ayyay containig FIR coeficiants

        Returns
        -------
        step_response (dict(dict(numpy.array))): nested dictionary of numpy ayyay containig stap responce.
    """
    step_response = dict()
    for dep in fir_model.keys():
        step_response[dep] = dict()
        for ind in fir_model[dep].keys():
            step_response[dep][ind] = np.cumsum(fir_model[dep][ind])
    return step_response