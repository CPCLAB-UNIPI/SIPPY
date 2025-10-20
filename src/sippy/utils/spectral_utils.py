"""
Shared spectral analysis utilities for frequency domain system identification.

This module provides unified interfaces for spectral analysis operations used across
multiple identification algorithms and analysis functions. Centralized here to avoid
code duplication and ensure consistency.

Functions include:
- Power spectrum and cross-spectrum computation (Welch's method or FFT-based correlations)
- Frequency response estimation from spectra
- Coherence computation
- Spectral smoothing via windowing
- Data windowing functions
"""

from typing import Optional, Tuple
import warnings
import numpy as np
from scipy import signal as scipy_signal, fftpack
from scipy.fft import fft, ifft, fftfreq

# Import compiled utilities for performance
try:
    from .compiled_utils import (
        NUMBA_AVAILABLE,
    )
except ImportError:
    NUMBA_AVAILABLE = False


def compute_power_spectrum_welch(
    x: np.ndarray, 
    dt: float = 1.0, 
    nperseg: int = 1024,
    window: str = "hann"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute power spectral density using Welch's method.

    Welch's method divides data into segments, computes periodogram for each,
    and averages to reduce variance.

    Parameters:
    -----------
    x : ndarray
        Input signal (1D array)
    dt : float
        Sampling interval (seconds)
    nperseg : int
        Length of each segment for Welch's method
    window : str
        Window function ('hann', 'hamming', 'blackman', etc.)

    Returns:
    --------
    freqs : ndarray
        Frequency array (Hz)
    Pxx : ndarray
        Power spectral density
    """
    freqs, Pxx = scipy_signal.welch(x, fs=1/dt, nperseg=nperseg, window=window)
    return freqs, Pxx


def compute_cross_spectrum_welch(
    u: np.ndarray,
    y: np.ndarray,
    dt: float = 1.0,
    nperseg: int = 1024,
    window: str = "hann"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute cross-spectral density using Welch's method.

    Parameters:
    -----------
    u : ndarray
        Input signal
    y : ndarray
        Output signal
    dt : float
        Sampling interval (seconds)
    nperseg : int
        Length of each segment for Welch's method
    window : str
        Window function

    Returns:
    --------
    freqs : ndarray
        Frequency array (Hz)
    Pxy : ndarray
        Complex cross-spectral density
    """
    freqs, Pxy = scipy_signal.csd(u, y, fs=1/dt, nperseg=nperseg, window=window)
    return freqs, Pxy


def compute_correlations_fft(
    u: np.ndarray, y: np.ndarray, max_lag: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute input autocorrelation and input-output cross-correlation using FFT.

    FFT-based method is efficient (O(N log N)) and avoids circular artifacts
    through zero-padding.

    Parameters:
    -----------
    u : ndarray
        Input signal
    y : ndarray
        Output signal
    max_lag : int, optional
        Maximum lag to compute (default: len(u)-1)

    Returns:
    --------
    tau : ndarray
        Lag vector [-max_lag, ..., 0, ..., +max_lag]
    R_u : ndarray
        Input autocorrelation
    R_uy : ndarray
        Input-output cross-correlation
    """
    N = len(u)
    if max_lag is None:
        max_lag = N - 1
    max_lag = min(max_lag, N - 1)

    # Zero-pad to avoid circular correlation artifacts
    u_fft = fft(u, n=2 * N)
    y_fft = fft(y, n=2 * N)

    # Autocorrelation: R_u(τ) = IFFT(|FFT(u)|²)
    R_u_full = np.real(ifft(u_fft * np.conj(u_fft)))

    # Cross-correlation: R_uy(τ) = IFFT(FFT(u) * conj(FFT(y)))
    R_uy_full = np.real(ifft(u_fft * np.conj(y_fft)))

    # Normalize
    R_u_full = R_u_full / N
    R_uy_full = R_uy_full / N

    # Create symmetric lag vector
    tau = np.arange(-max_lag, max_lag + 1)

    # Extract correlation values for specified lags
    R_u = np.concatenate([R_u_full[-(max_lag):], R_u_full[:max_lag + 1]])
    R_uy = np.concatenate([R_uy_full[-(max_lag):], R_uy_full[:max_lag + 1]])

    return tau, R_u, R_uy


def compute_spectra_from_correlation(
    R_u: np.ndarray, R_uy: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute power and cross spectra from correlations using DTFT.

    Theory:
        Φ_u(ω) = Σ R_u(τ)exp(-iωτ)     [Real, even function]
        Φ_uy(ω) = Σ R_uy(τ)exp(-iωτ)   [Complex]

    Parameters:
    -----------
    R_u : ndarray
        Input autocorrelation
    R_uy : ndarray
        Input-output cross-correlation

    Returns:
    --------
    Phi_u : ndarray
        Input power spectrum (real)
    Phi_uy : ndarray
        Cross spectrum (complex)
    omega : ndarray
        Normalized frequency vector [-π, π]
    """
    N_fft = len(R_u)

    # Apply FFT (implements DTFT on sampled correlations)
    Phi_u_raw = fft(R_u, n=N_fft)
    Phi_uy_raw = fft(R_uy, n=N_fft)

    # Frequency vector: ω ∈ [-π, π]
    omega = fftfreq(N_fft, d=1.0) * 2 * np.pi

    # Sort to ascending frequency order
    idx = np.argsort(omega)
    omega = omega[idx]
    Phi_u = np.real(Phi_u_raw[idx])  # Should be real (autocorr is even)
    Phi_uy = Phi_uy_raw[idx]  # Complex

    return Phi_u, Phi_uy, omega


def compute_output_spectrum(
    y: np.ndarray, n_freq: Optional[int] = None
) -> np.ndarray:
    """
    Compute output power spectrum directly from data.

    Parameters:
    -----------
    y : ndarray
        Output signal
    n_freq : int, optional
        Target frequency resolution (if None, uses signal length)

    Returns:
    --------
    Phi_y : ndarray
        Power spectrum with length n_freq
    """
    N = len(y)
    if n_freq is None:
        n_freq = N

    # Compute spectrum using same FFT size as input spectrum
    y_fft = fft(y, n=n_freq)
    Phi_y = np.real(np.abs(y_fft) ** 2 / N)
    return Phi_y


def compute_frequency_response(
    cross_spectrum: np.ndarray, input_spectrum: np.ndarray
) -> np.ndarray:
    """
    Estimate frequency response from spectra.

    Theory:
        G(e^iω) = Φ_uy(ω) / Φ_u(ω)

    This follows from the Wiener-Hopf equation relating input-output
    cross-correlation to frequency response.

    Parameters:
    -----------
    cross_spectrum : ndarray
        Input-output cross spectrum (complex or real)
    input_spectrum : ndarray
        Input power spectrum (real)

    Returns:
    --------
    G : ndarray
        Complex frequency response
    """
    # Avoid division by zero
    epsilon = 1e-12 * np.max(np.abs(input_spectrum))
    input_spectrum_safe = input_spectrum + epsilon

    # Element-wise complex division
    G = cross_spectrum / input_spectrum_safe

    return G


def compute_coherence(
    cross_spectrum: np.ndarray,
    input_spectrum: np.ndarray,
    output_spectrum: np.ndarray,
) -> np.ndarray:
    """
    Compute coherence function for model validation.

    Theory:
        γ²(ω) = |Φ_uy(ω)|² / (Φ_u(ω) · Φ_y(ω))

    Range: γ²(ω) ∈ [0, 1]

    Interpretation:
        - γ² = 1: Perfect linear relationship
        - γ² < 1: Presence of noise, nonlinearity, or unmeasured disturbances

    Parameters:
    -----------
    cross_spectrum : ndarray
        Input-output cross spectrum (complex)
    input_spectrum : ndarray
        Input power spectrum (real)
    output_spectrum : ndarray
        Output power spectrum (real)

    Returns:
    --------
    coherence : ndarray
        Coherence function γ²(ω) ∈ [0, 1]
    """
    # Match lengths
    min_len = min(len(cross_spectrum), len(input_spectrum), len(output_spectrum))
    cross_spectrum = cross_spectrum[:min_len]
    input_spectrum = input_spectrum[:min_len]
    output_spectrum = output_spectrum[:min_len]

    # Compute coherence
    numerator = np.abs(cross_spectrum) ** 2
    denominator = input_spectrum * output_spectrum

    # Avoid division by zero
    epsilon = 1e-12 * np.max(denominator)
    coherence = numerator / (denominator + epsilon)

    # Clamp to valid range [0, 1]
    coherence = np.clip(coherence, 0.0, 1.0)

    return coherence


def create_window(N: int, window_type: str) -> np.ndarray:
    """
    Create tapering window to reduce FFT leakage.

    Parameters:
    -----------
    N : int
        Window length
    window_type : str
        Window type ('hann', 'hamming', 'blackman', 'none')

    Returns:
    --------
    window : ndarray
        Window function
    """
    if window_type == "hann":
        return np.hanning(N)
    elif window_type == "hamming":
        return np.hamming(N)
    elif window_type == "blackman":
        return np.blackman(N)
    elif window_type == "none":
        return np.ones(N)
    else:
        raise ValueError(f"Unknown window_type: {window_type}")


def create_hamming_window(window_size: int, normalize: bool = True) -> np.ndarray:
    """
    Create normalized Hamming window for spectral smoothing.

    The Hamming window provides good frequency selectivity while
    minimizing sidelobe levels.

    Parameters:
    -----------
    window_size : int
        Size of window (will be made odd for symmetry)
    normalize : bool
        Whether to normalize window to sum to 1

    Returns:
    --------
    window : ndarray
        Normalized Hamming window
    """
    if window_size % 2 == 0:
        window_size += 1  # Ensure odd for symmetry

    window = np.hamming(window_size)
    if normalize:
        window = window / np.sum(window)

    return window


def smooth_frequency_response(
    G: np.ndarray, window: np.ndarray
) -> np.ndarray:
    """
    Apply spectral smoothing using provided window function.

    Reduces variance of frequency response estimate at cost of slight bias.
    Smooths magnitude and phase separately to preserve continuity.

    Parameters:
    -----------
    G : ndarray
        Complex frequency response
    window : ndarray
        Normalized window for smoothing

    Returns:
    --------
    G_smooth : ndarray
        Smoothed complex frequency response
    """
    # Smooth magnitude and phase separately
    magnitude = np.abs(G)
    phase = np.angle(G)

    # Apply convolution-based smoothing
    magnitude_smooth = np.convolve(magnitude, window, mode="same")
    phase_smooth = np.convolve(phase, window, mode="same")

    # Reconstruct complex frequency response
    G_smooth = magnitude_smooth * np.exp(1j * phase_smooth)

    return G_smooth


def extract_magnitude_phase(G: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract magnitude (dB) and phase (degrees) from complex frequency response.

    Parameters:
    -----------
    G : ndarray
        Complex frequency response

    Returns:
    --------
    magnitude_db : ndarray
        Magnitude in dB = 20*log10|G|
    phase_deg : ndarray
        Unwrapped phase in degrees
    """
    magnitude = np.abs(G)
    magnitude_db = 20 * np.log10(magnitude + 1e-12)  # Avoid log(0)

    phase_rad = np.angle(G)
    phase_deg = np.degrees(np.unwrap(phase_rad))  # Unwrap for continuity

    return magnitude_db, phase_deg


def validate_signal_pair(
    u: np.ndarray, y: np.ndarray, min_length: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Validate and preprocess input-output signal pair.

    Parameters:
    -----------
    u : ndarray
        Input signal
    y : ndarray
        Output signal
    min_length : int
        Minimum acceptable signal length

    Returns:
    --------
    u_valid : ndarray
        Validated input signal
    y_valid : ndarray
        Validated output signal

    Raises:
    -------
    ValueError
        If signals are invalid
    """
    # Convert to numpy arrays
    u = np.asarray(u, dtype=float).flatten()
    y = np.asarray(y, dtype=float).flatten()

    # Check for NaN and inf
    if np.any(np.isnan(u)) or np.any(np.isnan(y)):
        raise ValueError("Input/output contains NaN values")
    if np.any(np.isinf(u)) or np.any(np.isinf(y)):
        raise ValueError("Input/output contains infinite values")

    # Check dimensions
    if len(u) != len(y):
        raise ValueError(
            f"Input and output must have same length: {len(u)} != {len(y)}"
        )

    if len(u) < min_length:
        raise ValueError(
            f"Need at least {min_length} samples for reliable identification, got {len(u)}"
        )

    return u, y


def denormalize_frequency(
    omega: np.ndarray, dt: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert normalized frequency to physical units.

    Parameters:
    -----------
    omega : ndarray
        Normalized frequency (rad/sample) in range [-π, π]
    dt : float
        Sampling interval (seconds)

    Returns:
    --------
    omega_rad : ndarray
        Angular frequency (rad/s)
    freq_hz : ndarray
        Frequency (Hz)
    """
    omega_rad = omega / dt  # Convert to rad/s
    freq_hz = omega_rad / (2 * np.pi)  # Convert to Hz

    return omega_rad, freq_hz
