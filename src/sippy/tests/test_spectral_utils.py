"""
Test cases for shared spectral analysis utilities.

Tests the spectral_utils module which provides unified interfaces for spectral
analysis operations used across multiple identification algorithms.
"""

import numpy as np
import pytest
from sippy.utils.spectral_utils import (
    compute_power_spectrum_welch,
    compute_cross_spectrum_welch,
    compute_correlations_fft,
    compute_spectra_from_correlation,
    compute_output_spectrum,
    compute_frequency_response,
    compute_coherence,
    create_window,
    create_hamming_window,
    smooth_frequency_response,
    extract_magnitude_phase,
    validate_signal_pair,
    denormalize_frequency,
)


class TestPowerSpectrumComputation:
    """Test power spectrum computation functions."""

    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.x = np.random.randn(1000)
        self.dt = 0.01

    def test_compute_power_spectrum_welch_returns_arrays(self):
        """Test Welch's method returns frequency and power arrays."""
        freqs, Pxx = compute_power_spectrum_welch(self.x, dt=self.dt)
        
        assert isinstance(freqs, np.ndarray)
        assert isinstance(Pxx, np.ndarray)
        assert len(freqs) == len(Pxx)

    def test_compute_power_spectrum_welch_positive_power(self):
        """Test power spectrum values are non-negative."""
        freqs, Pxx = compute_power_spectrum_welch(self.x, dt=self.dt)
        assert np.all(Pxx >= 0)

    def test_compute_power_spectrum_welch_frequency_range(self):
        """Test frequency array has expected range."""
        freqs, Pxx = compute_power_spectrum_welch(self.x, dt=self.dt, nperseg=1024)
        
        # Frequencies should be positive and up to Nyquist
        assert np.all(freqs >= 0)
        assert np.max(freqs) <= 1 / (2 * self.dt)


class TestCrossSpectrumComputation:
    """Test cross spectrum computation."""

    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.u = np.random.randn(1000)
        self.y = self.u + 0.1 * np.random.randn(1000)
        self.dt = 0.01

    def test_compute_cross_spectrum_welch_returns_arrays(self):
        """Test cross spectrum computation returns arrays."""
        freqs, Pxy = compute_cross_spectrum_welch(self.u, self.y, dt=self.dt)
        
        assert isinstance(freqs, np.ndarray)
        assert isinstance(Pxy, np.ndarray)
        assert len(freqs) == len(Pxy)

    def test_compute_cross_spectrum_welch_complex(self):
        """Test cross spectrum is complex-valued."""
        freqs, Pxy = compute_cross_spectrum_welch(self.u, self.y, dt=self.dt)
        assert np.iscomplexobj(Pxy)


class TestCorrelationComputation:
    """Test correlation-based spectral computation."""

    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.u = np.random.randn(1000)
        self.y = self.u + 0.1 * np.random.randn(1000)

    def test_compute_correlations_fft_returns_tuple(self):
        """Test correlation computation returns expected tuple."""
        tau, R_u, R_uy = compute_correlations_fft(self.u, self.y, max_lag=100)
        
        assert isinstance(tau, np.ndarray)
        assert isinstance(R_u, np.ndarray)
        assert isinstance(R_uy, np.ndarray)
        assert len(tau) == len(R_u) == len(R_uy)

    def test_compute_correlations_fft_autocorr_even(self):
        """Test autocorrelation is even symmetric."""
        tau, R_u, R_uy = compute_correlations_fft(self.u, self.y, max_lag=100)
        
        # R_u should be symmetric around lag 0
        n_lag = len(tau) // 2
        np.testing.assert_allclose(R_u[:n_lag], R_u[-1:-n_lag-1:-1], rtol=1e-10)

    def test_compute_correlations_fft_default_max_lag(self):
        """Test default max_lag uses full length."""
        N = len(self.u)
        tau, R_u, R_uy = compute_correlations_fft(self.u, self.y, max_lag=None)
        
        # Should have 2*(N-1)+1 elements
        assert len(tau) <= 2 * N - 1


class TestSpectralEstimation:
    """Test spectral density estimation from correlations."""

    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.u = np.random.randn(1000)
        self.y = self.u + 0.1 * np.random.randn(1000)
        tau, self.R_u, self.R_uy = compute_correlations_fft(self.u, self.y, max_lag=100)

    def test_compute_spectra_from_correlation_returns_tuple(self):
        """Test spectral computation returns expected tuple."""
        Phi_u, Phi_uy, omega = compute_spectra_from_correlation(self.R_u, self.R_uy)
        
        assert isinstance(Phi_u, np.ndarray)
        assert isinstance(Phi_uy, np.ndarray)
        assert isinstance(omega, np.ndarray)
        assert len(Phi_u) == len(Phi_uy) == len(omega)

    def test_compute_spectra_from_correlation_phi_u_real(self):
        """Test input power spectrum is real."""
        Phi_u, Phi_uy, omega = compute_spectra_from_correlation(self.R_u, self.R_uy)
        
        assert np.all(np.isreal(Phi_u))

    def test_compute_spectra_from_correlation_phi_uy_complex(self):
        """Test cross spectrum is complex."""
        Phi_u, Phi_uy, omega = compute_spectra_from_correlation(self.R_u, self.R_uy)
        
        assert np.iscomplexobj(Phi_uy)

    def test_compute_output_spectrum_returns_array(self):
        """Test output spectrum computation."""
        Phi_y = compute_output_spectrum(self.y, n_freq=100)
        
        assert isinstance(Phi_y, np.ndarray)
        assert len(Phi_y) == 100


class TestFrequencyResponseEstimation:
    """Test frequency response estimation."""

    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.u = np.random.randn(1000)
        self.y = self.u + 0.1 * np.random.randn(1000)
        
        # Get spectra
        tau, R_u, R_uy = compute_correlations_fft(self.u, self.y, max_lag=100)
        self.Phi_u, self.Phi_uy, self.omega = compute_spectra_from_correlation(R_u, R_uy)

    def test_compute_frequency_response_returns_array(self):
        """Test frequency response is computed."""
        G = compute_frequency_response(self.Phi_uy, self.Phi_u)
        
        assert isinstance(G, np.ndarray)
        assert np.iscomplexobj(G)
        assert len(G) == len(self.Phi_u)

    def test_compute_frequency_response_no_nans(self):
        """Test no NaN values in frequency response."""
        G = compute_frequency_response(self.Phi_uy, self.Phi_u)
        
        # Should have valid values (may be inf at zero power)
        assert not np.all(np.isnan(G))


class TestCoherenceComputation:
    """Test coherence function computation."""

    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.u = np.random.randn(1000)
        # Create output with high coherence to input
        self.y = 0.9 * self.u + 0.1 * np.random.randn(1000)
        
        tau, R_u, R_uy = compute_correlations_fft(self.u, self.y, max_lag=100)
        self.Phi_u, self.Phi_uy, self.omega = compute_spectra_from_correlation(R_u, R_uy)
        self.Phi_y = compute_output_spectrum(self.y, n_freq=len(self.Phi_u))

    def test_compute_coherence_range(self):
        """Test coherence values in [0, 1]."""
        coh = compute_coherence(self.Phi_uy, self.Phi_u, self.Phi_y)
        
        assert np.all(coh >= 0.0)
        assert np.all(coh <= 1.0)

    def test_compute_coherence_has_high_values(self):
        """Test coherence achieves high values for related signals."""
        coh = compute_coherence(self.Phi_uy, self.Phi_u, self.Phi_y)
        
        # For related signals, max coherence should be high
        assert np.max(coh) > 0.8


class TestWindowFunctions:
    """Test window function creation."""

    def test_create_window_types(self):
        """Test all window types."""
        for wtype in ["hann", "hamming", "blackman", "none"]:
            w = create_window(100, wtype)
            
            assert isinstance(w, np.ndarray)
            assert len(w) == 100

    def test_create_hamming_window_normalized(self):
        """Test Hamming window is normalized."""
        w = create_hamming_window(11, normalize=True)
        
        # Should sum to 1
        assert np.isclose(np.sum(w), 1.0)

    def test_create_hamming_window_odd(self):
        """Test Hamming window is made odd."""
        w = create_hamming_window(12, normalize=True)
        
        # Should be odd size
        assert len(w) % 2 == 1


class TestSpectralSmoothing:
    """Test frequency response smoothing."""

    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.u = np.random.randn(1000)
        self.y = self.u + 0.1 * np.random.randn(1000)
        
        tau, R_u, R_uy = compute_correlations_fft(self.u, self.y, max_lag=100)
        Phi_u, Phi_uy, self.omega = compute_spectra_from_correlation(R_u, R_uy)
        self.G = compute_frequency_response(Phi_uy, Phi_u)

    def test_smooth_frequency_response_returns_array(self):
        """Test smoothing returns complex array."""
        window = create_hamming_window(11)
        G_smooth = smooth_frequency_response(self.G, window)
        
        assert isinstance(G_smooth, np.ndarray)
        assert np.iscomplexobj(G_smooth)
        assert len(G_smooth) == len(self.G)


class TestMagnitudePhaseExtraction:
    """Test magnitude and phase extraction."""

    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.u = np.random.randn(1000)
        self.y = self.u + 0.1 * np.random.randn(1000)
        
        tau, R_u, R_uy = compute_correlations_fft(self.u, self.y, max_lag=100)
        Phi_u, Phi_uy, self.omega = compute_spectra_from_correlation(R_u, R_uy)
        self.G = compute_frequency_response(Phi_uy, Phi_u)

    def test_extract_magnitude_phase_returns_tuple(self):
        """Test extraction returns magnitude and phase."""
        mag_db, phase_deg = extract_magnitude_phase(self.G)
        
        assert isinstance(mag_db, np.ndarray)
        assert isinstance(phase_deg, np.ndarray)
        assert len(mag_db) == len(phase_deg) == len(self.G)

    def test_magnitude_consistency(self):
        """Test magnitude extraction is consistent."""
        mag_db, phase_deg = extract_magnitude_phase(self.G)
        
        # Verify magnitude calculation
        expected_mag_db = 20 * np.log10(np.abs(self.G) + 1e-12)
        np.testing.assert_allclose(mag_db, expected_mag_db, rtol=1e-5)


class TestSignalValidation:
    """Test signal validation utilities."""

    def setup_method(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.u = np.random.randn(1000)
        self.y = np.random.randn(1000)

    def test_validate_signal_pair_valid(self):
        """Test validation passes for valid signals."""
        u_valid, y_valid = validate_signal_pair(self.u, self.y)
        
        assert np.array_equal(u_valid, self.u.flatten())
        assert np.array_equal(y_valid, self.y.flatten())

    def test_validate_signal_pair_mismatched_length_raises(self):
        """Test validation fails for mismatched lengths."""
        y_short = self.y[:500]
        
        with pytest.raises(ValueError, match="same length"):
            validate_signal_pair(self.u, y_short)

    def test_validate_signal_pair_short_data_raises(self):
        """Test validation fails for short data."""
        short_u = np.random.randn(50)
        short_y = np.random.randn(50)
        
        with pytest.raises(ValueError, match="at least"):
            validate_signal_pair(short_u, short_y, min_length=100)

    def test_validate_signal_pair_nan_raises(self):
        """Test validation fails for NaN values."""
        u_nan = self.u.copy()
        u_nan[0] = np.nan
        
        with pytest.raises(ValueError, match="NaN"):
            validate_signal_pair(u_nan, self.y)

    def test_validate_signal_pair_inf_raises(self):
        """Test validation fails for infinite values."""
        y_inf = self.y.copy()
        y_inf[0] = np.inf
        
        with pytest.raises(ValueError, match="infinite"):
            validate_signal_pair(self.u, y_inf)


class TestFrequencyDenormalization:
    """Test frequency denormalization."""

    def test_denormalize_frequency_returns_tuple(self):
        """Test denormalization returns rad/s and Hz."""
        omega = np.linspace(-np.pi, np.pi, 100)
        dt = 0.01
        
        omega_rad, freq_hz = denormalize_frequency(omega, dt)
        
        assert isinstance(omega_rad, np.ndarray)
        assert isinstance(freq_hz, np.ndarray)
        assert len(omega_rad) == len(freq_hz) == len(omega)

    def test_denormalize_frequency_nyquist(self):
        """Test denormalization at Nyquist frequency."""
        omega = np.array([np.pi])
        dt = 0.01
        
        omega_rad, freq_hz = denormalize_frequency(omega, dt)
        
        # At Nyquist, f should be fs/2 = 1/(2*dt) = 50 Hz
        assert np.isclose(freq_hz[0], 50.0)
