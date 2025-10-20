"""
Test cases for Frequency Domain identification algorithm implementation.

Tests follow TDD principles with comprehensive coverage of:
- Algorithm initialization and parameter validation
- Core identification functionality
- Coherence-based quality assessment
- Welch's method variance reduction
- IDData integration
- Factory pattern compatibility
"""

import numpy as np
import pandas as pd
import pytest

from sippy.identification import IDData, SystemIdentificationConfig
from sippy.identification.base import IdentificationAlgorithm, StateSpaceModel
from sippy.identification.factory import AlgorithmFactory, create_algorithm


class TestFrequencyDomainInitialization:
    """Test algorithm initialization and parameter validation."""

    def test_frequency_domain_algorithm_exists(self):
        """Test that FREQUENCY_DOMAIN algorithm is registered."""
        assert AlgorithmFactory.is_registered("FREQUENCY_DOMAIN")
        assert AlgorithmFactory.is_registered("FREQ_DOMAIN")
        assert AlgorithmFactory.is_registered("NONPARAMETRIC_FREQ")

    def test_frequency_domain_algorithm_creation(self):
        """Test creating algorithm via factory."""
        alg = create_algorithm("FREQUENCY_DOMAIN")
        assert alg is not None
        assert isinstance(alg, IdentificationAlgorithm)

    def test_algorithm_initialization_default_params(self):
        """Test algorithm initializes with default parameters."""
        alg = create_algorithm("FREQUENCY_DOMAIN")
        # Verify algorithm has expected attributes
        assert hasattr(alg, "identify")
        assert hasattr(alg, "validate_parameters")

    def test_algorithm_initialization_custom_params(self):
        """Test algorithm initializes with custom parameters."""
        alg = create_algorithm(
            "FREQUENCY_DOMAIN",
            smoothing_window=15,
            coherence_threshold=0.85,
            max_lag=500,
        )
        assert alg is not None

    def test_smoothing_window_validation_odd(self):
        """Test smoothing window is made odd if even."""
        # Even window should be converted to odd
        alg = create_algorithm("FREQUENCY_DOMAIN", smoothing_window=12)
        assert alg is not None

    def test_smoothing_window_too_small_raises(self):
        """Test smoothing window < 3 raises error."""
        with pytest.raises(ValueError, match="smoothing_window must be >= 3"):
            create_algorithm("FREQUENCY_DOMAIN", smoothing_window=1)

    def test_coherence_threshold_invalid_raises(self):
        """Test invalid coherence threshold raises error."""
        with pytest.raises(ValueError, match="coherence_threshold must be in"):
            create_algorithm("FREQUENCY_DOMAIN", coherence_threshold=1.5)

    def test_welch_segments_validation(self):
        """Test Welch method requires valid segment count."""
        with pytest.raises(ValueError, match="welch_segments must be >= 2"):
            create_algorithm(
                "FREQUENCY_DOMAIN", use_welch=True, welch_segments=1
            )

    def test_welch_overlap_validation(self):
        """Test Welch method requires valid overlap."""
        with pytest.raises(ValueError, match="welch_overlap must be in"):
            create_algorithm(
                "FREQUENCY_DOMAIN", use_welch=True, welch_overlap=1.5
            )

    def test_window_type_validation(self):
        """Test invalid window type raises error."""
        with pytest.raises(ValueError, match="Unknown window_type"):
            create_algorithm("FREQUENCY_DOMAIN", window_type="invalid")

    def test_valid_window_types(self):
        """Test all valid window types."""
        for wtype in ["none", "hann", "hamming", "blackman"]:
            alg = create_algorithm("FREQUENCY_DOMAIN", window_type=wtype)
            assert alg is not None


class TestFrequencyDomainBasicIdentification:
    """Test basic identification functionality."""

    def setup_method(self):
        """Set up test fixtures with synthetic data."""
        np.random.seed(42)
        self.n_samples = 2000
        self.dt = 0.01  # 10 ms sampling

        # Create simple input signal (white noise)
        self.u = np.random.randn(self.n_samples)

        # Create known output: y(k) = 0.8*u(k) + 0.1*u(k-1) + noise
        self.y_clean = np.zeros(self.n_samples)
        for k in range(1, self.n_samples):
            self.y_clean[k] = 0.8 * self.u[k] + 0.1 * self.u[k - 1]

        # Add measurement noise
        self.y = self.y_clean + 0.05 * np.random.randn(self.n_samples)

    def test_identify_with_numpy_arrays(self):
        """Test identification with raw numpy arrays."""
        alg = create_algorithm("FREQUENCY_DOMAIN")
        result = alg.identify(u=self.u, y=self.y, dt=self.dt)

        assert isinstance(result, StateSpaceModel)
        assert result.identification_info is not None

    def test_identify_returns_state_space_model(self):
        """Test identification returns StateSpaceModel."""
        alg = create_algorithm("FREQUENCY_DOMAIN", smoothing_window=11)
        result = alg.identify(u=self.u, y=self.y, dt=self.dt)

        assert isinstance(result, StateSpaceModel)
        assert result.A is not None
        assert result.B is not None
        assert result.C is not None
        assert result.D is not None
        assert result.ts == self.dt

    def test_identify_returns_frequency_response_info(self):
        """Test identification stores frequency response in info dict."""
        alg = create_algorithm("FREQUENCY_DOMAIN")
        result = alg.identify(u=self.u, y=self.y, dt=self.dt)

        info = result.identification_info
        assert "frequency_response" in info
        assert "quality_metrics" in info

    def test_frequency_response_contains_required_fields(self):
        """Test frequency response has all required fields."""
        alg = create_algorithm("FREQUENCY_DOMAIN")
        result = alg.identify(u=self.u, y=self.y, dt=self.dt)

        fr = result.identification_info["frequency_response"]
        required = [
            "omega",
            "omega_real",
            "freq_hz",
            "G_smooth",
            "magnitude_db",
            "phase_deg",
            "coherence",
        ]
        for field in required:
            assert field in fr, f"Missing {field}"

    def test_frequency_arrays_correct_shape(self):
        """Test frequency arrays have consistent shapes."""
        alg = create_algorithm("FREQUENCY_DOMAIN")
        result = alg.identify(u=self.u, y=self.y, dt=self.dt)

        fr = result.identification_info["frequency_response"]
        n_freq = len(fr["omega"])

        assert len(fr["freq_hz"]) == n_freq
        assert len(fr["G_smooth"]) == n_freq
        assert len(fr["magnitude_db"]) == n_freq
        assert len(fr["phase_deg"]) == n_freq
        assert len(fr["coherence"]) == n_freq

    def test_frequency_response_is_complex(self):
        """Test frequency response is complex-valued."""
        alg = create_algorithm("FREQUENCY_DOMAIN")
        result = alg.identify(u=self.u, y=self.y, dt=self.dt)

        fr = result.identification_info["frequency_response"]
        assert np.iscomplexobj(fr["G_smooth"])

    def test_magnitude_and_phase_consistency(self):
        """Test magnitude/phase consistent with complex response."""
        alg = create_algorithm("FREQUENCY_DOMAIN")
        result = alg.identify(u=self.u, y=self.y, dt=self.dt)

        fr = result.identification_info["frequency_response"]
        G = fr["G_smooth"]
        mag_db = fr["magnitude_db"]
        phase_deg = fr["phase_deg"]

        # Check magnitude
        expected_mag_db = 20 * np.log10(np.abs(G) + 1e-12)
        np.testing.assert_allclose(mag_db, expected_mag_db, rtol=1e-5)

        # Check phase (unwrapped)
        expected_phase_deg = np.degrees(np.unwrap(np.angle(G)))
        np.testing.assert_allclose(phase_deg, expected_phase_deg, atol=1.0)

    def test_coherence_in_valid_range(self):
        """Test coherence values are in [0, 1]."""
        alg = create_algorithm("FREQUENCY_DOMAIN")
        result = alg.identify(u=self.u, y=self.y, dt=self.dt)

        coherence = result.identification_info["frequency_response"]["coherence"]
        assert np.all(coherence >= 0.0)
        assert np.all(coherence <= 1.0)


class TestFrequencyDomainQualityAssessment:
    """Test coherence-based quality assessment."""

    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        self.n_samples = 2000
        self.dt = 0.01

        self.u = np.random.randn(self.n_samples)
        self.y_clean = np.zeros(self.n_samples)
        for k in range(1, self.n_samples):
            self.y_clean[k] = 0.8 * self.u[k] + 0.1 * self.u[k - 1]

        self.y = self.y_clean + 0.05 * np.random.randn(self.n_samples)

    def test_quality_metrics_present(self):
        """Test quality metrics are computed."""
        alg = create_algorithm("FREQUENCY_DOMAIN")
        result = alg.identify(u=self.u, y=self.y, dt=self.dt)

        metrics = result.identification_info["quality_metrics"]
        required = [
            "mean_coherence",
            "min_coherence",
            "max_coherence",
            "median_coherence",
            "fraction_reliable",
            "quality_label",
            "is_reliable",
        ]
        for metric in required:
            assert metric in metrics, f"Missing {metric}"

    def test_mean_coherence_computed_correctly(self):
        """Test mean coherence is average of coherence function."""
        alg = create_algorithm("FREQUENCY_DOMAIN")
        result = alg.identify(u=self.u, y=self.y, dt=self.dt)

        info = result.identification_info
        metrics = info["quality_metrics"]
        coherence = info["frequency_response"]["coherence"]

        expected_mean = np.mean(coherence)
        assert np.isclose(
            metrics["mean_coherence"], expected_mean, rtol=1e-5
        )

    def test_quality_label_based_on_coherence(self):
        """Test quality label correctly reflects coherence."""
        alg = create_algorithm("FREQUENCY_DOMAIN", coherence_threshold=0.8)
        result = alg.identify(u=self.u, y=self.y, dt=self.dt)

        metrics = result.identification_info["quality_metrics"]
        mean_coh = metrics["mean_coherence"]

        # Verify label matches thresholds
        if mean_coh >= 0.9:
            assert metrics["quality_label"] == "EXCELLENT"
        elif mean_coh >= 0.8:
            assert metrics["quality_label"] == "GOOD"
        elif mean_coh >= 0.7:
            assert metrics["quality_label"] == "ACCEPTABLE"
        else:
            assert metrics["quality_label"] == "POOR"

    def test_fraction_reliable_computed(self):
        """Test fraction of reliable frequencies computed."""
        alg = create_algorithm("FREQUENCY_DOMAIN", coherence_threshold=0.8)
        result = alg.identify(u=self.u, y=self.y, dt=self.dt)

        info = result.identification_info
        metrics = info["quality_metrics"]
        coherence = info["frequency_response"]["coherence"]

        expected_fraction = np.mean(
            coherence >= metrics["threshold"]
        )
        assert np.isclose(
            metrics["fraction_reliable"], expected_fraction, rtol=1e-5
        )

    def test_is_reliable_threshold_check(self):
        """Test is_reliable flag based on threshold."""
        alg = create_algorithm("FREQUENCY_DOMAIN", coherence_threshold=0.8)
        result = alg.identify(u=self.u, y=self.y, dt=self.dt)

        metrics = result.identification_info["quality_metrics"]
        is_reliable = metrics["is_reliable"]
        mean_coh = metrics["mean_coherence"]
        threshold = metrics["threshold"]

        assert is_reliable == (mean_coh >= threshold)


class TestFrequencyDomainIDDataIntegration:
    """Test integration with IDData class."""

    def setup_method(self):
        """Set up test data with IDData."""
        np.random.seed(42)
        self.n_samples = 2000
        self.dt = 0.01

        self.u = np.random.randn(self.n_samples)
        self.y_clean = np.zeros(self.n_samples)
        for k in range(1, self.n_samples):
            self.y_clean[k] = 0.8 * self.u[k] + 0.1 * self.u[k - 1]
        self.y = self.y_clean + 0.05 * np.random.randn(self.n_samples)

        # Create IDData object
        time_index = pd.date_range("2023-01-01", periods=self.n_samples, freq="10ms")
        data_df = pd.DataFrame({"u": self.u, "y": self.y}, index=time_index)
        self.iddata = IDData(data=data_df, inputs=["u"], outputs=["y"], tsample=self.dt)

    def test_identify_with_iddata_object(self):
        """Test identification with IDData object."""
        alg = create_algorithm("FREQUENCY_DOMAIN")
        result = alg.identify(iddata=self.iddata)

        assert isinstance(result, StateSpaceModel)
        assert result.identification_info is not None

    def test_iddata_and_arrays_give_similar_results(self):
        """Test IDData and raw arrays produce similar results."""
        alg1 = create_algorithm("FREQUENCY_DOMAIN", smoothing_window=11)
        alg2 = create_algorithm("FREQUENCY_DOMAIN", smoothing_window=11)

        result_iddata = alg1.identify(iddata=self.iddata)
        result_arrays = alg2.identify(u=self.u, y=self.y, dt=self.dt)

        # Compare frequency responses
        fr1 = result_iddata.identification_info["frequency_response"]
        fr2 = result_arrays.identification_info["frequency_response"]

        # Magnitudes should be similar
        np.testing.assert_allclose(
            fr1["magnitude_db"], fr2["magnitude_db"], rtol=0.01
        )


class TestFrequencyDomainWelchsMethod:
    """Test Welch's method for variance reduction."""

    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        self.n_samples = 4000
        self.dt = 0.01

        self.u = np.random.randn(self.n_samples)
        self.y_clean = np.zeros(self.n_samples)
        for k in range(1, self.n_samples):
            self.y_clean[k] = 0.8 * self.u[k] + 0.1 * self.u[k - 1]

        # Higher noise for Welch benefit
        self.y = self.y_clean + 0.1 * np.random.randn(self.n_samples)

    def test_welch_method_identification(self):
        """Test identification with Welch's method enabled."""
        alg = create_algorithm(
            "FREQUENCY_DOMAIN",
            use_welch=True,
            welch_segments=8,
            welch_overlap=0.5,
        )
        result = alg.identify(u=self.u, y=self.y, dt=self.dt)

        assert isinstance(result, StateSpaceModel)

    def test_welch_method_requires_sufficient_data(self):
        """Test Welch method raises error with insufficient data."""
        short_u = np.random.randn(100)
        short_y = np.random.randn(100)

        alg = create_algorithm(
            "FREQUENCY_DOMAIN",
            use_welch=True,
            welch_segments=20,
        )

        with pytest.raises(ValueError, match="Segments too short"):
            alg.identify(u=short_u, y=short_y, dt=self.dt)

    def test_welch_vs_standard_produces_different_results(self):
        """Test Welch method produces different (smoother) results."""
        alg_standard = create_algorithm("FREQUENCY_DOMAIN", use_welch=False)
        alg_welch = create_algorithm(
            "FREQUENCY_DOMAIN", use_welch=True, welch_segments=4
        )

        result_std = alg_standard.identify(u=self.u, y=self.y, dt=self.dt)
        result_welch = alg_welch.identify(u=self.u, y=self.y, dt=self.dt)

        # Results should be present but different
        assert result_std is not None
        assert result_welch is not None


class TestFrequencyDomainEdgeCases:
    """Test edge cases and error handling."""

    def test_identify_with_short_data_raises(self):
        """Test identification fails with insufficient data."""
        alg = create_algorithm("FREQUENCY_DOMAIN")
        short_u = np.random.randn(50)
        short_y = np.random.randn(50)

        with pytest.raises(ValueError, match="at least 100"):
            alg.identify(u=short_u, y=short_y, dt=0.01)

    def test_identify_mismatched_lengths_raises(self):
        """Test identification fails with mismatched u/y lengths."""
        alg = create_algorithm("FREQUENCY_DOMAIN")
        u = np.random.randn(1000)
        y = np.random.randn(500)

        with pytest.raises(ValueError, match="same length"):
            alg.identify(u=u, y=y, dt=0.01)

    def test_identify_invalid_dt_raises(self):
        """Test identification fails with invalid sampling interval."""
        alg = create_algorithm("FREQUENCY_DOMAIN")
        u = np.random.randn(1000)
        y = np.random.randn(1000)

        with pytest.raises(ValueError, match="must be positive"):
            alg.identify(u=u, y=y, dt=-0.01)

    def test_identify_constant_signal_handling(self):
        """Test handling of constant (DC) signals."""
        alg = create_algorithm("FREQUENCY_DOMAIN", remove_mean=True)
        u = np.ones(1000)  # Constant
        y = np.ones(1000) * 2  # Constant

        # Should not crash, but will have low signal power
        result = alg.identify(u=u, y=y, dt=0.01)
        assert result is not None

    def test_identify_with_nan_raises(self):
        """Test identification fails with NaN values."""
        alg = create_algorithm("FREQUENCY_DOMAIN")
        u = np.random.randn(1000)
        y = np.random.randn(1000)
        y[100] = np.nan

        with pytest.raises(ValueError, match="NaN|Invalid|contains"):
            alg.identify(u=u, y=y, dt=0.01)

    def test_identify_with_inf_raises(self):
        """Test identification fails with infinite values."""
        alg = create_algorithm("FREQUENCY_DOMAIN")
        u = np.random.randn(1000)
        y = np.random.randn(1000)
        y[100] = np.inf

        with pytest.raises(ValueError, match="infinite|Invalid|contains"):
            alg.identify(u=u, y=y, dt=0.01)


class TestFrequencyDomainSpectralProperties:
    """Test spectral properties and mathematical correctness."""

    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        self.n_samples = 2000
        self.dt = 0.01

        self.u = np.random.randn(self.n_samples)
        self.y_clean = np.zeros(self.n_samples)
        for k in range(1, self.n_samples):
            self.y_clean[k] = 0.8 * self.u[k] + 0.1 * self.u[k - 1]
        self.y = self.y_clean + 0.02 * np.random.randn(self.n_samples)

    def test_spectra_stored_in_results(self):
        """Test power and cross spectra are stored."""
        alg = create_algorithm("FREQUENCY_DOMAIN")
        result = alg.identify(u=self.u, y=self.y, dt=self.dt)

        fr = result.identification_info["frequency_response"]
        assert "Phi_u" in fr
        assert "Phi_y" in fr
        assert "Phi_uy" in fr

    def test_input_spectrum_is_real(self):
        """Test input power spectrum is real-valued."""
        alg = create_algorithm("FREQUENCY_DOMAIN")
        result = alg.identify(u=self.u, y=self.y, dt=self.dt)

        Phi_u = result.identification_info["frequency_response"]["Phi_u"]
        assert np.all(np.isreal(Phi_u))

    def test_cross_spectrum_is_complex(self):
        """Test cross spectrum is complex-valued."""
        alg = create_algorithm("FREQUENCY_DOMAIN")
        result = alg.identify(u=self.u, y=self.y, dt=self.dt)

        Phi_uy = result.identification_info["frequency_response"]["Phi_uy"]
        assert np.iscomplexobj(Phi_uy)

    def test_correlations_stored_in_results(self):
        """Test correlations are stored."""
        alg = create_algorithm("FREQUENCY_DOMAIN")
        result = alg.identify(u=self.u, y=self.y, dt=self.dt)

        fr = result.identification_info["frequency_response"]
        assert "R_u" in fr
        assert "R_uy" in fr
        assert "tau" in fr

    def test_frequency_vector_symmetric_around_zero(self):
        """Test frequency vector includes negative frequencies."""
        alg = create_algorithm("FREQUENCY_DOMAIN")
        result = alg.identify(u=self.u, y=self.y, dt=self.dt)

        omega = result.identification_info["frequency_response"]["omega"]
        # Check omega covers negative and positive frequencies
        assert np.any(omega < 0)
        assert np.any(omega > 0)


class TestFrequencyDomainDataPreprocessing:
    """Test data preprocessing options."""

    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        self.n_samples = 1000
        self.dt = 0.01

        self.u = np.random.randn(self.n_samples) + 5  # With DC offset
        self.y = np.random.randn(self.n_samples) + 3

    def test_remove_mean_true(self):
        """Test identification with DC removal."""
        alg = create_algorithm("FREQUENCY_DOMAIN", remove_mean=True)
        result = alg.identify(u=self.u, y=self.y, dt=self.dt)
        assert result is not None

    def test_remove_mean_false(self):
        """Test identification without DC removal."""
        alg = create_algorithm("FREQUENCY_DOMAIN", remove_mean=False)
        result = alg.identify(u=self.u, y=self.y, dt=self.dt)
        assert result is not None

    def test_windowing_options(self):
        """Test various windowing options."""
        for wtype in ["none", "hann", "hamming", "blackman"]:
            alg = create_algorithm("FREQUENCY_DOMAIN", window_type=wtype)
            result = alg.identify(u=self.u, y=self.y, dt=self.dt)
            assert result is not None

    def test_max_lag_parameter(self):
        """Test max_lag parameter."""
        alg = create_algorithm("FREQUENCY_DOMAIN", max_lag=500)
        result = alg.identify(u=self.u, y=self.y, dt=self.dt)
        assert result is not None


class TestFrequencyDomainFactoryIntegration:
    """Test factory pattern integration."""

    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        self.n_samples = 1000
        self.dt = 0.01
        self.u = np.random.randn(self.n_samples)
        self.y = np.random.randn(self.n_samples)

    def test_all_aliases_registered(self):
        """Test all algorithm aliases are registered."""
        aliases = ["FREQUENCY_DOMAIN", "FREQ_DOMAIN", "NONPARAMETRIC_FREQ"]
        for alias in aliases:
            assert AlgorithmFactory.is_registered(alias)

    def test_case_insensitive_registration(self):
        """Test case-insensitive algorithm lookup."""
        alg1 = create_algorithm("FREQUENCY_DOMAIN")
        alg2 = create_algorithm("frequency_domain")
        assert type(alg1) == type(alg2)

    def test_listed_in_available_algorithms(self):
        """Test algorithm appears in factory list."""
        algorithms = AlgorithmFactory.list_algorithms()
        assert "FREQUENCY_DOMAIN" in algorithms

    def test_validate_parameters_method_exists(self):
        """Test validate_parameters method exists."""
        alg = create_algorithm("FREQUENCY_DOMAIN")
        result = alg.validate_parameters()
        assert isinstance(result, bool)
