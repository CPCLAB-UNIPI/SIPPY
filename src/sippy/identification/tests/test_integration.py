"""
Integration tests for the complete system identification workflow.
"""

import numpy as np
import pytest

from sippy.identification import (
    SystemIdentification,
    SystemIdentificationConfig,
    system_identification,
)


class TestSystemIdentification:
    """Test the main SystemIdentification class."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample test data."""
        np.random.seed(42)
        n_points = 200
        u = np.random.randn(2, n_points)  # 2 inputs
        y = np.zeros((1, n_points))
        # Simple system: y[k] = 0.9*y[k-1] + 0.5*u1[k-1] + 0.3*u2[k-1] + noise
        for i in range(1, n_points):
            y[0, i] = (
                0.9 * y[0, i - 1]
                + 0.5 * u[0, i - 1]
                + 0.3 * u[1, i - 1]
                + 0.05 * np.random.randn()
            )
        return y, u

    def test_default_identification(self, sample_data):
        """Test identification with default configuration."""
        y, u = sample_data
        identifier = SystemIdentification()
        model = identifier.identify(y, u)

        assert model is not None
        assert model.n >= 0
        assert model.A.shape[0] == model.A.shape[1]  # Square A matrix
        assert model.B.shape[0] == model.n
        assert model.C.shape[1] == model.n

    def test_custom_config(self, sample_data):
        """Test identification with custom configuration."""
        y, u = sample_data
        config = SystemIdentificationConfig(
            method="N4SID", ss_f=15, ss_fixed_order=2, ss_threshold=0.1
        )
        identifier = SystemIdentification(config)
        model = identifier.identify(y, u)

        assert model is not None
        assert model.n <= 2  # Should respect the fixed order

    def test_algorithm_methods(self, sample_data):
        """Test different identification methods."""
        y, u = sample_data
        methods = ["N4SID"]  # Start with N4SID, add others when implemented

        for method in methods:
            config = SystemIdentificationConfig(method=method, ss_fixed_order=1)
            identifier = SystemIdentification(config)
            model = identifier.identify(y, u)
            assert model is not None

    def test_centering_options(self, sample_data):
        """Test different centering options."""
        y, u = sample_data
        centering_options = ["None", "InitVal", "MeanVal"]

        for centering in centering_options:
            config = SystemIdentificationConfig(centering=centering, ss_fixed_order=1)
            identifier = SystemIdentification(config)
            model = identifier.identify(y, u)
            assert model is not None


class TestBackwardCompatibility:
    """Test backward compatibility with original API."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample test data."""
        np.random.seed(42)
        n_points = 100
        u = np.random.randn(1, n_points)
        y = np.zeros((1, n_points))
        for i in range(1, n_points):
            y[0, i] = 0.8 * y[0, i - 1] + 0.5 * u[0, i - 1] + 0.1 * np.random.randn()
        return y, u

    def test_original_function_signature(self, sample_data):
        """Test that the original function signature works."""
        y, u = sample_data
        # This should mimic the original API exactly
        model = system_identification(
            y=y, u=u, id_method="N4SID", tsample=1.0, SS_fixed_order=2, SS_f=10
        )
        assert model is not None
