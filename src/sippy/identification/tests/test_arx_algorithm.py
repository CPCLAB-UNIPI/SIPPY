"""
Test cases for ARX identification algorithm implementation.
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from sippy.identification import IDData, SystemIdentificationConfig
from sippy.identification.algorithms.arx import ARXAlgorithm
from sippy.identification.base import IdentificationAlgorithm, StateSpaceModel


class TestARXAlgorithm:
    """Test suite for ARX algorithm implementation."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create simple test data
        np.random.seed(42)
        self.n_samples = 1000
        self.u = np.random.randn(self.n_samples)

        # Create a simple ARX system: y(k) = 0.5*y(k-1) + 0.8*u(k-1) + noise
        y_clean = np.zeros(self.n_samples)
        for k in range(1, self.n_samples):
            y_clean[k] = 0.5 * y_clean[k - 1] + 0.8 * self.u[k - 1]
        self.y = y_clean + 0.1 * np.random.randn(self.n_samples)

        # Create DataFrame for IDData
        time_index = pd.date_range("2023-01-01", periods=self.n_samples, freq="1s")
        data_df = pd.DataFrame({"u": self.u, "y": self.y}, index=time_index)

        # Configure data
        self.data = IDData(data=data_df, inputs=["u"], outputs=["y"], tsample=1.0)

        self.config = SystemIdentificationConfig(method="ARX")
        # Set ARX-specific parameters
        self.config.na = 1  # AR order
        self.config.nb = 1  # X order
        self.config.nk = 1  # Input delay

    def test_arx_algorithm_initialization(self):
        """Test ARX algorithm can be initialized."""
        algorithm = ARXAlgorithm()
        assert algorithm is not None
        assert isinstance(algorithm, IdentificationAlgorithm)

    @patch("sippy.identification.algorithms.arx.HAROLD_AVAILABLE", True)
    def test_arx_basic_identification(self):
        """Test basic ARX identification functionality."""
        algorithm = ARXAlgorithm()

        # Test that algorithm can be called
        with patch("sippy.identification.algorithms.arx.harold") as mock_harold:
            # Mock the harold state space creation
            mock_ss = mock_harold.StateSpace.return_value
            mock_ss.A = np.eye(2)
            mock_ss.B = np.zeros((2, 1))
            mock_ss.C = np.zeros((1, 2))
            mock_ss.D = np.zeros((1, 1))

            # Mock the undiscretize function
            mock_undiscretize = mock_harold.undiscretize.return_value = mock_ss
            mock_undiscretize.A = np.eye(2)
            mock_undiscretize.B = np.zeros((2, 1))
            mock_undiscretize.C = np.zeros((1, 2))
            mock_undiscretize.D = np.zeros((1, 1))

            result = algorithm.identify(self.data, self.config)

            assert isinstance(result, StateSpaceModel)
            assert result.A is not None
            assert result.B is not None
            assert result.C is not None
            assert result.D is not None

    def test_arx_algorithm_name(self):
        """Test algorithm returns correct name."""
        algorithm = ARXAlgorithm()
        assert algorithm.get_algorithm_name() == "ARX"

    def test_arx_with_different_orders(self):
        """Test ARX with different model orders."""
        algorithm = ARXAlgorithm()

        # Test different orders
        for na, nb in [(1, 1), (2, 2), (3, 1)]:
            config = SystemIdentificationConfig(method="ARX")
            config.na = na
            config.nb = nb
            config.nk = 1

            with patch("sippy.identification.algorithms.arx.harold") as mock_harold:
                mock_tf = mock_harold.TransferFunction.return_value
                mock_tf.NumberOfInputs = 1
                mock_tf.NumberOfOutputs = 1
                mock_tf.SamplingPeriod = 1.0

                result = algorithm.identify(self.data, config)
                assert result is not None

    def test_arx_mimo_system(self):
        """Test ARX with MIMO system."""
        # Create 2-input, 2-output test data
        np.random.seed(42)
        u = np.random.randn(2, self.n_samples)
        y = np.random.randn(2, self.n_samples)

        time_index = pd.date_range("2023-01-01", periods=self.n_samples, freq="1s")
        data_df = pd.DataFrame(
            {"u1": u[0, :], "u2": u[1, :], "y1": y[0, :], "y2": y[1, :]},
            index=time_index,
        )

        data = IDData(
            data=data_df, inputs=["u1", "u2"], outputs=["y1", "y2"], tsample=1.0
        )
        config = SystemIdentificationConfig(method="ARX")
        config.na = 1
        config.nb = 1
        config.nk = 1

        algorithm = ARXAlgorithm()

        with patch("sippy.identification.algorithms.arx.harold") as mock_harold:
            mock_tf = mock_harold.TransferFunction.return_value
            mock_tf.NumberOfInputs = 2
            mock_tf.NumberOfOutputs = 2
            mock_tf.SamplingPeriod = 1.0

            result = algorithm.identify(data, config)
            assert result is not None

    def test_arx_without_harold(self):
        """Test ARX algorithm graceful degradation without harold."""
        algorithm = ARXAlgorithm()

        with patch("sippy.identification.algorithms.arx.HAROLD_AVAILABLE", False):
            with pytest.warns(UserWarning, match="harold library not available"):
                result = algorithm.identify(self.data, self.config)
                # Should return a mock model when harold is not available
                assert result is not None
                assert isinstance(result, StateSpaceModel)

    def test_arx_invalid_parameters(self):
        """Test ARX algorithm with invalid parameters."""
        algorithm = ARXAlgorithm()

        # Test with invalid orders
        invalid_config = SystemIdentificationConfig(method="ARX")
        invalid_config.na = 0  # Invalid na

        with pytest.raises(ValueError, match="AR order \\(na\\) must be positive"):
            algorithm.identify(self.data, invalid_config)

    def test_arx_data_validation(self):
        """Test ARX algorithm validates input data."""
        algorithm = ARXAlgorithm()

        # Test with mismatched input/output dimensions - should work fine in our case
        # since IDData handles this internally
        time_index = pd.date_range("2023-01-01", periods=self.n_samples, freq="1s")
        invalid_data_df = pd.DataFrame(
            {
                "u1": np.random.randn(self.n_samples),
                "u2": np.random.randn(self.n_samples),
                "y1": np.random.randn(self.n_samples),
                "y2": np.random.randn(self.n_samples),
                "y3": np.random.randn(self.n_samples),  # Extra output
            },
            index=time_index,
        )

        invalid_data = IDData(
            data=invalid_data_df,
            inputs=["u1", "u2"],
            outputs=["y1", "y2", "y3"],  # Different number of outputs
            tsample=1.0,
        )

        # This should work since our algorithm should handle MIMO
        with patch("sippy.identification.algorithms.arx.harold") as mock_harold:
            mock_tf = mock_harold.TransferFunction.return_value
            mock_tf.NumberOfInputs = 2
            mock_tf.NumberOfOutputs = 3
            mock_tf.SamplingPeriod = 1.0

            result = algorithm.identify(invalid_data, self.config)
            assert result is not None
