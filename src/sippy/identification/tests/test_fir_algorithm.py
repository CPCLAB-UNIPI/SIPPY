"""
Test cases for FIR identification algorithm implementation.
"""
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from sippy.identification import IDData, SystemIdentificationConfig
from sippy.identification.algorithms.fir import FIRAlgorithm
from sippy.identification.base import IdentificationAlgorithm, StateSpaceModel


class TestFIRAlgorithm:
    """Test suite for FIR algorithm implementation."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create simple test data
        np.random.seed(42)
        self.n_samples = 1000
        self.u = np.random.randn(self.n_samples)

        # Create a simple FIR system: y(k) = 0.8*u(k-1) + 0.3*u(k-2) + noise
        fir_coeffs = [0.8, 0.3]
        y_clean = np.zeros(self.n_samples)
        for k in range(2, self.n_samples):
            for i, coeff in enumerate(fir_coeffs):
                if k - i - 1 >= 0:
                    y_clean[k] += coeff * self.u[k - i - 1]
        self.y = y_clean + 0.1 * np.random.randn(self.n_samples)

        # Create DataFrame for IDData
        time_index = pd.date_range('2023-01-01', periods=self.n_samples, freq='1s')
        data_df = pd.DataFrame({
            'u': self.u,
            'y': self.y
        }, index=time_index)

        # Configure data
        self.data = IDData(
            data=data_df,
            inputs=['u'],
            outputs=['y'],
            tsample=1.0
        )

        self.config = SystemIdentificationConfig(method='FIR')
        # Set FIR-specific parameters
        self.config.nb = 3  # Number of FIR coefficients
        self.config.nk = 1  # Input delay

    def test_fir_algorithm_initialization(self):
        """Test FIR algorithm can be initialized."""
        algorithm = FIRAlgorithm()
        assert algorithm is not None
        assert isinstance(algorithm, IdentificationAlgorithm)

    def test_fir_algorithm_name(self):
        """Test algorithm returns correct name."""
        algorithm = FIRAlgorithm()
        assert algorithm.get_algorithm_name() == "FIR"

    @patch('sippy.identification.algorithms.fir.HAROLD_AVAILABLE', True)
    def test_fir_basic_identification(self):
        """Test basic FIR identification functionality."""
        algorithm = FIRAlgorithm()

        # Test that algorithm can be called
        with patch('sippy.identification.algorithms.fir.harold') as mock_harold:
            # Mock the harold state space creation
            mock_ss = mock_harold.StateSpace.return_value
            mock_ss.A = np.zeros((3, 3))
            mock_ss.B = np.zeros((3, 1))
            mock_ss.C = np.zeros((1, 3))
            mock_ss.D = np.zeros((1, 1))

            result = algorithm.identify(self.data, self.config)

            assert isinstance(result, StateSpaceModel)
            assert result.A is not None
            assert result.B is not None
            assert result.C is not None
            assert result.D is not None

    def test_fir_with_different_orders(self):
        """Test FIR with different number of coefficients."""
        algorithm = FIRAlgorithm()

        # Test different coefficient counts
        for nb in [2, 3, 5, 10]:
            config = SystemIdentificationConfig(method='FIR')
            config.nb = nb
            config.nk = 1

            with patch('sippy.identification.algorithms.fir.harold') as mock_harold:
                mock_ss = mock_harold.StateSpace.return_value
                mock_ss.A = np.zeros((nb, nb))
                mock_ss.B = np.zeros((nb, 1))
                mock_ss.C = np.zeros((1, nb))
                mock_ss.D = np.zeros((1, 1))

                result = algorithm.identify(self.data, config)
                assert result is not None

    def test_fir_mimo_system(self):
        """Test FIR with MIMO system."""
        # Create 2-input, 2-output test data
        np.random.seed(42)
        u = np.random.randn(2, self.n_samples)
        y = np.random.randn(2, self.n_samples)

        time_index = pd.date_range('2023-01-01', periods=self.n_samples, freq='1s')
        data_df = pd.DataFrame({
            'u1': u[0, :],
            'u2': u[1, :],
            'y1': y[0, :],
            'y2': y[1, :]
        }, index=time_index)

        data = IDData(
            data=data_df,
            inputs=['u1', 'u2'],
            outputs=['y1', 'y2'],
            tsample=1.0
        )
        config = SystemIdentificationConfig(method='FIR')
        config.nb = 5
        config.nk = 1

        algorithm = FIRAlgorithm()

        with patch('sippy.identification.algorithms.fir.harold') as mock_harold:
            mock_ss = mock_harold.StateSpace.return_value
            mock_ss.A = np.zeros((5, 5))
            mock_ss.B = np.zeros((5, 2))
            mock_ss.C = np.zeros((2, 5))
            mock_ss.D = np.zeros((2, 2))

            result = algorithm.identify(data, config)
            assert result is not None

    def test_fir_without_harold(self):
        """Test FIR algorithm graceful degradation without harold."""
        algorithm = FIRAlgorithm()

        with patch('sippy.identification.algorithms.fir.HAROLD_AVAILABLE', False):
            result = algorithm.identify(self.data, self.config)
            # Should return a mock model when harold is not available
            assert result is not None
            assert isinstance(result, StateSpaceModel)

    def test_fir_invalid_parameters(self):
        """Test FIR algorithm with invalid parameters."""
        algorithm = FIRAlgorithm()

        # Test with invalid coefficient count
        invalid_config = SystemIdentificationConfig(method='FIR')
        invalid_config.nb = 0  # Invalid nb

        with pytest.raises(ValueError, match="Number of FIR coefficients must be positive"):
            algorithm.identify(self.data, invalid_config)

    def test_fir_data_validation(self):
        """Test FIR algorithm validates input data."""
        algorithm = FIRAlgorithm()

        # Test with MIMO data
        time_index = pd.date_range('2023-01-01', periods=self.n_samples, freq='1s')
        data_df = pd.DataFrame({
            'u1': np.random.randn(self.n_samples),
            'u2': np.random.randn(self.n_samples),
            'y1': np.random.randn(self.n_samples),
            'y2': np.random.randn(self.n_samples),
            'y3': np.random.randn(self.n_samples)  # Extra output
        }, index=time_index)

        data = IDData(
            data=data_df,
            inputs=['u1', 'u2'],
            outputs=['y1', 'y2', 'y3'],  # Different number of outputs
            tsample=1.0
        )

        # This should work since our algorithm should handle MIMO
        config = SystemIdentificationConfig(method='FIR')
        config.nb = 5
        config.nk = 1

        with patch('sippy.identification.algorithms.fir.harold') as mock_harold:
            mock_ss = mock_harold.StateSpace.return_value
            mock_ss.A = np.eye(5)
            mock_ss.B = np.zeros((5, 2))
            mock_ss.C = np.zeros((3, 5))
            mock_ss.D = np.zeros((3, 2))

            result = algorithm.identify(data, config)
            assert result is not None
