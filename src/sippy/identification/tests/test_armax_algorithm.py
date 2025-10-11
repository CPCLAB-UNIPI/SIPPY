"""
Test cases for ARMAX identification algorithm implementation.
"""
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from sippy.identification import IDData, SystemIdentificationConfig
from sippy.identification.algorithms.armax import ARMAXAlgorithm
from sippy.identification.base import IdentificationAlgorithm, StateSpaceModel


class TestARMAXAlgorithm:
    """Test suite for ARMAX algorithm implementation."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create simple test data with ARMAX characteristics
        np.random.seed(42)
        self.n_samples = 1000
        self.u = np.random.randn(self.n_samples)

        # Create a simple ARMAX system: y(k) = 0.7*y(k-1) + 0.5*u(k-1) + 0.3*e(k-1) + 0.1*e(k)
        e = np.random.randn(self.n_samples) * 0.1
        y_clean = np.zeros(self.n_samples)
        for k in range(1, self.n_samples):
            y_clean[k] = 0.7 * y_clean[k-1] + 0.5 * self.u[k-1] + 0.3 * e[k-1]
        self.y = y_clean + 0.05 * np.random.randn(self.n_samples)

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

        self.config = SystemIdentificationConfig(method='ARMAX')
        # Set ARMAX-specific parameters
        self.config.na = 1  # AR order
        self.config.nb = 1  # X order
        self.config.nc = 1  # MA order
        self.config.nk = 1  # Input delay

    def test_armax_algorithm_initialization(self):
        """Test ARMAX algorithm can be initialized."""
        algorithm = ARMAXAlgorithm()
        assert algorithm is not None
        assert isinstance(algorithm, IdentificationAlgorithm)

    def test_armax_algorithm_name(self):
        """Test algorithm returns correct name."""
        algorithm = ARMAXAlgorithm()
        assert algorithm.get_algorithm_name() == "ARMAX"

    def test_armax_parameter_validation(self):
        """Test ARMAX parameter validation."""
        algorithm = ARMAXAlgorithm()

        # Test valid parameters
        algorithm.validate_parameters(na=1, nb=1, nc=1, nk=0)
        algorithm.validate_parameters(na=2, nb=3, nc=2, nk=1)

        # Test boundary conditions
        algorithm.validate_parameters(na=1, nb=1, nc=1, nk=0)
        with pytest.raises(ValueError, match="AR order \\(na\\) must be positive"):
            algorithm.validate_parameters(na=0, nb=1, nc=1, nk=0)
        with pytest.raises(ValueError, match="X order \\(nb\\) must be positive"):
            algorithm.validate_parameters(na=1, nb=0, nc=1, nk=0)
        with pytest.raises(ValueError, match="MA order \\(nc\\) must be positive"):
            algorithm.validate_parameters(na=1, nb=1, nc=0, nk=0)
        with pytest.raises(ValueError, match="Input delay \\(nk\\) must be non-negative"):
            algorithm.validate_parameters(na=1, nb=1, nc=1, nk=-1)

    @patch('sippy.identification.algorithms.armax.HAROLD_AVAILABLE', True)
    def test_armax_basic_identification(self):
        """Test basic ARMAX identification functionality."""
        algorithm = ARMAXAlgorithm()

        with patch('sippy.identification.algorithms.armax.harold') as mock_harold:
            # Mock the harold state space creation
            mock_ss = mock_harold.StateSpace.return_value
            mock_ss.A = np.eye(3)
            mock_ss.B = np.zeros((3, 1))
            mock_ss.C = np.zeros((1, 3))
            mock_ss.D = np.zeros((1, 1))

            result = algorithm.identify(self.data, self.config)

            assert isinstance(result, StateSpaceModel)
            assert result.A is not None
            assert result.B is not None
            assert result.C is not None
            assert result.D is not None

    def test_armax_with_different_orders(self):
        """Test ARMAX with different model orders."""
        algorithm = ARMAXAlgorithm()

        # Test different orders
        for na, nb, nc in [(1, 1, 1), (2, 2, 1), (1, 2, 2), (2, 1, 2)]:
            config = SystemIdentificationConfig(method='ARMAX')
            config.na = na
            config.nb = nb
            config.nc = nc
            config.nk = 1

            with patch('sippy.identification.algorithms.armax.harold') as mock_harold:
                mock_ss = mock_harold.StateSpace.return_value
                mock_ss.A = np.eye(na + nc)
                mock_ss.B = np.zeros((na + nc, 1))
                mock_ss.C = np.zeros((1, na + nc))
                mock_ss.D = np.zeros((1, 1))

                result = algorithm.identify(self.data, config)
                assert result is not None

    def test_armax_mimo_system(self):
        """Test ARMAX with MIMO system."""
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
        config = SystemIdentificationConfig(method='ARMAX')
        config.na = 1
        config.nb = 1
        config.nc = 1
        config.nk = 1

        algorithm = ARMAXAlgorithm()

        with patch('sippy.identification.algorithms.armax.harold') as mock_harold:
            mock_ss = mock_harold.StateSpace.return_value
            mock_ss.A = np.eye(3)
            mock_ss.B = np.zeros((3, 2))
            mock_ss.C = np.zeros((2, 3))
            mock_ss.D = np.zeros((2, 2))

            result = algorithm.identify(data, config)
            assert result is not None

    def test_armax_without_harold(self):
        """Test ARMAX algorithm graceful degradation without harold."""
        algorithm = ARMAXAlgorithm()

        with patch('sippy.identification.algorithms.armax.HAROLD_AVAILABLE', False):
            result = algorithm.identify(self.data, self.config)
            # Should return a mock model when harold is not available
            assert result is not None
            assert isinstance(result, StateSpaceModel)

    def test_armax_data_validation(self):
        """Test ARMAX algorithm validates input data."""
        algorithm = ARMAXAlgorithm()

        # Test with insufficient data
        small_data_time_index = pd.date_range('2023-01-01', periods=5, freq='1s')
        small_data_df = pd.DataFrame({
            'u': np.random.randn(5),
            'y': np.random.randn(5)
        }, index=small_data_time_index)

        small_data = IDData(
            data=small_data_df,
            inputs=['u'],
            outputs=['y'],
            tsample=1.0
        )

        config = SystemIdentificationConfig(method='ARMAX')
        config.na = 4  # Requires more data than available
        config.nb = 2
        config.nc = 2
        config.nk = 1

        with pytest.raises(ValueError, match="Not enough data points"):
            algorithm.identify(small_data, config)

    def test_armax_order_calculation(self):
        """Test that ARMAX calculates correct model order."""
        algorithm = ARMAXAlgorithm()

        config = SystemIdentificationConfig(method='ARMAX')
        config.na = 2
        config.nb = 1
        config.nc = 1
        config.nk = 0

        with patch('sippy.identification.algorithms.armax.harold') as mock_harold:
            mock_ss = mock_harold.StateSpace.return_value
            mock_ss.A = np.eye(3)  # na + nc = 3
            mock_ss.B = np.zeros((3, 1))
            mock_ss.C = np.zeros((1, 3))
            mock_ss.D = np.zeros((1, 1))

            result = algorithm.identify(self.data, config)
            assert result.A.shape == (3, 3)  # State dimension = na + nc
            assert result.n == 3

    def test_armax_noise_modeling(self):
        """Test ARMAX properly models noise dynamics."""
        algorithm = ARMAXAlgorithm()

        config = SystemIdentificationConfig(method='ARMAX')
        config.na = 1
        config.nb = 1
        config.nc = 1
        config.nk = 0

        with patch('sippy.identification.algorithms.armax.harold') as mock_harold:
            mock_ss = mock_harold.StateSpace.return_value
            mock_ss.A = np.array([[0.7, 0.3], [-0.2, -0.5]])
            mock_ss.B = np.array([[0.5], [0.1]])
            mock_ss.C = np.array([[1.0, 0.0]])
            mock_ss.D = np.array([[0.0]])

            result = algorithm.identify(self.data, config)
            assert result is not None
            # State matrix should reflect AR + MA dynamics
            assert result.A.shape == (2, 2)  # na + nc = 2
