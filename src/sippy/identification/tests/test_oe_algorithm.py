"""
Test cases for OE (Output Error) identification algorithm implementation.
"""
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from sippy.identification import IDData, SystemIdentificationConfig
from sippy.identification.algorithms.oe import OEAlgorithm
from sippy.identification.base import IdentificationAlgorithm, StateSpaceModel


class TestOEAlgorithm:
    """Test suite for OE algorithm implementation."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create simple test data with OE characteristics
        np.random.seed(42)
        self.n_samples = 1000
        self.u = np.random.randn(self.n_samples)

        # Create a simple OE system: y(k) = (0.5*u(k-1) + 0.3*u(k-2)) / (1 + 0.3*yp(k-1) - 0.2*yp(k-2)) + noise
        y_clean = np.zeros(self.n_samples)
        noise_free_output = np.zeros(self.n_samples)
        for k in range(2, self.n_samples):
            # Noise-free output feedback
            denominator = 1 + 0.3 * noise_free_output[k-1] - 0.2 * noise_free_output[k-2]
            numerator = 0.5 * self.u[k-1] + 0.3 * self.u[k-2]
            noise_free_output[k] = numerator / denominator
            y_clean[k] = noise_free_output[k]
        
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

        self.config = SystemIdentificationConfig(method='OE')
        # Set OE-specific parameters
        self.config.nb = 2  # Numerator order
        self.config.nf = 2  # Denominator order
        self.config.nk = 1  # Input delay

    def test_oe_algorithm_initialization(self):
        """Test OE algorithm can be initialized."""
        algorithm = OEAlgorithm()
        assert algorithm is not None
        assert isinstance(algorithm, IdentificationAlgorithm)

    def test_oe_algorithm_name(self):
        """Test algorithm returns correct name."""
        algorithm = OEAlgorithm()
        assert algorithm.get_algorithm_name() == "OE"

    def test_oe_parameter_validation(self):
        """Test OE parameter validation."""
        algorithm = OEAlgorithm()

        # Test valid parameters
        algorithm.validate_parameters(nb=2, nf=2, nk=0)
        algorithm.validate_parameters(nb=1, nf=1, nk=1)
        algorithm.validate_parameters(nb=3, nf=4, nk=2)

        # Test boundary conditions
        with pytest.raises(ValueError, match="Numerator order \\(nb\\) must be positive"):
            algorithm.validate_parameters(nb=0, nf=2, nk=0)
        with pytest.raises(ValueError, match="Denominator order \\(nf\\) must be positive"):
            algorithm.validate_parameters(nb=2, nf=0, nk=0)
        with pytest.raises(ValueError, match="Input delay \\(nk\\) must be non-negative"):
            algorithm.validate_parameters(nb=2, nf=2, nk=-1)

    @patch('sippy.identification.algorithms.oe.HAROLD_AVAILABLE', True)
    def test_oe_basic_identification(self):
        """Test basic OE identification functionality."""
        algorithm = OEAlgorithm()

        with patch('sippy.identification.algorithms.oe.harold') as mock_harold:
            # Mock the harold state space creation
            mock_ss = mock_harold.StateSpace.return_value
            mock_ss.A = np.eye(2)
            mock_ss.B = np.zeros((2, 1))
            mock_ss.C = np.zeros((1, 2))
            mock_ss.D = np.zeros((1, 1))

            result = algorithm.identify(self.data, self.config)

            assert isinstance(result, StateSpaceModel)
            assert result.A is not None
            assert result.B is not None
            assert result.C is not None
            assert result.D is not None

    def test_oe_with_different_orders(self):
        """Test OE with different model orders."""
        algorithm = OEAlgorithm()

        # Test different orders
        for nb, nf in [(2, 2), (3, 2), (2, 3), (3, 3)]:
            config = SystemIdentificationConfig(method='OE')
            config.nb = nb
            config.nf = nf
            config.nk = 1

            with patch('sippy.identification.algorithms.oe.harold') as mock_harold:
                mock_ss = mock_harold.StateSpace.return_value
                mock_ss.A = np.eye(nf)
                mock_ss.B = np.zeros((nf, 1))
                mock_ss.C = np.zeros((1, nf))
                mock_ss.D = np.zeros((1, 1))

                result = algorithm.identify(self.data, config)
                assert result is not None

    def test_oe_mimo_system(self):
        """Test OE with MIMO system."""
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
        config = SystemIdentificationConfig(method='OE')
        config.nb = 2
        config.nf = 2
        config.nk = 1

        algorithm = OEAlgorithm()

        with patch('sippy.identification.algorithms.oe.harold') as mock_harold:
            mock_ss = mock_harold.StateSpace.return_value
            mock_ss.A = np.eye(2)
            mock_ss.B = np.zeros((2, 2))
            mock_ss.C = np.zeros((2, 2))
            mock_ss.D = np.zeros((2, 2))

            result = algorithm.identify(data, config)
            assert result is not None

    def test_oe_without_harold(self):
        """Test OE algorithm graceful degradation without harold."""
        algorithm = OEAlgorithm()

        with patch('sippy.identification.algorithms.oe.HAROLD_AVAILABLE', False):
            result = algorithm.identify(self.data, self.config)
            # Should return a mock model when harold is not available
            assert result is not None
            assert isinstance(result, StateSpaceModel)

    def test_oe_data_validation(self):
        """Test OE algorithm validates input data."""
        algorithm = OEAlgorithm()

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

        config = SystemIdentificationConfig(method='OE')
        config.nb = 5  # Requires more data than available
        config.nf = 4
        config.nk = 1

        with pytest.raises(ValueError, match="Not enough data points"):
            algorithm.identify(small_data, config)

    def test_oe_order_calculation(self):
        """Test that OE calculates correct model order."""
        algorithm = OEAlgorithm()

        config = SystemIdentificationConfig(method='OE')
        config.nb = 2
        config.nf = 3  # Denominator order determines state dimension
        config.nk = 0

        with patch('sippy.identification.algorithms.oe.harold') as mock_harold:
            mock_ss = mock_harold.StateSpace.return_value
            mock_ss.A = np.eye(3)  # nf = 3
            mock_ss.B = np.zeros((3, 1))
            mock_ss.C = np.zeros((1, 3))
            mock_ss.D = np.zeros((1, 1))

            result = algorithm.identify(self.data, config)
            assert result.A.shape == (3, 3)  # State dimension = nf
            assert result.n == 3

    def test_oe_noise_modeling(self):
        """Test OE properly models output error structure."""
        algorithm = OEAlgorithm()

        config = SystemIdentificationConfig(method='OE')
        config.nb = 2
        config.nf = 2
        config.nk = 0

        with patch('sippy.identification.algorithms.oe.harold') as mock_harold:
            mock_ss = mock_harold.StateSpace.return_value
            mock_ss.A = np.array([[0.0, 1.0], [-0.3, 0.2]])  # F coefficients
            mock_ss.B = np.array([[0.1], [0.5]])  # B coefficients
            mock_ss.C = np.array([[0.0, 1.0]])  # Observer canonical form
            mock_ss.D = np.array([[0.0]])

            result = algorithm.identify(self.data, config)
            assert result is not None
            # State matrix should reflect F polynomial coefficients
            assert result.A.shape == (2, 2)  # nf = 2 states
