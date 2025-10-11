"""
Test suite for ARMA algorithm implementation.
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from sippy.identification.algorithms.arma import ARMAAlgorithm
from sippy.identification.base import SystemIdentificationConfig, StateSpaceModel
from sippy.identification.iddata import IDData


class TestARMAAlgorithm:
    """Test cases for ARMA algorithm."""

    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        self.n_samples = 1000

        # Create test time series data (ARMA typically works with SISO)
        t = np.linspace(0, 100, self.n_samples)
        # Generate ARMA(2,1) process: y[k] = 0.6*y[k-1] + 0.2*y[k-2] + e[k] + 0.3*e[k-1]
        y = np.zeros(self.n_samples)
        e = np.random.normal(0, 0.1, self.n_samples)

        for k in range(2, self.n_samples):
            y[k] = 0.6 * y[k - 1] + 0.2 * y[k - 2] + e[k] + 0.3 * e[k - 1]

        # Create IDData (we need dummy inputs but ARMA ignores them)
        u = np.zeros((1, self.n_samples))  # ARMA has no exogenous inputs

        time_index = pd.date_range("2023-01-01", periods=self.n_samples, freq="1s")
        data_df = pd.DataFrame({"y1": y, "u1": u[0, :]}, index=time_index)

        self.data = IDData(data=data_df, inputs=["u1"], outputs=["y1"], tsample=1.0)

        self.config = SystemIdentificationConfig(method="ARMA")
        self.config.na = 2
        self.config.nc = 1

    def test_arma_algorithm_initialization(self):
        """Test ARMA algorithm can be initialized."""
        algorithm = ARMAAlgorithm()
        assert algorithm.get_algorithm_name() == "ARMA"

    def test_arma_algorithm_name(self):
        """Test ARMA algorithm name."""
        algorithm = ARMAAlgorithm()
        assert algorithm.get_algorithm_name() == "ARMA"

    def test_arma_basic_identification(self):
        """Test ARMA basic identification."""
        algorithm = ARMAAlgorithm()

        result = algorithm.identify(self.data, self.config)

        assert result is not None
        assert isinstance(result, StateSpaceModel)
        assert result.A is not None
        assert result.B is not None
        assert result.C is not None
        assert result.D is not None

    def test_arma_with_different_orders(self):
        """Test ARMA with different model orders."""
        algorithm = ARMAAlgorithm()

        # Test ARMA(3,2)
        config = SystemIdentificationConfig(method="ARMA")
        config.na = 3
        config.nc = 2

        result = algorithm.identify(self.data, config)
        assert result is not None
        assert isinstance(result, StateSpaceModel)

    def test_arma_mimo_system(self):
        """Test ARMA with MIMO system (though ARMA is typically SISO)."""
        # Create 2-output time series data
        np.random.seed(42)
        n_samples = 500
        t = np.linspace(0, 50, n_samples)

        # Two independent ARMA processes
        y1 = np.zeros(n_samples)
        y2 = np.zeros(n_samples)
        e1 = np.random.normal(0, 0.1, n_samples)
        e2 = np.random.normal(0, 0.1, n_samples)

        for k in range(2, n_samples):
            y1[k] = 0.6 * y1[k - 1] + 0.2 * y1[k - 2] + e1[k] + 0.3 * e1[k - 1]
            y2[k] = 0.4 * y2[k - 1] + 0.3 * y2[k - 2] + e2[k] + 0.2 * e2[k - 1]

        y = np.vstack([y1, y2])
        u = np.zeros((1, n_samples))  # Dummy input

        time_index = pd.date_range("2023-01-01", periods=n_samples, freq="1s")
        data_df = pd.DataFrame({"y1": y1, "y2": y2, "u1": u[0, :]}, index=time_index)

        data = IDData(data=data_df, inputs=["u1"], outputs=["y1", "y2"], tsample=1.0)

        config = SystemIdentificationConfig(method="ARMA")
        config.na = 2
        config.nc = 1

        algorithm = ARMAAlgorithm()
        result = algorithm.identify(data, config)

        assert result is not None
        assert isinstance(result, StateSpaceModel)

    def test_arma_without_harold(self):
        """Test ARMA algorithm graceful degradation without harold."""
        algorithm = ARMAAlgorithm()

        with patch("sippy.identification.algorithms.arma.HAROLD_AVAILABLE", False):
            result = algorithm.identify(self.data, self.config)
            assert result is not None
            assert isinstance(result, StateSpaceModel)

    def test_arma_invalid_parameters(self):
        """Test ARMA algorithm with invalid parameters."""
        algorithm = ARMAAlgorithm()

        # Test with zero AR order
        invalid_config = SystemIdentificationConfig(method="ARMA")
        invalid_config.na = 0
        invalid_config.nc = 1

        with pytest.raises(ValueError, match="AR order .* must be positive"):
            algorithm.identify(self.data, invalid_config)

        # Test with zero MA order
        invalid_config = SystemIdentificationConfig(method="ARMA")
        invalid_config.na = 1
        invalid_config.nc = 0

        with pytest.raises(ValueError, match="MA order .* must be positive"):
            algorithm.identify(self.data, invalid_config)

    def test_arma_insufficient_data(self):
        """Test ARMA algorithm with insufficient data."""
        algorithm = ARMAAlgorithm()

        # Create very short dataset
        n_samples = 5
        np.random.seed(42)
        y = np.random.randn(1, n_samples)
        u = np.zeros((1, n_samples))

        time_index = pd.date_range("2023-01-01", periods=n_samples, freq="1s")
        data_df = pd.DataFrame({"y1": y[0, :], "u1": u[0, :]}, index=time_index)

        short_data = IDData(data=data_df, inputs=["u1"], outputs=["y1"], tsample=1.0)

        config = SystemIdentificationConfig(method="ARMA")
        config.na = 5
        config.nc = 2

        with pytest.raises(ValueError, match="Not enough data"):
            algorithm.identify(short_data, config)

    def test_arma_state_space_models(self):
        """Test ARMA creates valid state-space models."""
        algorithm = ARMAAlgorithm()

        result = algorithm.identify(self.data, self.config)

        # Check state-space model properties
        assert hasattr(result, "A")
        assert hasattr(result, "B")
        assert hasattr(result, "C")
        assert hasattr(result, "D")

        # Check dimensions are consistent
        A, B, C, D = result.A, result.B, result.C, result.D
        assert A.shape[0] == A.shape[1]  # A square
        assert A.shape[0] == B.shape[0]  # A and B rows match
        assert C.shape[1] == A.shape[1]  # C columns match A columns
        assert C.shape[0] == D.shape[0]  # C and D rows match

    @pytest.mark.parametrize("na,nc", [(1, 1), (2, 1), (1, 2), (3, 2)])
    def test_arma_various_orders(self, na, nc):
        """Test ARMA with various order combinations."""
        algorithm = ARMAAlgorithm()

        config = SystemIdentificationConfig(method="ARMA")
        config.na = na
        config.nc = nc

        result = algorithm.identify(self.data, config)
        assert result is not None
        assert isinstance(result, StateSpaceModel)
