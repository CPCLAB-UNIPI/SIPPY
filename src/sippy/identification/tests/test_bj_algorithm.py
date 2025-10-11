"""
Test suite for Box-Jenkins (BJ) algorithm implementation.
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from sippy.identification.algorithms.bj import BJAlgorithm
from sippy.identification.base import StateSpaceModel, SystemIdentificationConfig
from sippy.identification.iddata import IDData


class TestBJAlgorithm:
    """Test cases for Box-Jenkins algorithm following TDD approach."""

    def setup_method(self):
        """Set up test data for BJ algorithm."""
        np.random.seed(42)
        self.n_samples = 1000

        # Create test data for BJ (SISO system with colored noise)
        t = np.linspace(0, 100, self.n_samples)
        u = np.random.normal(0, 1, self.n_samples)  # Input signal

        # Generate BJ process: y[k] = B(q)/D(q)*u[k] + C(q)/F(q)*e[k]
        # BJ(2,3,1,2) model as example
        na = 0  # No AR part in BJ
        nb = 2
        nc = 1
        nd = 3
        nf = 2

        # Simulate simplified BJ process
        y = np.zeros(self.n_samples)
        e = np.random.normal(0, 0.1, self.n_samples)

        for k in range(3, self.n_samples):
            # Input part: B(q)/D(q) * u[k] (simplified example)
            if k >= nb:
                input_part = 0.3 * u[k - 1] + 0.1 * u[k - 2]  # B(q)
                input_part -= 0.5 * y[k - 1] - 0.2 * y[k - 2] - 0.1 * y[k - 3]  # D(q)
            else:
                input_part = 0

            # Noise part: C(q)/F(q) * e[k]
            if k >= 1:
                noise_part = e[k] + 0.3 * e[k - 1]  # C(q)
                # F(q) would be in denominator, simplified here
            else:
                noise_part = e[k]

            y[k] = input_part + noise_part

        # Create IDData
        time_index = pd.date_range("2023-01-01", periods=self.n_samples, freq="1s")
        data_df = pd.DataFrame({"u1": u, "y1": y}, index=time_index)

        self.data = IDData(data=data_df, inputs=["u1"], outputs=["y1"], tsample=1.0)

        self.config = SystemIdentificationConfig(method="BJ")
        self.config.nb = 2  # Input polynomial order
        self.config.nc = 1  # Noise AR polynomial order
        self.config.nd = 3  # Noise MA polynomial order (D part)
        self.config.nf = 2  # Noise MA polynomial order (F part)
        self.config.nk = 1  # Input delay

    def test_bj_algorithm_initialization(self):
        """Test BJ algorithm can be initialized."""
        algorithm = BJAlgorithm()
        assert algorithm.get_algorithm_name() == "BJ"

    def test_bj_algorithm_name(self):
        """Test BJ algorithm name."""
        algorithm = BJAlgorithm()
        assert algorithm.get_algorithm_name() == "BJ"

    def test_bj_basic_identification(self):
        """Test BJ basic identification."""
        algorithm = BJAlgorithm()

        result = algorithm.identify(self.data, self.config)

        assert result is not None
        assert isinstance(result, StateSpaceModel)
        assert result.A is not None
        assert result.B is not None
        assert result.C is not None
        assert result.D is not None

    def test_bj_with_different_orders(self):
        """Test BJ with different model orders."""
        algorithm = BJAlgorithm()

        # Test BJ(1,2,1,1)
        config = SystemIdentificationConfig(method="BJ")
        config.nb = 1  # Input
        config.nc = 2  # Noise AR
        config.nd = 1  # D polynomial
        config.nf = 1  # F polynomial

        result = algorithm.identify(self.data, config)
        assert result is not None
        assert isinstance(result, StateSpaceModel)

    def test_bj_parameter_validation(self):
        """Test BJ parameter validation."""
        algorithm = BJAlgorithm()

        # Test with zero input order
        invalid_config = SystemIdentificationConfig(method="BJ")
        invalid_config.nb = 0
        invalid_config.nc = 1
        invalid_config.nd = 1
        invalid_config.nf = 1

        with pytest.raises(ValueError, match="Input order .* must be positive"):
            algorithm.identify(self.data, invalid_config)

        # Test with zero noise AR order
        invalid_config = SystemIdentificationConfig(method="BJ")
        invalid_config.nb = 1
        invalid_config.nc = 0
        invalid_config.nd = 1
        invalid_config.nf = 1

        with pytest.raises(ValueError, match="Noise AR order .* must be positive"):
            algorithm.identify(self.data, invalid_config)

        # Test with zero noise MA orders
        invalid_config = SystemIdentificationConfig(method="BJ")
        invalid_config.nb = 1
        invalid_config.nc = 1
        invalid_config.nd = 0
        invalid_config.nf = 0

        with pytest.raises(ValueError, match="Noise MA orders must be positive"):
            algorithm.identify(self.data, invalid_config)

    def test_bj_mimo_system(self):
        """Test BJ with MIMO system."""
        # Create 2-input, 2-output data
        np.random.seed(42)
        n_samples = 500

        u = np.random.randn(2, n_samples)
        y1 = np.random.randn(n_samples)
        y2 = np.random.randn(n_samples)

        # Simple input-output relationships
        for k in range(2, n_samples):
            y1[k] = (
                0.3 * u[0, k - 1]
                + 0.2 * u[1, k - 1]
                + 0.1 * y1[k - 1]
                + 0.1 * y2[k - 1]
            )
            y2[k] = (
                0.4 * u[0, k - 1]
                + 0.1 * u[1, k - 1]
                + 0.2 * y1[k - 1]
                + 0.05 * y2[k - 1]
            )

        y = np.vstack([y1, y2])

        time_index = pd.date_range("2023-01-01", periods=n_samples, freq="1s")
        data_df = pd.DataFrame(
            {"u1": u[0, :], "u2": u[1, :], "y1": y1, "y2": y2}, index=time_index
        )

        data = IDData(
            data=data_df, inputs=["u1", "u2"], outputs=["y1", "y2"], tsample=1.0
        )

        config = SystemIdentificationConfig(method="BJ")
        config.nb = 1
        config.nc = 1
        config.nd = 1
        config.nf = 1

        algorithm = BJAlgorithm()
        result = algorithm.identify(data, config)

        assert result is not None
        assert isinstance(result, StateSpaceModel)

    def test_bj_without_harold(self):
        """Test BJ algorithm graceful degradation without harold."""
        algorithm = BJAlgorithm()

        with patch("sippy.identification.algorithms.bj.HAROLD_AVAILABLE", False):
            result = algorithm.identify(self.data, self.config)
            assert result is not None
            assert isinstance(result, StateSpaceModel)

    def test_bj_insufficient_data(self):
        """Test BJ algorithm with insufficient data."""
        algorithm = BJAlgorithm()

        # Create very short dataset
        n_samples = 5
        np.random.seed(42)
        y = np.random.randn(1, n_samples)
        u = np.random.randn(1, n_samples)

        time_index = pd.date_range("2023-01-01", periods=n_samples, freq="1s")
        data_df = pd.DataFrame({"y1": y[0, :], "u1": u[0, :]}, index=time_index)

        short_data = IDData(data=data_df, inputs=["u1"], outputs=["y1"], tsample=1.0)

        config = SystemIdentificationConfig(method="BJ")
        config.nb = 8  # This will require at least 8 samples
        config.nc = 3
        config.nd = 4
        config.nf = 3

        with pytest.raises(ValueError, match="Not enough data"):
            algorithm.identify(short_data, config)

    def test_bj_state_space_models(self):
        """Test BJ creates valid state-space models."""
        algorithm = BJAlgorithm()

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

    def test_bj_algorithm_properties(self):
        """Test BJ algorithm has properties for model orders."""
        algorithm = BJAlgorithm()

        # Test that algorithm handles the BJ-specific structure
        # BJ has na=0 (no AR part), but has input and noise polynomials
        assert hasattr(algorithm, "get_algorithm_name")
        assert algorithm.get_algorithm_name() == "BJ"

    @pytest.mark.parametrize(
        "nb,nc,nd,nf",
        [
            (1, 1, 1, 1),  # Minimal BJ model
            (2, 1, 1, 1),  # Higher input order
            (1, 2, 1, 1),  # Higher noise AR
            (1, 1, 2, 1),  # Higher D polynomial
            (1, 1, 1, 2),  # Higher F polynomial
            (2, 2, 2, 2),  # Complex BJ model
        ],
    )
    def test_bj_various_orders(self, nb, nc, nd, nf):
        """Test BJ with various order combinations."""
        algorithm = BJAlgorithm()

        config = SystemIdentificationConfig(method="BJ")
        config.nb = nb
        config.nc = nc
        config.nd = nd
        config.nf = nf

        result = algorithm.identify(self.data, config)
        assert result is not None
        assert isinstance(result, StateSpaceModel)

    def test_bj_noise_modeling(self):
        """Test BJ properly handles noise modeling aspects."""
        algorithm = BJAlgorithm()

        # The BJ algorithm should handle colored noise through C(q)/F(q) structure
        # Create data with more complex noise characteristics
        np.random.seed(42)
        n_samples = 500
        u = np.random.normal(0, 1, n_samples)

        # Create colored noise
        e_white = np.random.normal(0, 0.1, n_samples)
        e_colored = np.zeros(n_samples)
        for k in range(2, n_samples):
            e_colored[k] = e_white[k] + 0.4 * e_white[k - 1] + 0.2 * e_white[k - 2]

        # Simple dynamics
        y = np.zeros(n_samples)
        for k in range(1, n_samples):
            y[k] = 0.3 * u[k - 1] + 0.1 * u[k - 2] + e_colored[k]

        time_index = pd.date_range("2023-01-01", periods=n_samples, freq="1s")
        data_df = pd.DataFrame({"u1": u, "y1": y}, index=time_index)

        data = IDData(data=data_df, inputs=["u1"], outputs=["y1"], tsample=1.0)

        config = SystemIdentificationConfig(method="BJ")
        config.nb = 2
        config.nc = 1
        config.nd = 1
        config.nf = 2

        result = algorithm.identify(data, config)
        assert result is not None
        assert isinstance(result, StateSpaceModel)

    def test_bj_order_calculation_consistency(self):
        """Test BJ order calculations are consistent with expected structure."""
        algorithm = BJAlgorithm()

        # For BJ, na should always be 0 (no AR part)
        # Total state dimension should reflect input and noise dynamics
        config = SystemIdentificationConfig(method="BJ")
        config.nb = 2
        config.nc = 1
        config.nd = 3
        config.nf = 2

        result = algorithm.identify(self.data, config)

        # The algorithm should create a model with appropriate state dimension
        # based on the sum of relevant polynomial orders
        A_state_dim = result.A.shape[0]

        # At minimum, should have states for input dynamics and noise dynamics
        expected_min_states = max(config.nb, config.nd, config.nc, config.nf)
        assert A_state_dim >= expected_min_states
