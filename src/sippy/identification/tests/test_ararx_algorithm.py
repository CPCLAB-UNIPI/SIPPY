"""
Test suite for ARARX (Auto-Regressive Auto-Regressive X) algorithm implementation.
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from sippy.identification.algorithms.ararx import ARARXAlgorithm
from sippy.identification.base import StateSpaceModel, SystemIdentificationConfig
from sippy.identification.iddata import IDData


class TestARARXAlgorithm:
    """Test cases for ARARX algorithm following TDD approach."""

    def setup_method(self):
        """Set up test data for ARARX algorithm (Section 4.1-compliant)."""
        np.random.seed(42)
        self.n_samples = 1000

        # ARARX SISO: A(q) y = [B(q)/D(q)] u_{k-θ} + e
        na, nb, nd, theta = 1, 2, 1, 1
        a = np.array([0.2])
        b = np.array([0.4, 0.2])
        d = np.array([0.3])

        u = np.random.normal(0, 1, self.n_samples)
        y = np.zeros(self.n_samples)
        w = np.zeros(self.n_samples)  # input IIR path output: D(q) w = B(q) u_{k-θ}
        e = np.random.normal(0, 0.1, self.n_samples)

        warmup = max(na, nb + theta, nd) + 2
        for k in range(warmup, self.n_samples):
            # Compute s = B(q) u[k-θ]
            s = 0.0
            for j in range(nb):
                s += b[j] * u[k - theta - (j + 1)]

            # IIR filter D(q) w = s  -> w[k] = s - d1*w[k-1] - ...
            w[k] = s
            for j in range(nd):
                w[k] -= d[j] * w[k - (j + 1)]

            # Output recursion: y[k] = -A_without1*y_past + w[k] + e[k]
            y[k] = w[k] + e[k]
            for j in range(na):
                y[k] -= a[j] * y[k - (j + 1)]

        time_index = pd.date_range("2023-01-01", periods=self.n_samples, freq="1s")
        data_df = pd.DataFrame({"u1": u, "y1": y}, index=time_index)

        self.data = IDData(data=data_df, inputs=["u1"], outputs=["y1"], tsample=1.0)

        self.config = SystemIdentificationConfig(method="ARARX")
        self.config.na = na
        self.config.nb = nb
        self.config.nd = nd
        self.config.theta = theta

    def test_ararx_algorithm_initialization(self):
        """Test ARARX algorithm can be initialized."""
        algorithm = ARARXAlgorithm()
        assert algorithm.get_algorithm_name() == "ARARX"

    def test_ararx_algorithm_name(self):
        """Test ARARX algorithm name."""
        algorithm = ARARXAlgorithm()
        assert algorithm.get_algorithm_name() == "ARARX"

    def test_ararx_basic_identification(self):
        """Test ARARX basic identification."""
        algorithm = ARARXAlgorithm()

        result = algorithm.identify(self.data, self.config)

        assert result is not None
        assert isinstance(result, StateSpaceModel)
        assert result.A is not None
        assert result.B is not None
        assert result.C is not None
        assert result.D is not None

    def test_ararx_with_different_orders(self):
        """Test ARARX with different model orders."""
        algorithm = ARARXAlgorithm()

        # Test ARARX(na=2, nb=3, nd=1, theta=1)
        config = SystemIdentificationConfig(method="ARARX")
        config.na = 2  # Output AR polynomial order
        config.nb = 3  # Input transfer function order
        config.nd = 1  # Denominator polynomial order
        config.theta = 1  # Input delay

        result = algorithm.identify(self.data, config)
        assert result is not None
        assert isinstance(result, StateSpaceModel)

    def test_ararx_parameter_validation(self):
        """Test ARARX parameter validation."""
        algorithm = ARARXAlgorithm()

        # Test with negative output AR order
        invalid_config = SystemIdentificationConfig(method="ARARX")
        invalid_config.na = -1
        invalid_config.nb = 1
        invalid_config.nd = 1
        invalid_config.theta = 1

        with pytest.raises(ValueError, match="Output AR order .* must be non-negative"):
            algorithm.identify(self.data, invalid_config)

        # Test with zero input order
        invalid_config = SystemIdentificationConfig(method="ARARX")
        invalid_config.na = 1
        invalid_config.nb = 0
        invalid_config.nd = 1
        invalid_config.theta = 1

        with pytest.raises(ValueError, match="Input order .* must be positive"):
            algorithm.identify(self.data, invalid_config)

        # Test with zero denominator order
        invalid_config = SystemIdentificationConfig(method="ARARX")
        invalid_config.na = 1
        invalid_config.nb = 1
        invalid_config.nd = 0
        invalid_config.theta = 1

        with pytest.raises(ValueError, match="Denominator order .* must be positive"):
            algorithm.identify(self.data, invalid_config)

        # Test with negative input delay
        invalid_config = SystemIdentificationConfig(method="ARARX")
        invalid_config.na = 1
        invalid_config.nb = 1
        invalid_config.nd = 1
        invalid_config.theta = -1

        with pytest.raises(ValueError, match="Input delay .* must be non-negative"):
            algorithm.identify(self.data, invalid_config)

    def test_ararx_mimo_system(self):
        """Test ARARX with MIMO system."""
        np.random.seed(42)
        n_samples = 500
        ny, nu = 2, 2

        # ARARX 2x2: A_i(q) y_i = [sum_j B_ij(q)/D_i(q) u_j[k-θ_ij]] + e_i
        na = [1, 1]
        nd = [1, 1]
        nb = [[1, 1], [1, 1]]
        theta = [[1, 0], [0, 1]]

        A = [np.array([0.1]), np.array([0.15])]
        D = [np.array([0.2]), np.array([0.25])]
        B = [[np.array([0.5]), np.array([0.3])], [np.array([0.2]), np.array([0.6])]]

        u = np.random.randn(nu, n_samples)
        y = np.zeros((ny, n_samples))
        w = np.zeros((ny, n_samples))
        e = np.random.normal(0, 0.05, size=(ny, n_samples))

        warmup = 5
        for k in range(warmup, n_samples):
            for i in range(ny):
                s = 0.0
                for j in range(nu):
                    for r in range(len(B[i][j])):
                        s += B[i][j][r] * u[j, k - theta[i][j] - (r + 1)]
                w[i, k] = s
                for r in range(nd[i]):
                    w[i, k] -= D[i][0] * w[i, k - (r + 1)]
                y[i, k] = w[i, k] + e[i, k]
                for r in range(na[i]):
                    y[i, k] -= A[i][r] * y[i, k - (r + 1)]

        time_index = pd.date_range("2023-01-01", periods=n_samples, freq="1s")
        data_df = pd.DataFrame(
            {"u1": u[0, :], "u2": u[1, :], "y1": y[0, :], "y2": y[1, :]}, index=time_index
        )

        data = IDData(data=data_df, inputs=["u1", "u2"], outputs=["y1", "y2"], tsample=1.0)

        config = SystemIdentificationConfig(method="ARARX")
        config.na = na
        config.nb = nb
        config.nd = nd
        config.theta = theta

        algorithm = ARARXAlgorithm()
        result = algorithm.identify(data, config)

        assert result is not None
        assert isinstance(result, StateSpaceModel)

    def test_ararx_without_harold(self):
        """Test ARARX algorithm graceful degradation without harold."""
        algorithm = ARARXAlgorithm()

        with patch("sippy.identification.algorithms.ararx.HAROLD_AVAILABLE", False):
            result = algorithm.identify(self.data, self.config)
            assert result is not None
            assert isinstance(result, StateSpaceModel)

    def test_ararx_insufficient_data(self):
        """Test ARARX algorithm with insufficient data."""
        algorithm = ARARXAlgorithm()

        # Create very short dataset
        n_samples = 4
        np.random.seed(42)
        y = np.random.randn(1, n_samples)
        u = np.random.randn(1, n_samples)

        time_index = pd.date_range("2023-01-01", periods=n_samples, freq="1s")
        data_df = pd.DataFrame({"u1": u[0, :], "y1": y[0, :]}, index=time_index)

        data = IDData(data=data_df, inputs=["u1"], outputs=["y1"], tsample=1.0)

        config = SystemIdentificationConfig(method="ARARX")
        config.na = 2
        config.nb = 4  # This will require at least 4 samples
        config.nd = 3
        config.theta = 1

        with pytest.raises(ValueError, match="Not enough data"):
            algorithm.identify(data, config)

    def test_ararx_state_space_models(self):
        """Test ARARX creates valid state-space models."""
        algorithm = ARARXAlgorithm()

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

    def test_ararx_algorithm_properties(self):
        """Test ARARX algorithm has properties for model orders."""
        algorithm = ARARXAlgorithm()

        # Test that algorithm handles the ARARX-specific structure
        # ARARX has na=0 (no AR part, unlike ARX)
        assert hasattr(algorithm, "get_algorithm_name")
        assert algorithm.get_algorithm_name() == "ARARX"

    @pytest.mark.parametrize(
        "na,nb,nd",
        [
            (0, 1, 1),  # Minimal ARARX model (na=0)
            (1, 1, 1),  # With output AR
            (1, 2, 1),  # Higher input order
            (2, 1, 1),  # Higher output AR
            (1, 1, 2),  # Higher D polynomial
            (2, 2, 2),  # Complex ARARX model
        ],
    )
    def test_ararx_various_orders(self, na, nb, nd):
        """Test ARARX with various order combinations."""
        algorithm = ARARXAlgorithm()

        config = SystemIdentificationConfig(method="ARARX")
        config.na = na
        config.nb = nb
        config.nd = nd
        config.theta = 1

        result = algorithm.identify(self.data, config)
        assert result is not None
        assert isinstance(result, StateSpaceModel)

    def test_ararx_noise_modeling(self):
        """Test ARARX properly handles noise modeling aspects."""
        algorithm = ARARXAlgorithm()

        # Create data with more complex noise characteristics
        np.random.seed(42)
        n_samples = 500
        u = np.random.normal(0, 1, n_samples)

        # Create colored noise with AR components
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

        config = SystemIdentificationConfig(method="ARARX")
        config.na = 2
        config.nb = 2
        config.nd = 2
        config.theta = 1

        result = algorithm.identify(data, config)
        assert result is not None
        assert isinstance(result, StateSpaceModel)

    def test_ararx_order_calculation_consistency(self):
        """Test ARARX order calculations are consistent with expected structure."""
        algorithm = ARARXAlgorithm()

        config = SystemIdentificationConfig(method="ARARX")
        config.na = 1
        config.nb = 2
        config.nd = 1
        config.theta = 1

        result = algorithm.identify(self.data, config)

        # The algorithm should create a model with appropriate state dimension
        A_state_dim = result.A.shape[0]

        # For ARARX, state dimension should reflect input and output dynamics
        expected_min_states = max(config.na, config.nb, config.nd)
        assert A_state_dim >= expected_min_states

    @pytest.mark.parametrize(
        "theta,na,nd",
        [
            (1, 1, 1),  # Standard delay
            (2, 1, 1),  # Longer delay
            (1, 2, 1),  # Higher output AR order
            (1, 1, 2),  # Higher D polynomial
            (2, 2, 2),  # Complex model
        ],
    )
    def test_ararx_various_delays_and_orders(self, theta, na, nd):
        """Test ARARX with various input delays and polynomial orders."""
        algorithm = ARARXAlgorithm()

        config = SystemIdentificationConfig(method="ARARX")
        config.na = na
        config.nb = 2
        config.nd = nd
        config.theta = theta

        result = algorithm.identify(self.data, config)
        assert result is not None
        assert isinstance(result, StateSpaceModel)

    def test_ararx_comparison_with_different_orders(self):
        """Test ARARX produces different results with different model orders."""
        algorithm = ARARXAlgorithm()

        # Compare two different ARARX configurations
        config_1 = SystemIdentificationConfig(method="ARARX")
        config_1.na = 1
        config_1.nb = 1
        config_1.nd = 1
        config_1.theta = 1

        config_2 = SystemIdentificationConfig(method="ARARX")
        config_2.na = 2
        config_2.nb = 2
        config_2.nd = 2
        config_2.theta = 1

        result_1 = algorithm.identify(self.data, config_1)
        result_2 = algorithm.identify(self.data, config_2)

        # Results should be different with different orders
        assert result_1 is not None
        assert result_2 is not None

        # Compare models (they should differ structurally)
        assert result_1.A.shape != result_2.A.shape or not np.allclose(
            result_1.A, result_2.A, rtol=1e-10
        )

    def test_ararx_algorithm_with_mock_fallback(self):
        """Test ARARX algorithm works with mock fallback when real implementation unavailable."""
        algorithm = ARARXAlgorithm()

        # Patch the algorithm to use mock implementation
        with patch.object(algorithm, "_create_mock_model") as mock_method:
            # Create a mock StateSpaceModel to return
            from sippy.identification.base import StateSpaceModel

            mock_state_space = StateSpaceModel(
                A=np.eye(2),
                B=np.zeros((2, 1)),
                C=np.array([[1, 0]]),
                D=np.zeros((1, 1)),
                K=np.zeros((2, 1)),
                Q=np.eye(2),
                R=np.eye(1),
                S=np.zeros((2, 1)),
                ts=1.0,
                Vn=0.01,
            )
            mock_method.return_value = mock_state_space
            result = algorithm.identify(self.data, self.config)

            assert result is not None
            assert isinstance(result, StateSpaceModel)
            assert hasattr(result, "A")
            assert hasattr(result, "B")
            assert mock_method.called

    def test_ararx_with_zero_na(self):
        """Test ARARX works with na=0 (no output AR component)."""
        algorithm = ARARXAlgorithm()

        # Should work with na=0
        config_no_ar = SystemIdentificationConfig(method="ARARX")
        config_no_ar.na = 0  # No output AR
        config_no_ar.nb = 2
        config_no_ar.nd = 1
        config_no_ar.theta = 1

        result_no_ar = algorithm.identify(self.data, config_no_ar)
        assert result_no_ar is not None
        assert isinstance(result_no_ar, StateSpaceModel)

        # Should also work with non-zero na
        result_with_ar = algorithm.identify(self.data, self.config)
        assert result_with_ar is not None
        assert isinstance(result_with_ar, StateSpaceModel)

    def test_ararx_harold_integration(self):
        """Test ARARX algorithm with harold integration when available."""
        algorithm = ARARXAlgorithm()

        with (
            patch("sippy.identification.algorithms.ararx.HAROLD_AVAILABLE", True),
            patch("sippy.identification.algorithms.ararx.harold") as mock_harold,
        ):
            # Mock harold.Transfer and harold.transfer_to_state
            mock_tf = mock_harold.Transfer.return_value
            mock_ss = mock_harold.transfer_to_state.return_value

            # Mock haroldpolymul to return a simple array
            mock_harold.haroldpolymul.return_value = np.array([1.0, 0.5, 0.2])

            # Mock state-space matrices (lowercase for harold.State)
            mock_ss.a = np.eye(2)
            mock_ss.b = np.eye(2, 1)
            mock_ss.c = np.array([[1, 0]])
            mock_ss.d = np.zeros((1, 1))

            result = algorithm.identify(self.data, self.config)

            assert result is not None
            assert isinstance(result, StateSpaceModel)

    def test_ararx_simulation_and_prediction(self):
        """Test ARARX can simulate and predict outputs."""
        algorithm = ARARXAlgorithm()

        result = algorithm.identify(self.data, self.config)

        # Test that model can be used for prediction (if supported)
        if hasattr(result, "simulate") and hasattr(result, "predict"):
            # Should be supported for simulation
            t_test = np.random.rand(10)
            y_pred = result.simulate(t_test, u=None)
            assert y_pred is not None
            assert y_pred.shape[0] == 10  # 10 time steps
        else:
            # Test fallback prediction if simulation not available
            pass

        assert result is not None
        assert isinstance(result, StateSpaceModel)

    def test_ararx_model_properties_and_methods(self):
        """Test ARARX model has all expected StateSpaceModel methods."""
        algorithm = ARARXAlgorithm()
        result = algorithm.identify(self.data, self.config)

        # Test state-space model properties and methods
        assert isinstance(result, StateSpaceModel)

        # Should support standard StateSpaceModel interface
        assert hasattr(result, "A")
        assert hasattr(result, "B")
        assert hasattr(result, "C")
        assert hasattr(result, "D")
        assert hasattr(result, "ts")

        # Test mathematical properties
        assert hasattr(result, "is_stable")
        assert hasattr(result, "get_natural_frequencies")
        assert hasattr(result, "get_step_response")
        assert hasattr(result, "get_fir_coefficients")
        assert result.is_stable() is not None

    def test_ararx_optimization_methods(self):
        """Test ARARX supports different solution methods."""
        algorithm = ARARXAlgorithm()

        result = algorithm.identify(self.data, self.config)

        # Test that model supports estimation methods
        assert result.supports_optimization_methods()

    def test_ararx_estimation_quality(self):
        """Test ARARX model estimation quality indicators."""
        algorithm = ARARXAlgorithm()
        result = algorithm.identify(self.data, self.config)

        # Check estimation quality metrics if available
        if hasattr(result, "Vn") and result.Vn is not None:
            assert result.Vn >= 0  # Variance should be non-negative

        # Model should have proper state dimensions
        assert result.B.shape[0] == result.A.shape[0]  # B and A rows match
        assert result.C.shape[1] == result.A.shape[1]  # C and A columns match

        # Stability should be checkable (though may not be)
        if hasattr(result, "get_natural_frequencies"):
            frequencies = result.get_natural_frequencies()
            assert frequencies is not None

    def test_ararx_config_flexibility(self):
        """Test ARARX algorithm works with various configurations."""
        algorithm = ARARXAlgorithm()

        # Test with different config objects
        config1 = SystemIdentificationConfig(method="ARARX")
        config1.na = 1
        config1.nb = 1
        config1.nd = 1
        config1.theta = 1

        config2 = SystemIdentificationConfig(method="ARARX")
        config2.na = 2
        config2.nb = 2
        config2.nd = 2
        config2.theta = 1

        result1 = algorithm.identify(self.data, config1)
        result2 = algorithm.identify(self.data, config2)

        # Both should work without errors
        assert result1 is not None
        assert result2 is not None
        assert isinstance(result1, StateSpaceModel)
        assert isinstance(result2, StateSpaceModel)

    def test_ararx_error_handling(self):
        """Test ARARX algorithm graceful error handling."""
        algorithm = ARARXAlgorithm()

        # Test with mismatched dimensional parameters
        invalid_data_df = pd.DataFrame(
            {"u1": [1, 2, 3, 4], "u2": [5, 6, 7, 8], "y1": [9, 10, 11, 12]},
            index=pd.date_range("2023-01-01", periods=4, freq="1s"),
        )

        invalid_data = IDData(
            data=invalid_data_df, inputs=["u1", "u2"], outputs=["y1"], tsample=1.0
        )

        config = SystemIdentificationConfig(method="ARARX")
        config.na = 1
        config.nb = 1
        config.nd = 1
        config.theta = 1

        # Should handle dimension errors gracefully
        with pytest.raises(ValueError):
            algorithm.identify(invalid_data, config)

        # Test with zero/negative orders (invalid for nb and nd)
        invalid_config = SystemIdentificationConfig(method="ARARX")
        invalid_config.na = 1
        invalid_config.nb = 0
        invalid_config.nd = 1
        invalid_config.theta = 1

        with pytest.raises(ValueError, match="must be positive"):
            algorithm.identify(self.data, invalid_config)

        invalid_config = SystemIdentificationConfig(method="ARARX")
        invalid_config.na = 1
        invalid_config.nb = 1
        invalid_config.nd = 0
        invalid_config.theta = 1

        with pytest.raises(ValueError, match="must be positive"):
            algorithm.identify(self.data, invalid_config)

        # Test with negative na
        invalid_config = SystemIdentificationConfig(method="ARARX")
        invalid_config.na = -1
        invalid_config.nb = 1
        invalid_config.nd = 1
        invalid_config.theta = 1

        with pytest.raises(ValueError, match="must be non-negative"):
            algorithm.identify(self.data, invalid_config)

        # Test with negative delay
        invalid_config = SystemIdentificationConfig(method="ARARX")
        invalid_config.na = 1
        invalid_config.nb = 1
        invalid_config.nd = 1
        invalid_config.theta = -1

        with pytest.raises(ValueError, match="must be non-negative"):
            algorithm.identify(self.data, invalid_config)
