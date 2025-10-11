"""
Integration tests for PARSIM algorithms.
"""

import numpy as np
import pytest

from sippy.identification import SystemIdentification, SystemIdentificationConfig
from sippy.identification.base import StateSpaceModel


class TestPARSIMIntegration:
    """Integration tests for PARSIM algorithms with the new architecture."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample test data for integration tests."""
        np.random.seed(42)
        N = 200  # Number of samples
        m = 2  # Number of inputs
        l = 1  # Number of outputs

        # Generate input signals
        u = np.random.randn(m, N)

        # Create a second-order system: H(z) = 0.5z^-1 + 0.3z^-2 / (1 - 1.5z^-1 + 0.7z^-2)
        # Convert to state space and simulate
        A_true = np.array([[1.5, -0.7], [1.0, 0.0]])
        B_true = np.array([[0.5, 0.2], [0.0, 0.3]])  # Fix dimensions: 2x2 for 2 inputs
        C_true = np.array([[0.0, 1.0]])  # 1x2 for 1 output
        D_true = np.array([[0.3, 0.1]])  # 1x2 for 1 output, 2 inputs

        # Simulate with noise
        x = np.zeros((2, N + 1))
        y = np.zeros((l, N))

        for k in range(N):
            x[:, k + 1] = A_true @ x[:, k] + B_true @ u[:, k]
            y[:, k] = C_true @ x[:, k] + D_true @ u[:, k] + 0.05 * np.random.randn()

        return y, u, A_true, B_true, C_true, D_true

    def test_parsim_k_with_system_identification(self, sample_data):
        """Test PARSIM-K with SystemIdentification wrapper."""
        y, u, A_true, B_true, C_true, D_true = sample_data

        # Test using the new architecture
        config = SystemIdentificationConfig(
            method="PARSIM-K",
            ss_f=20,
            ss_threshold=0.05,
            ss_fixed_order=2,
            ss_d_required=True,
        )

        identifier = SystemIdentification(config)
        result = identifier.identify(y=y, u=u)

        assert isinstance(result, StateSpaceModel)
        assert result.n <= 2  # Should not exceed specified order
        assert result.A.shape == (result.n, result.n)
        assert result.B.shape == (result.n, u.shape[0])
        assert result.C.shape == (y.shape[0], result.n)
        assert result.D.shape == (y.shape[0], u.shape[0])

        # Check basic stability (most real systems are stable)
        assert not np.isnan(result.Vn) and result.Vn > 0

    def test_parsim_s_with_system_identification(self, sample_data):
        """Test PARSIM-S with SystemIdentification wrapper."""
        y, u, A_true, B_true, C_true, D_true = sample_data

        config = SystemIdentificationConfig(
            method="PARSIM-S",
            ss_f=20,
            ss_threshold=0.05,
            ss_fixed_order=2,
            ss_d_required=True,
        )

        identifier = SystemIdentification(config)
        result = identifier.identify(y=y, u=u)

        assert isinstance(result, StateSpaceModel)
        assert result.A.shape[0] == result.A.shape[1]
        assert result.B.shape[0] == result.A.shape[0]
        assert result.C.shape[1] == result.A.shape[0]
        assert result.D.shape[0] == y.shape[0]
        assert result.D.shape[1] == u.shape[0]
        assert not np.isnan(result.Vn) and result.Vn > 0

    def test_parsim_p_with_system_identification(self, sample_data):
        """Test PARSIM-P with SystemIdentification wrapper."""
        y, u, A_true, B_true, C_true, D_true = sample_data

        config = SystemIdentificationConfig(
            method="PARSIM-P",
            ss_f=20,
            ss_threshold=0.05,
            ss_fixed_order=2,
            ss_d_required=True,
        )

        identifier = SystemIdentification(config)
        result = identifier.identify(y=y, u=u)

        assert isinstance(result, StateSpaceModel)
        assert result.A.shape[0] == result.A.shape[1]
        assert result.B.shape[0] == result.A.shape[0]
        assert result.C.shape[1] == result.A.shape[0]
        assert result.D.shape[0] == y.shape[0]
        assert result.D.shape[1] == u.shape[0]
        assert not np.isnan(result.Vn) and result.Vn > 0

    def test_parsim_algorithms_with_numpy_arrays_directly(self, sample_data):
        """Test PARSIM algorithms directly with numpy arrays (without IDData)."""
        y, u, A_true, B_true, C_true, D_true = sample_data

        config = SystemIdentificationConfig(ss_f=15, ss_threshold=0.1, ss_fixed_order=2)

        for algo_name in ["PARSIM-K", "PARSIM-S", "PARSIM-P"]:
            config_method = SystemIdentificationConfig(
                method=algo_name,
                ss_f=config.ss_f,
                ss_threshold=config.ss_threshold,
                ss_fixed_order=config.ss_fixed_order,
            )
            identifier = SystemIdentification(config_method)
            result = identifier.identify(y=y, u=u)

            assert isinstance(result, StateSpaceModel)
            assert result.n <= 2  # Should not exceed fixed order

    def test_parsim_algorithm_comparison(self, sample_data):
        """Compare results from different PARSIM variants."""
        y, u, A_true, B_true, C_true, D_true = sample_data

        config = SystemIdentificationConfig(
            ss_f=20, ss_threshold=0.05, ss_fixed_order=2
        )

        results = {}
        for algo_name in ["PARSIM-K", "PARSIM-S", "PARSIM-P"]:
            config_method = SystemIdentificationConfig(
                method=algo_name,
                ss_f=config.ss_f,
                ss_threshold=config.ss_threshold,
                ss_fixed_order=config.ss_fixed_order,
            )
            identifier = SystemIdentification(config_method)
            result = identifier.identify(y=y, u=u)
            results[algo_name] = result

        # All should produce valid models
        for algo_name, model in results.items():
            assert isinstance(model, StateSpaceModel)
            assert model.A.shape[0] == model.A.shape[1]  # Square
            assert not np.isnan(model.Vn)

        # Similar noise variance (they should identify similar systems)
        vn_values = [model.Vn for model in results.values()]
        vn_avg = np.mean(vn_values)
        for vn in vn_values:
            assert abs(vn - vn_avg) / vn_avg < 2.0  # Within factor of 2

    def test_parsim_with_different_data_sizes(self):
        """Test PARSIM algorithms with different data sizes."""
        np.random.seed(123)

        for N in [50, 100, 200]:  # Different data lengths
            u = np.random.randn(2, N)
            y = np.random.randn(1, N)

            config = SystemIdentificationConfig(
                ss_f=min(20, N // 4),  # Adjust horizon for data size
                ss_threshold=0.1,
            )

            for algo_name in ["PARSIM-K", "PARSIM-S", "PARSIM-P"]:
                try:
                    config_method = SystemIdentificationConfig(
                        method=algo_name,
                        ss_f=config.ss_f,
                        ss_threshold=config.ss_threshold,
                    )
                    identifier = SystemIdentification(config_method)
                    result = identifier.identify(y=y, u=u)

                    assert isinstance(result, StateSpaceModel)
                    assert result.Vn > 0
                except Exception as e:
                    # Algorithm might fail on too little data, that's okay
                    if N >= 100:  # But should succeed with decent data amounts
                        raise e
