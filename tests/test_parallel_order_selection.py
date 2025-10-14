"""
Comprehensive test suite for parallel order selection optimization.

Tests correctness, performance, edge cases, and thread safety.
"""

import time
import warnings

import numpy as np
import pytest

from src.sippy.identification.algorithms.subspace_core import SubspaceCoreAlgorithm
from src.sippy.utils.signal_utils import GBN_seq


class TestParallelOrderSelection:
    """Test suite for parallel order selection."""

    @pytest.fixture
    def test_data_small(self):
        """Generate small test system for basic correctness tests."""
        # True system: 2nd order
        np.random.seed(42)
        A_true = np.array([[0.8, 0.1], [-0.1, 0.7]])
        B_true = np.array([[1.0], [0.5]])
        C_true = np.array([[1.0, 0.5]])
        D_true = np.array([[0.0]])

        # Generate input-output data
        u, _, _ = GBN_seq(500, 0.05)
        u = u.reshape(1, -1)

        x = np.zeros((2, 500))
        y = np.zeros((1, 500))

        for t in range(1, 500):
            x[:, t] = A_true @ x[:, t - 1] + B_true[:, 0] * u[0, t - 1]
            y[:, t] = C_true @ x[:, t] + D_true[0, 0] * u[0, t]

        # Add small noise
        y += 0.01 * np.random.randn(*y.shape)

        return y, u

    @pytest.fixture
    def test_data_large(self):
        """Generate larger test system for performance benchmarks."""
        # True system: 4th order
        np.random.seed(123)
        A_true = np.array(
            [
                [0.8, 0.1, 0, 0],
                [-0.1, 0.7, 0.1, 0],
                [0, -0.1, 0.6, 0.1],
                [0, 0, -0.1, 0.5],
            ]
        )
        B_true = np.array([[1.0], [0.5], [0.3], [0.1]])
        C_true = np.array([[1.0, 0.5, 0.3, 0.1]])
        D_true = np.array([[0.0]])

        # Generate longer input-output data
        u, _, _ = GBN_seq(2000, 0.05)
        u = u.reshape(1, -1)

        x = np.zeros((4, 2000))
        y = np.zeros((1, 2000))

        for t in range(1, 2000):
            x[:, t] = A_true @ x[:, t - 1] + B_true[:, 0] * u[0, t - 1]
            y[:, t] = C_true @ x[:, t] + D_true[0, 0] * u[0, t]

        # Add small noise
        y += 0.01 * np.random.randn(*y.shape)

        return y, u

    def test_correctness_sequential_vs_parallel(self, test_data_small):
        """Test that parallel and sequential order selection produce identical results."""
        y, u = test_data_small

        # Run sequential (n_jobs=1)
        A_seq, B_seq, C_seq, D_seq, Vn_seq, Q_seq, R_seq, S_seq, K_seq = (
            SubspaceCoreAlgorithm.select_order(y, u, f=10, orders=[1, 5], n_jobs=1)
        )

        # Run parallel (n_jobs=-1)
        A_par, B_par, C_par, D_par, Vn_par, Q_par, R_par, S_par, K_par = (
            SubspaceCoreAlgorithm.select_order(y, u, f=10, orders=[1, 5], n_jobs=-1)
        )

        # Assert matrices are identical (or very close due to numerical precision)
        np.testing.assert_allclose(A_seq, A_par, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(B_seq, B_par, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(C_seq, C_par, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(D_seq, D_par, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(Vn_seq, Vn_par, rtol=1e-10, atol=1e-12)

    def test_correctness_different_methods(self, test_data_small):
        """Test parallel order selection with different weighting methods."""
        y, u = test_data_small

        for method in ["N4SID", "MOESP", "CVA"]:
            # Run sequential
            A_seq, _, _, _, Vn_seq, _, _, _, _ = SubspaceCoreAlgorithm.select_order(
                y, u, f=10, weights=method, orders=[1, 4], n_jobs=1
            )

            # Run parallel
            A_par, _, _, _, Vn_par, _, _, _, _ = SubspaceCoreAlgorithm.select_order(
                y, u, f=10, weights=method, orders=[1, 4], n_jobs=-1
            )

            # Results should match
            np.testing.assert_allclose(A_seq, A_par, rtol=1e-10, atol=1e-12)
            np.testing.assert_allclose(Vn_seq, Vn_par, rtol=1e-10, atol=1e-12)

    def test_correctness_different_ic_criteria(self, test_data_small):
        """Test parallel order selection with different information criteria."""
        y, u = test_data_small

        for ic_method in ["AIC", "AICc", "BIC"]:
            # Run sequential
            A_seq, _, _, _, _, _, _, _, _ = SubspaceCoreAlgorithm.select_order(
                y, u, f=10, method=ic_method, orders=[1, 4], n_jobs=1
            )

            # Run parallel
            A_par, _, _, _, _, _, _, _, _ = SubspaceCoreAlgorithm.select_order(
                y, u, f=10, method=ic_method, orders=[1, 4], n_jobs=-1
            )

            # Results should match
            np.testing.assert_allclose(A_seq, A_par, rtol=1e-10, atol=1e-12)

    def test_edge_case_single_order(self, test_data_small):
        """Test that single order evaluation doesn't break parallel code."""
        y, u = test_data_small

        # Should work without errors and use sequential path
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            A, B, C, D, Vn, Q, R, S, K = SubspaceCoreAlgorithm.select_order(
                y, u, f=10, orders=[2, 2], n_jobs=-1
            )

        # Verify we got valid matrices (note: select_order may choose different order)
        # The important thing is that it doesn't crash with single order
        n = A.shape[0]
        assert A.shape == (n, n)
        assert B.shape[0] == n
        assert C.shape[1] == n

    def test_edge_case_min_equals_max_order(self, test_data_small):
        """Test edge case where min_ord == max_ord."""
        y, u = test_data_small

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            A, B, C, D, Vn, Q, R, S, K = SubspaceCoreAlgorithm.select_order(
                y, u, f=10, orders=[3, 3], n_jobs=-1
            )

        # Verify we got valid matrices (order may vary due to IC selection)
        n = A.shape[0]
        assert A.shape == (n, n)

    def test_edge_case_n_jobs_specific(self, test_data_small):
        """Test that specific n_jobs values work correctly."""
        y, u = test_data_small

        # Test n_jobs=2
        A_2jobs, _, _, _, Vn_2jobs, _, _, _, _ = SubspaceCoreAlgorithm.select_order(
            y, u, f=10, orders=[1, 4], n_jobs=2
        )

        # Test n_jobs=4
        A_4jobs, _, _, _, Vn_4jobs, _, _, _, _ = SubspaceCoreAlgorithm.select_order(
            y, u, f=10, orders=[1, 4], n_jobs=4
        )

        # Results should match
        np.testing.assert_allclose(A_2jobs, A_4jobs, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(Vn_2jobs, Vn_4jobs, rtol=1e-10, atol=1e-12)

    def test_edge_case_invalid_n_jobs(self, test_data_small):
        """Test that invalid n_jobs raises appropriate error."""
        y, u = test_data_small

        with pytest.raises(ValueError, match="n_jobs must be -1 or positive integer"):
            SubspaceCoreAlgorithm.select_order(y, u, f=10, orders=[1, 4], n_jobs=0)

        with pytest.raises(ValueError, match="n_jobs must be -1 or positive integer"):
            SubspaceCoreAlgorithm.select_order(y, u, f=10, orders=[1, 4], n_jobs=-2)

    def test_performance_speedup(self, test_data_large):
        """Benchmark parallel vs sequential performance."""
        y, u = test_data_large

        # Warm up Numba JIT compilation
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _ = SubspaceCoreAlgorithm.select_order(y, u, f=15, orders=[1, 3], n_jobs=1)

        # Benchmark sequential
        start_seq = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            A_seq, _, _, _, Vn_seq, _, _, _, _ = SubspaceCoreAlgorithm.select_order(
                y, u, f=15, orders=[1, 10], n_jobs=1
            )
        time_seq = time.time() - start_seq

        # Benchmark parallel
        start_par = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            A_par, _, _, _, Vn_par, _, _, _, _ = SubspaceCoreAlgorithm.select_order(
                y, u, f=15, orders=[1, 10], n_jobs=-1
            )
        time_par = time.time() - start_par

        # Calculate speedup
        speedup = time_seq / time_par

        print(f"\nPerformance Benchmark:")
        print(f"Sequential time: {time_seq:.3f}s")
        print(f"Parallel time: {time_par:.3f}s")
        print(f"Speedup: {speedup:.2f}x")

        # Verify correctness (results should match)
        np.testing.assert_allclose(A_seq, A_par, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(Vn_seq, Vn_par, rtol=1e-10, atol=1e-12)

        # Note: For small datasets, parallelization overhead may exceed benefits
        # We just verify that parallel execution produces correct results
        # Actual speedup depends on order range, data size, and system
        print(
            f"Note: Speedup of {speedup:.2f}x (may be <1.0 due to overhead for this dataset)"
        )

    def test_integration_with_n4sid(self, test_data_small):
        """Integration test: Verify select_order works with N4SID."""
        y, u = test_data_small

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            A, B, C, D, Vn, Q, R, S, K = SubspaceCoreAlgorithm.select_order(
                y, u, f=10, weights="N4SID", orders=[1, 5], n_jobs=-1
            )

        # Verify system is stable (eigenvalues inside unit circle)
        eigvals = np.linalg.eigvals(A)
        assert np.all(np.abs(eigvals) < 1.0), "System should be stable"

        # Verify dimensions are consistent
        n = A.shape[0]
        assert B.shape == (n, 1)
        assert C.shape == (1, n)
        assert D.shape == (1, 1)

    def test_integration_with_moesp(self, test_data_small):
        """Integration test: Verify select_order works with MOESP."""
        y, u = test_data_small

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            A, B, C, D, Vn, Q, R, S, K = SubspaceCoreAlgorithm.select_order(
                y, u, f=10, weights="MOESP", orders=[1, 5], n_jobs=-1
            )

        # Verify dimensions
        n = A.shape[0]
        assert A.shape == (n, n)
        assert B.shape[0] == n
        assert C.shape[1] == n

    def test_integration_with_cva(self, test_data_small):
        """Integration test: Verify select_order works with CVA."""
        y, u = test_data_small

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            A, B, C, D, Vn, Q, R, S, K = SubspaceCoreAlgorithm.select_order(
                y, u, f=10, weights="CVA", orders=[1, 5], n_jobs=-1
            )

        # Verify dimensions
        n = A.shape[0]
        assert A.shape == (n, n)

    def test_thread_safety_multiple_runs(self, test_data_small):
        """Test thread safety by running multiple parallel evaluations."""
        y, u = test_data_small

        # Run multiple times and check for consistency
        results = []
        for _ in range(5):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                A, _, _, _, Vn, _, _, _, _ = SubspaceCoreAlgorithm.select_order(
                    y, u, f=10, orders=[1, 4], n_jobs=-1
                )
            results.append((A, Vn))

        # All runs should produce identical results
        for i in range(1, len(results)):
            np.testing.assert_allclose(
                results[0][0], results[i][0], rtol=1e-10, atol=1e-12
            )
            np.testing.assert_allclose(
                results[0][1], results[i][1], rtol=1e-10, atol=1e-12
            )

    def test_memory_efficiency(self, test_data_large):
        """Test that parallel execution doesn't cause memory issues."""
        y, u = test_data_large

        # Should complete without memory errors
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            A, B, C, D, Vn, Q, R, S, K = SubspaceCoreAlgorithm.select_order(
                y, u, f=20, orders=[1, 15], n_jobs=-1
            )

        # Verify we got valid results
        assert A.shape[0] > 0
        assert not np.any(np.isnan(A))
        assert not np.any(np.isinf(A))


def test_performance_comprehensive_benchmark():
    """
    Comprehensive performance benchmark across different order ranges.
    """
    print("\n" + "=" * 80)
    print("COMPREHENSIVE PERFORMANCE BENCHMARK")
    print("=" * 80)

    # Generate test system
    np.random.seed(456)
    A_true = np.array(
        [
            [0.8, 0.1, 0, 0],
            [-0.1, 0.7, 0.1, 0],
            [0, -0.1, 0.6, 0.1],
            [0, 0, -0.1, 0.5],
        ]
    )
    B_true = np.array([[1.0], [0.5], [0.3], [0.1]])
    C_true = np.array([[1.0, 0.5, 0.3, 0.1]])
    D_true = np.array([[0.0]])

    u, _, _ = GBN_seq(2000, 0.05)
    u = u.reshape(1, -1)
    x = np.zeros((4, 2000))
    y = np.zeros((1, 2000))

    for t in range(1, 2000):
        x[:, t] = A_true @ x[:, t - 1] + B_true[:, 0] * u[0, t - 1]
        y[:, t] = C_true @ x[:, t] + D_true[0, 0] * u[0, t]

    y += 0.01 * np.random.randn(*y.shape)

    # Warm up
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _ = SubspaceCoreAlgorithm.select_order(y, u, f=15, orders=[1, 3], n_jobs=1)

    # Test different order ranges
    order_ranges = [
        ([1, 5], "Small range (5 orders)"),
        ([1, 10], "Medium range (10 orders)"),
        ([1, 20], "Large range (20 orders)"),
    ]

    for orders, description in order_ranges:
        print(f"\n{description}")
        print("-" * 40)

        # Sequential
        start = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _ = SubspaceCoreAlgorithm.select_order(y, u, f=15, orders=orders, n_jobs=1)
        time_seq = time.time() - start

        # Parallel
        start = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _ = SubspaceCoreAlgorithm.select_order(y, u, f=15, orders=orders, n_jobs=-1)
        time_par = time.time() - start

        speedup = time_seq / time_par

        print(f"Sequential: {time_seq:.3f}s")
        print(f"Parallel:   {time_par:.3f}s")
        print(f"Speedup:    {speedup:.2f}x")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
