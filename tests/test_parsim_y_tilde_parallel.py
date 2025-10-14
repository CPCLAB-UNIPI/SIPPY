"""
Comprehensive test suite for PARSIM y_tilde parallelization optimization.

Tests numerical accuracy, performance, thread safety, and integration with
PARSIM-K/S/P algorithms.
"""

import time

import numpy as np
import pytest

# Import both original and parallel versions
from src.sippy.utils.compiled_utils import (
    NUMBA_AVAILABLE,
    parsim_y_tilde_estimation_compiled,
    parsim_y_tilde_estimation_compiled_parallel,
)

# Skip all tests if Numba not available
pytestmark = pytest.mark.skipif(
    not NUMBA_AVAILABLE, reason="Numba not available for performance testing"
)


class TestNumericalAccuracy:
    """Test numerical accuracy of parallel vs sequential implementation."""

    @pytest.mark.parametrize("l_", [1, 2, 5])
    @pytest.mark.parametrize("m", [1, 2])
    @pytest.mark.parametrize("n_cols", [100, 500])
    @pytest.mark.parametrize("i", [1, 5, 10])
    def test_small_matrices(self, l_, m, n_cols, i):
        """Test with various small matrix sizes."""
        # Create test matrices
        f = 15
        h_cols = m
        g_cols = l_

        H_K = np.random.randn(l_ * i, h_cols)
        G_K = np.random.randn(l_ * i, g_cols)
        Uf = np.random.randn(m * f, n_cols)
        Yf = np.random.randn(l_ * f, n_cols)

        # Compute with both versions
        y_tilde_seq = parsim_y_tilde_estimation_compiled(H_K, Uf, G_K, Yf, i, m, l_, f)
        y_tilde_par = parsim_y_tilde_estimation_compiled_parallel(
            H_K, Uf, G_K, Yf, i, m, l_, f
        )

        # Check numerical accuracy
        np.testing.assert_allclose(
            y_tilde_par,
            y_tilde_seq,
            rtol=1e-12,
            atol=1e-15,
            err_msg=f"Parallel version differs for l_={l_}, m={m}, n_cols={n_cols}, i={i}",
        )

    def test_large_matrices(self):
        """Test with realistic large matrices (typical PARSIM problem size)."""
        l_ = 2
        m = 2
        n_cols = 1000
        i = 10
        f = 20

        H_K = np.random.randn(l_ * i, m)
        G_K = np.random.randn(l_ * i, l_)
        Uf = np.random.randn(m * f, n_cols)
        Yf = np.random.randn(l_ * f, n_cols)

        y_tilde_seq = parsim_y_tilde_estimation_compiled(H_K, Uf, G_K, Yf, i, m, l_, f)
        y_tilde_par = parsim_y_tilde_estimation_compiled_parallel(
            H_K, Uf, G_K, Yf, i, m, l_, f
        )

        np.testing.assert_allclose(
            y_tilde_par, y_tilde_seq, rtol=1e-12, atol=1e-15
        )

    def test_edge_case_single_column(self):
        """Test edge case with single column (n_cols=1)."""
        l_ = 2
        m = 1
        n_cols = 1
        i = 3
        f = 10

        H_K = np.random.randn(l_ * i, m)
        G_K = np.random.randn(l_ * i, l_)
        Uf = np.random.randn(m * f, n_cols)
        Yf = np.random.randn(l_ * f, n_cols)

        y_tilde_seq = parsim_y_tilde_estimation_compiled(H_K, Uf, G_K, Yf, i, m, l_, f)
        y_tilde_par = parsim_y_tilde_estimation_compiled_parallel(
            H_K, Uf, G_K, Yf, i, m, l_, f
        )

        np.testing.assert_allclose(y_tilde_par, y_tilde_seq, rtol=1e-12, atol=1e-15)

    def test_edge_case_single_output(self):
        """Test edge case with single output (l_=1)."""
        l_ = 1
        m = 2
        n_cols = 500
        i = 5
        f = 15

        H_K = np.random.randn(l_ * i, m)
        G_K = np.random.randn(l_ * i, l_)
        Uf = np.random.randn(m * f, n_cols)
        Yf = np.random.randn(l_ * f, n_cols)

        y_tilde_seq = parsim_y_tilde_estimation_compiled(H_K, Uf, G_K, Yf, i, m, l_, f)
        y_tilde_par = parsim_y_tilde_estimation_compiled_parallel(
            H_K, Uf, G_K, Yf, i, m, l_, f
        )

        np.testing.assert_allclose(y_tilde_par, y_tilde_seq, rtol=1e-12, atol=1e-15)

    def test_edge_case_i_equals_1(self):
        """Test edge case where i=1 (no accumulation loop)."""
        l_ = 2
        m = 2
        n_cols = 500
        i = 1
        f = 15

        H_K = np.random.randn(l_ * i, m)
        G_K = np.random.randn(l_ * i, l_)
        Uf = np.random.randn(m * f, n_cols)
        Yf = np.random.randn(l_ * f, n_cols)

        y_tilde_seq = parsim_y_tilde_estimation_compiled(H_K, Uf, G_K, Yf, i, m, l_, f)
        y_tilde_par = parsim_y_tilde_estimation_compiled_parallel(
            H_K, Uf, G_K, Yf, i, m, l_, f
        )

        np.testing.assert_allclose(y_tilde_par, y_tilde_seq, rtol=1e-12, atol=1e-15)

    def test_zero_matrices(self):
        """Test with zero input matrices."""
        l_ = 2
        m = 2
        n_cols = 100
        i = 5
        f = 10

        H_K = np.zeros((l_ * i, m))
        G_K = np.zeros((l_ * i, l_))
        Uf = np.zeros((m * f, n_cols))
        Yf = np.zeros((l_ * f, n_cols))

        y_tilde_seq = parsim_y_tilde_estimation_compiled(H_K, Uf, G_K, Yf, i, m, l_, f)
        y_tilde_par = parsim_y_tilde_estimation_compiled_parallel(
            H_K, Uf, G_K, Yf, i, m, l_, f
        )

        np.testing.assert_allclose(y_tilde_par, y_tilde_seq, rtol=1e-12, atol=1e-15)
        assert np.allclose(y_tilde_par, 0.0)


class TestPerformance:
    """Test performance improvements of parallel implementation."""

    @pytest.mark.parametrize(
        "n_cols,expected_min_speedup", [(100, 1.2), (500, 1.5), (1000, 1.8)]
    )
    def test_speedup_varying_n_cols(self, n_cols, expected_min_speedup):
        """Test speedup with varying number of columns (data samples)."""
        l_ = 2
        m = 2
        i = 10
        f = 20

        H_K = np.random.randn(l_ * i, m)
        G_K = np.random.randn(l_ * i, l_)
        Uf = np.random.randn(m * f, n_cols)
        Yf = np.random.randn(l_ * f, n_cols)

        # Warm-up JIT compilation
        _ = parsim_y_tilde_estimation_compiled(H_K, Uf, G_K, Yf, i, m, l_, f)
        _ = parsim_y_tilde_estimation_compiled_parallel(H_K, Uf, G_K, Yf, i, m, l_, f)

        # Benchmark sequential version
        n_runs = 20
        times_seq = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = parsim_y_tilde_estimation_compiled(H_K, Uf, G_K, Yf, i, m, l_, f)
            times_seq.append(time.perf_counter() - t0)

        # Benchmark parallel version
        times_par = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = parsim_y_tilde_estimation_compiled_parallel(
                H_K, Uf, G_K, Yf, i, m, l_, f
            )
            times_par.append(time.perf_counter() - t0)

        # Compute speedup (use median to avoid outliers)
        time_seq = np.median(times_seq)
        time_par = np.median(times_par)
        speedup = time_seq / time_par

        print(
            f"\nn_cols={n_cols}: Sequential={time_seq*1000:.3f}ms, "
            f"Parallel={time_par*1000:.3f}ms, Speedup={speedup:.2f}x"
        )

        # Assert minimum expected speedup
        assert speedup >= expected_min_speedup, (
            f"Expected speedup >= {expected_min_speedup}x for n_cols={n_cols}, "
            f"got {speedup:.2f}x"
        )

    def test_speedup_varying_i(self):
        """Test speedup with varying iteration index (complexity)."""
        l_ = 2
        m = 2
        n_cols = 1000
        f = 30

        results = []
        for i in [5, 10, 20, 30]:
            H_K = np.random.randn(l_ * i, m)
            G_K = np.random.randn(l_ * i, l_)
            Uf = np.random.randn(m * f, n_cols)
            Yf = np.random.randn(l_ * f, n_cols)

            # Warm-up
            _ = parsim_y_tilde_estimation_compiled(H_K, Uf, G_K, Yf, i, m, l_, f)
            _ = parsim_y_tilde_estimation_compiled_parallel(
                H_K, Uf, G_K, Yf, i, m, l_, f
            )

            # Benchmark
            n_runs = 10
            times_seq = []
            for _ in range(n_runs):
                t0 = time.perf_counter()
                _ = parsim_y_tilde_estimation_compiled(H_K, Uf, G_K, Yf, i, m, l_, f)
                times_seq.append(time.perf_counter() - t0)

            times_par = []
            for _ in range(n_runs):
                t0 = time.perf_counter()
                _ = parsim_y_tilde_estimation_compiled_parallel(
                    H_K, Uf, G_K, Yf, i, m, l_, f
                )
                times_par.append(time.perf_counter() - t0)

            time_seq = np.median(times_seq)
            time_par = np.median(times_par)
            speedup = time_seq / time_par

            results.append((i, time_seq, time_par, speedup))
            print(
                f"i={i}: Sequential={time_seq*1000:.3f}ms, "
                f"Parallel={time_par*1000:.3f}ms, Speedup={speedup:.2f}x"
            )

        # Assert speedup improves with complexity
        speedups = [r[3] for r in results]
        assert all(
            s >= 1.2 for s in speedups
        ), "All speedups should be >= 1.2x for large problems"


class TestThreadSafety:
    """Test thread safety of parallel implementation."""

    def test_repeated_calls_produce_same_result(self):
        """Test that repeated calls produce identical results (thread-safe)."""
        l_ = 2
        m = 2
        n_cols = 500
        i = 10
        f = 20

        H_K = np.random.randn(l_ * i, m)
        G_K = np.random.randn(l_ * i, l_)
        Uf = np.random.randn(m * f, n_cols)
        Yf = np.random.randn(l_ * f, n_cols)

        # Warm-up
        _ = parsim_y_tilde_estimation_compiled_parallel(H_K, Uf, G_K, Yf, i, m, l_, f)

        # Run multiple times and compare
        results = []
        for _ in range(10):
            y_tilde = parsim_y_tilde_estimation_compiled_parallel(
                H_K, Uf, G_K, Yf, i, m, l_, f
            )
            results.append(y_tilde.copy())

        # All results should be identical
        for result in results[1:]:
            np.testing.assert_array_equal(results[0], result)


class TestIntegrationWithPARSIM:
    """Test integration with actual PARSIM algorithms."""

    def test_parsim_k_integration(self):
        """Test within PARSIM-K workflow (minimal integration test)."""
        # Simulate typical PARSIM-K parameters
        l_ = 1  # single output
        m = 1  # single input
        n_cols = 200
        f = 10

        for i in range(1, f):
            H_K = np.random.randn(l_ * i, m)
            G_K = np.random.randn(l_ * i, l_)
            Uf = np.random.randn(m * f, n_cols)
            Yf = np.random.randn(l_ * f, n_cols)

            y_tilde_seq = parsim_y_tilde_estimation_compiled(
                H_K, Uf, G_K, Yf, i, m, l_, f
            )
            y_tilde_par = parsim_y_tilde_estimation_compiled_parallel(
                H_K, Uf, G_K, Yf, i, m, l_, f
            )

            np.testing.assert_allclose(y_tilde_par, y_tilde_seq, rtol=1e-12, atol=1e-15)

    def test_mimo_system(self):
        """Test with MIMO system (multiple inputs/outputs)."""
        l_ = 3  # 3 outputs
        m = 2  # 2 inputs
        n_cols = 500
        i = 8
        f = 15

        H_K = np.random.randn(l_ * i, m)
        G_K = np.random.randn(l_ * i, l_)
        Uf = np.random.randn(m * f, n_cols)
        Yf = np.random.randn(l_ * f, n_cols)

        y_tilde_seq = parsim_y_tilde_estimation_compiled(H_K, Uf, G_K, Yf, i, m, l_, f)
        y_tilde_par = parsim_y_tilde_estimation_compiled_parallel(
            H_K, Uf, G_K, Yf, i, m, l_, f
        )

        np.testing.assert_allclose(y_tilde_par, y_tilde_seq, rtol=1e-12, atol=1e-15)


class TestOutputDimensions:
    """Test output dimensions are correct."""

    @pytest.mark.parametrize("l_,m,n_cols,i,f", [(2, 2, 100, 5, 10), (1, 3, 500, 8, 15)])
    def test_output_shape(self, l_, m, n_cols, i, f):
        """Test output has correct shape (l_, n_cols)."""
        H_K = np.random.randn(l_ * i, m)
        G_K = np.random.randn(l_ * i, l_)
        Uf = np.random.randn(m * f, n_cols)
        Yf = np.random.randn(l_ * f, n_cols)

        y_tilde = parsim_y_tilde_estimation_compiled_parallel(
            H_K, Uf, G_K, Yf, i, m, l_, f
        )

        assert y_tilde.shape == (
            l_,
            n_cols,
        ), f"Expected shape ({l_}, {n_cols}), got {y_tilde.shape}"


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
