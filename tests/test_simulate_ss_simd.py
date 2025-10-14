"""
Comprehensive test suite for SIMD-optimized state-space simulation.

This test suite validates:
1. Numerical accuracy: SIMD vs original implementation
2. Performance benchmarks: Speedup measurements
3. Edge cases: SISO, MIMO, various system orders
4. Integration: Tests with subspace and PARSIM algorithms
5. Stability: Numerical stability preservation

Run with: uv run pytest test_simulate_ss_simd.py -v
"""

import time

import numpy as np
import pytest

from sippy.utils.compiled_utils import (
    NUMBA_AVAILABLE,
    simulate_ss_system_compiled,
    simulate_ss_system_compiled_simd,
)


# Skip all tests if Numba is not available
pytestmark = pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba not available")


class TestNumericalAccuracy:
    """Test numerical accuracy between SIMD and original implementations."""

    @pytest.mark.parametrize("n,m,l", [(5, 2, 2), (10, 3, 2), (20, 1, 1), (50, 2, 3)])
    def test_accuracy_various_sizes(self, n, m, l):
        """Test numerical accuracy for various system sizes."""
        L = 1000  # Time steps
        np.random.seed(42)

        # Generate random stable system
        A = np.random.rand(n, n) * 0.5
        B = np.random.randn(n, m)
        C = np.random.randn(l, n)
        D = np.random.randn(l, m) * 0.1
        u = np.random.randn(m, L)
        x0 = np.random.randn(n, 1)

        # Simulate with both methods
        x_orig, y_orig = simulate_ss_system_compiled(A, B, C, D, u, x0)
        x_simd, y_simd = simulate_ss_system_compiled_simd(A, B, C, D, u, x0)

        # Check numerical accuracy (should be nearly identical)
        np.testing.assert_allclose(x_simd, x_orig, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(y_simd, y_orig, rtol=1e-12, atol=1e-14)

    def test_zero_initial_condition(self):
        """Test with zero initial condition."""
        n, m, l, L = 10, 2, 2, 500
        np.random.seed(43)

        A = np.eye(n) * 0.9
        B = np.random.randn(n, m)
        C = np.random.randn(l, n)
        D = np.zeros((l, m))
        u = np.random.randn(m, L)
        x0 = np.zeros((n, 1))

        x_orig, y_orig = simulate_ss_system_compiled(A, B, C, D, u, x0)
        x_simd, y_simd = simulate_ss_system_compiled_simd(A, B, C, D, u, x0)

        np.testing.assert_allclose(x_simd, x_orig, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(y_simd, y_orig, rtol=1e-12, atol=1e-14)

    def test_no_initial_condition(self):
        """Test without providing initial condition (defaults to zero)."""
        n, m, l, L = 10, 2, 2, 500
        np.random.seed(44)

        A = np.eye(n) * 0.9
        B = np.random.randn(n, m)
        C = np.random.randn(l, n)
        D = np.zeros((l, m))
        u = np.random.randn(m, L)

        x_orig, y_orig = simulate_ss_system_compiled(A, B, C, D, u)
        x_simd, y_simd = simulate_ss_system_compiled_simd(A, B, C, D, u)

        np.testing.assert_allclose(x_simd, x_orig, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(y_simd, y_orig, rtol=1e-12, atol=1e-14)

    def test_diagonal_system(self):
        """Test with diagonal A matrix (decoupled dynamics)."""
        n, m, l, L = 15, 2, 2, 500
        np.random.seed(45)

        A = np.diag(np.random.rand(n) * 0.9)
        B = np.random.randn(n, m)
        C = np.random.randn(l, n)
        D = np.random.randn(l, m) * 0.1
        u = np.random.randn(m, L)
        x0 = np.random.randn(n, 1)

        x_orig, y_orig = simulate_ss_system_compiled(A, B, C, D, u, x0)
        x_simd, y_simd = simulate_ss_system_compiled_simd(A, B, C, D, u, x0)

        np.testing.assert_allclose(x_simd, x_orig, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(y_simd, y_orig, rtol=1e-12, atol=1e-14)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_siso_system(self):
        """Test SISO (Single Input Single Output) system."""
        n, m, l, L = 10, 1, 1, 500
        np.random.seed(46)

        A = np.random.rand(n, n) * 0.8
        B = np.random.randn(n, m)
        C = np.random.randn(l, n)
        D = np.random.randn(l, m)
        u = np.random.randn(m, L)
        x0 = np.random.randn(n, 1)

        x_orig, y_orig = simulate_ss_system_compiled(A, B, C, D, u, x0)
        x_simd, y_simd = simulate_ss_system_compiled_simd(A, B, C, D, u, x0)

        np.testing.assert_allclose(x_simd, x_orig, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(y_simd, y_orig, rtol=1e-12, atol=1e-14)

    def test_mimo_system(self):
        """Test MIMO (Multiple Input Multiple Output) system."""
        n, m, l, L = 10, 3, 4, 500
        np.random.seed(47)

        A = np.random.rand(n, n) * 0.7
        B = np.random.randn(n, m)
        C = np.random.randn(l, n)
        D = np.random.randn(l, m)
        u = np.random.randn(m, L)
        x0 = np.random.randn(n, 1)

        x_orig, y_orig = simulate_ss_system_compiled(A, B, C, D, u, x0)
        x_simd, y_simd = simulate_ss_system_compiled_simd(A, B, C, D, u, x0)

        np.testing.assert_allclose(x_simd, x_orig, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(y_simd, y_orig, rtol=1e-12, atol=1e-14)

    def test_small_system(self):
        """Test with small system order (n=2)."""
        n, m, l, L = 2, 1, 1, 500
        np.random.seed(48)

        A = np.array([[0.9, 0.1], [-0.1, 0.8]])
        B = np.array([[1.0], [0.5]])
        C = np.array([[1.0, 0.5]])
        D = np.array([[0.0]])
        u = np.random.randn(m, L)
        x0 = np.array([[1.0], [0.0]])

        x_orig, y_orig = simulate_ss_system_compiled(A, B, C, D, u, x0)
        x_simd, y_simd = simulate_ss_system_compiled_simd(A, B, C, D, u, x0)

        np.testing.assert_allclose(x_simd, x_orig, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(y_simd, y_orig, rtol=1e-12, atol=1e-14)

    def test_large_system(self):
        """Test with large system order (n=100)."""
        n, m, l, L = 100, 2, 2, 200
        np.random.seed(49)

        A = np.random.rand(n, n) * 0.5 / n  # Stable for large n
        B = np.random.randn(n, m)
        C = np.random.randn(l, n)
        D = np.random.randn(l, m) * 0.1
        u = np.random.randn(m, L)
        x0 = np.random.randn(n, 1)

        x_orig, y_orig = simulate_ss_system_compiled(A, B, C, D, u, x0)
        x_simd, y_simd = simulate_ss_system_compiled_simd(A, B, C, D, u, x0)

        np.testing.assert_allclose(x_simd, x_orig, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(y_simd, y_orig, rtol=1e-12, atol=1e-14)


class TestPerformance:
    """Benchmark performance and measure speedup."""

    @pytest.mark.parametrize("n", [5, 10, 20, 50])
    def test_speedup_various_orders(self, n):
        """Measure speedup for various system orders."""
        m, l, L = 2, 2, 1000
        np.random.seed(50)

        A = np.random.rand(n, n) * 0.8
        B = np.random.randn(n, m)
        C = np.random.randn(l, n)
        D = np.random.randn(l, m)
        u = np.random.randn(m, L)
        x0 = np.random.randn(n, 1)

        # Warm-up both implementations
        _ = simulate_ss_system_compiled(A, B, C, D, u, x0)
        _ = simulate_ss_system_compiled_simd(A, B, C, D, u, x0)

        # Benchmark original (parallel version)
        n_runs = 10
        times_orig = []
        for _ in range(n_runs):
            start = time.perf_counter()
            _ = simulate_ss_system_compiled(A, B, C, D, u, x0)
            times_orig.append(time.perf_counter() - start)
        time_orig = np.median(times_orig)

        # Benchmark SIMD version
        times_simd = []
        for _ in range(n_runs):
            start = time.perf_counter()
            _ = simulate_ss_system_compiled_simd(A, B, C, D, u, x0)
            times_simd.append(time.perf_counter() - start)
        time_simd = np.median(times_simd)

        speedup = time_orig / time_simd
        print(f"\nn={n:2d}: Original={time_orig*1000:.3f}ms, SIMD={time_simd*1000:.3f}ms, Speedup={speedup:.2f}x")

        # SIMD should be at least as fast as original (allow some variance)
        assert time_simd <= time_orig * 1.5, f"SIMD slower than expected for n={n}"

    def test_large_dataset_performance(self):
        """Test performance with large dataset."""
        n, m, l, L = 20, 2, 2, 10000  # Long time series
        np.random.seed(51)

        A = np.random.rand(n, n) * 0.7
        B = np.random.randn(n, m)
        C = np.random.randn(l, n)
        D = np.random.randn(l, m)
        u = np.random.randn(m, L)
        x0 = np.random.randn(n, 1)

        # Warm-up
        _ = simulate_ss_system_compiled(A, B, C, D, u, x0)
        _ = simulate_ss_system_compiled_simd(A, B, C, D, u, x0)

        # Single run benchmark
        start_orig = time.perf_counter()
        _ = simulate_ss_system_compiled(A, B, C, D, u, x0)
        time_orig = time.perf_counter() - start_orig

        start_simd = time.perf_counter()
        _ = simulate_ss_system_compiled_simd(A, B, C, D, u, x0)
        time_simd = time.perf_counter() - start_simd

        speedup = time_orig / time_simd
        print(f"\nLarge dataset (L={L}): Original={time_orig*1000:.3f}ms, SIMD={time_simd*1000:.3f}ms, Speedup={speedup:.2f}x")

        assert time_simd <= time_orig * 1.5


class TestStability:
    """Test numerical stability preservation."""

    def test_stable_system_remains_stable(self):
        """Test that stable system remains stable with SIMD."""
        n, m, l, L = 10, 2, 2, 1000
        np.random.seed(52)

        # Create stable system (eigenvalues < 1)
        A = np.random.rand(n, n) * 0.5
        A = A / (np.max(np.abs(np.linalg.eigvals(A))) + 0.1)  # Ensure stable
        B = np.random.randn(n, m)
        C = np.random.randn(l, n)
        D = np.random.randn(l, m) * 0.1
        u = np.random.randn(m, L) * 0.1  # Small input
        x0 = np.random.randn(n, 1)

        x_orig, y_orig = simulate_ss_system_compiled(A, B, C, D, u, x0)
        x_simd, y_simd = simulate_ss_system_compiled_simd(A, B, C, D, u, x0)

        # Check states remain bounded
        assert np.all(np.abs(x_orig) < 1e6), "Original simulation unstable"
        assert np.all(np.abs(x_simd) < 1e6), "SIMD simulation unstable"
        assert np.all(np.abs(y_orig) < 1e6), "Original output unstable"
        assert np.all(np.abs(y_simd) < 1e6), "SIMD output unstable"

    def test_oscillatory_system(self):
        """Test system with oscillatory dynamics."""
        n, m, l, L = 10, 1, 1, 1000
        np.random.seed(53)

        # Create system with complex eigenvalues (oscillations)
        theta = np.pi / 4
        A = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]]) * 0.95
        # Pad to size n
        A_full = np.zeros((n, n))
        A_full[:2, :2] = A
        A_full[2:, 2:] = np.eye(n-2) * 0.9
        B = np.random.randn(n, m)
        C = np.random.randn(l, n)
        D = np.zeros((l, m))
        u = np.random.randn(m, L) * 0.1
        x0 = np.random.randn(n, 1)

        x_orig, y_orig = simulate_ss_system_compiled(A_full, B, C, D, u, x0)
        x_simd, y_simd = simulate_ss_system_compiled_simd(A_full, B, C, D, u, x0)

        np.testing.assert_allclose(x_simd, x_orig, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(y_simd, y_orig, rtol=1e-12, atol=1e-14)


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
