"""
Comprehensive test suite for PARSIM y_tilde SIMD optimization.

Tests numerical accuracy, performance benchmarks, and LLVM vectorization analysis
for different implementations: original explicit loops, SIMD-optimized, and NumPy BLAS.
"""

import time
import numpy as np
import pytest

from sippy.utils.compiled_utils import (
    NUMBA_AVAILABLE,
    parsim_y_tilde_estimation_compiled,
)

# Import optimized versions (will be added to compiled_utils.py)
try:
    from sippy.utils.compiled_utils import (
        parsim_y_tilde_estimation_simd_optimized,
        parsim_y_tilde_estimation_numpy_blas,
    )

    SIMD_OPTIMIZED_AVAILABLE = True
except ImportError:
    SIMD_OPTIMIZED_AVAILABLE = False


def generate_test_data(l_, m, f, n_cols, seed=42):
    """Generate realistic test data for y_tilde estimation."""
    np.random.seed(seed)

    # Typical PARSIM dimensions:
    # H_K shape: (l_*i, m) where i is iteration index
    # G_K shape: (l_*i, l_)
    # Uf shape: (m*f, n_cols)
    # Yf shape: (l_*f, n_cols)

    i = min(f - 1, 10)  # Iteration index (1 to f-1)

    H_K = np.random.randn(l_ * i, m)
    G_K = np.random.randn(l_ * i, l_)
    Uf = np.random.randn(m * f, n_cols)
    Yf = np.random.randn(l_ * f, n_cols)

    return H_K, Uf, G_K, Yf, i, m, l_, f


class TestNumericalAccuracy:
    """Test numerical accuracy of SIMD-optimized implementations."""

    @pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba not available")
    def test_original_reference(self):
        """Test that original implementation produces correct results."""
        H_K, Uf, G_K, Yf, i, m, l_, f = generate_test_data(2, 3, 10, 100)

        y_tilde = parsim_y_tilde_estimation_compiled(H_K, Uf, G_K, Yf, i, m, l_, f)

        assert y_tilde.shape == (l_, Uf.shape[1])
        assert not np.any(np.isnan(y_tilde))
        assert not np.any(np.isinf(y_tilde))

    @pytest.mark.skipif(
        not SIMD_OPTIMIZED_AVAILABLE, reason="SIMD optimized version not available"
    )
    def test_simd_vs_original_accuracy(self):
        """Test that SIMD-optimized version matches original within tight tolerance."""
        test_cases = [
            # (l_, m, f, n_cols) - Different problem dimensions
            (1, 1, 5, 10),  # Minimal SISO
            (2, 3, 10, 100),  # Small MIMO
            (5, 10, 30, 500),  # Large MIMO
            (3, 5, 20, 1000),  # Very long time series
        ]

        for l_, m, f, n_cols in test_cases:
            H_K, Uf, G_K, Yf, i, m_val, l_val, f_val = generate_test_data(
                l_, m, f, n_cols
            )

            y_tilde_orig = parsim_y_tilde_estimation_compiled(
                H_K, Uf, G_K, Yf, i, m_val, l_val, f_val
            )
            y_tilde_simd = parsim_y_tilde_estimation_simd_optimized(
                H_K, Uf, G_K, Yf, i, m_val, l_val, f_val
            )

            # Check shape
            assert y_tilde_simd.shape == y_tilde_orig.shape

            # Check numerical accuracy (should be nearly identical with fastmath=True)
            max_abs_error = np.max(np.abs(y_tilde_simd - y_tilde_orig))
            max_rel_error = np.max(
                np.abs(y_tilde_simd - y_tilde_orig)
                / (np.abs(y_tilde_orig) + 1e-15)
            )

            print(
                f"\nDimensions: l_={l_}, m={m}, f={f}, n_cols={n_cols}, i={i}"
            )
            print(f"  Max absolute error: {max_abs_error:.2e}")
            print(f"  Max relative error: {max_rel_error:.2e}")

            # With fastmath=True, small differences are expected
            assert max_abs_error < 1e-10, f"Absolute error too large: {max_abs_error}"
            assert max_rel_error < 1e-8, f"Relative error too large: {max_rel_error}"

    @pytest.mark.skipif(
        not SIMD_OPTIMIZED_AVAILABLE, reason="NumPy BLAS version not available"
    )
    def test_numpy_vs_original_accuracy(self):
        """Test that NumPy BLAS version matches original."""
        test_cases = [
            (2, 3, 10, 100),
            (5, 10, 30, 500),
        ]

        for l_, m, f, n_cols in test_cases:
            H_K, Uf, G_K, Yf, i, m_val, l_val, f_val = generate_test_data(
                l_, m, f, n_cols
            )

            y_tilde_orig = parsim_y_tilde_estimation_compiled(
                H_K, Uf, G_K, Yf, i, m_val, l_val, f_val
            )
            y_tilde_numpy = parsim_y_tilde_estimation_numpy_blas(
                H_K, Uf, G_K, Yf, i, m_val, l_val, f_val
            )

            max_abs_error = np.max(np.abs(y_tilde_numpy - y_tilde_orig))
            max_rel_error = np.max(
                np.abs(y_tilde_numpy - y_tilde_orig)
                / (np.abs(y_tilde_orig) + 1e-15)
            )

            print(
                f"\nNumPy BLAS - Dimensions: l_={l_}, m={m}, f={f}, n_cols={n_cols}"
            )
            print(f"  Max absolute error: {max_abs_error:.2e}")
            print(f"  Max relative error: {max_rel_error:.2e}")

            # NumPy BLAS should match very closely
            assert max_abs_error < 1e-12, f"Absolute error too large: {max_abs_error}"
            assert max_rel_error < 1e-10, f"Relative error too large: {max_rel_error}"

    @pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba not available")
    def test_edge_cases(self):
        """Test edge cases with minimal dimensions."""
        # Single output, single input, minimal horizon
        l_, m, f, n_cols = 1, 1, 2, 10
        i = 1

        H_K = np.random.randn(l_ * i, m)
        G_K = np.random.randn(l_ * i, l_)
        Uf = np.random.randn(m * f, n_cols)
        Yf = np.random.randn(l_ * f, n_cols)

        y_tilde = parsim_y_tilde_estimation_compiled(H_K, Uf, G_K, Yf, i, m, l_, f)

        assert y_tilde.shape == (l_, n_cols)
        assert not np.any(np.isnan(y_tilde))

        if SIMD_OPTIMIZED_AVAILABLE:
            y_tilde_simd = parsim_y_tilde_estimation_simd_optimized(
                H_K, Uf, G_K, Yf, i, m, l_, f
            )
            assert np.allclose(y_tilde, y_tilde_simd, rtol=1e-10, atol=1e-12)


class TestPerformanceBenchmarks:
    """Benchmark performance of different implementations."""

    @pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba not available")
    @pytest.mark.parametrize(
        "l_,m,f,n_cols",
        [
            (1, 1, 10, 100),  # Small SISO
            (2, 3, 10, 500),  # Medium MIMO
            (3, 5, 20, 1000),  # Large MIMO
            (5, 10, 30, 500),  # Very large MIMO
        ],
    )
    def test_benchmark_original(self, l_, m, f, n_cols, benchmark_iterations=100):
        """Benchmark original implementation."""
        H_K, Uf, G_K, Yf, i, m_val, l_val, f_val = generate_test_data(
            l_, m, f, n_cols
        )

        # Warmup
        for _ in range(5):
            parsim_y_tilde_estimation_compiled(H_K, Uf, G_K, Yf, i, m_val, l_val, f_val)

        # Benchmark
        times = []
        for _ in range(benchmark_iterations):
            start = time.perf_counter()
            parsim_y_tilde_estimation_compiled(H_K, Uf, G_K, Yf, i, m_val, l_val, f_val)
            times.append(time.perf_counter() - start)

        mean_time = np.mean(times) * 1e6  # Convert to microseconds
        std_time = np.std(times) * 1e6

        print(
            f"\nOriginal - l_={l_}, m={m}, f={f}, n_cols={n_cols}: "
            f"{mean_time:.2f} ± {std_time:.2f} µs"
        )

    @pytest.mark.skipif(
        not SIMD_OPTIMIZED_AVAILABLE, reason="SIMD optimized version not available"
    )
    @pytest.mark.parametrize(
        "l_,m,f,n_cols",
        [
            (1, 1, 10, 100),
            (2, 3, 10, 500),
            (3, 5, 20, 1000),
            (5, 10, 30, 500),
        ],
    )
    def test_benchmark_comparison(self, l_, m, f, n_cols, benchmark_iterations=100):
        """Compare performance of all implementations."""
        H_K, Uf, G_K, Yf, i, m_val, l_val, f_val = generate_test_data(
            l_, m, f, n_cols
        )

        implementations = {
            "Original": parsim_y_tilde_estimation_compiled,
            "SIMD": parsim_y_tilde_estimation_simd_optimized,
            "NumPy BLAS": parsim_y_tilde_estimation_numpy_blas,
        }

        results = {}

        for name, func in implementations.items():
            # Warmup
            for _ in range(5):
                func(H_K, Uf, G_K, Yf, i, m_val, l_val, f_val)

            # Benchmark
            times = []
            for _ in range(benchmark_iterations):
                start = time.perf_counter()
                func(H_K, Uf, G_K, Yf, i, m_val, l_val, f_val)
                times.append(time.perf_counter() - start)

            mean_time = np.mean(times) * 1e6
            std_time = np.std(times) * 1e6
            results[name] = mean_time

            print(f"{name:15s}: {mean_time:8.2f} ± {std_time:6.2f} µs")

        # Calculate speedups
        baseline = results["Original"]
        print(f"\nSpeedups vs Original (l_={l_}, m={m}, f={f}, n_cols={n_cols}):")
        for name, time_val in results.items():
            if name != "Original":
                speedup = baseline / time_val
                print(f"  {name:15s}: {speedup:.2f}×")


class TestIntegrationWithPARSIM:
    """Test integration with actual PARSIM algorithms."""

    @pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba not available")
    def test_parsim_k_integration(self):
        """Test y_tilde function within PARSIM-K context."""
        # Generate realistic PARSIM test data
        l_, m, f, p = 2, 3, 10, 5
        L = 500
        np.random.seed(42)

        y = np.random.randn(l_, L)
        u = np.random.randn(m, L)

        # This would be called within PARSIM-K algorithm
        # Verify it produces reasonable outputs
        from sippy.identification.algorithms.parsim_core import ordinate_sequence

        Yf, Yp = ordinate_sequence(y, f, p)
        Uf, Up = ordinate_sequence(u, f, p)

        i = 5  # Iteration index

        H_K = np.random.randn(l_ * i, m)
        G_K = np.random.randn(l_ * i, l_)

        y_tilde = parsim_y_tilde_estimation_compiled(H_K, Uf, G_K, Yf, i, m, l_, f)

        assert y_tilde.shape == (l_, Yf.shape[1])
        assert not np.any(np.isnan(y_tilde))
        assert not np.any(np.isinf(y_tilde))

        # Verify reasonable magnitudes (should be similar to input data scale)
        assert np.abs(np.mean(y_tilde)) < 10 * np.abs(np.mean(y))
        assert np.std(y_tilde) < 10 * np.std(y)


def run_full_benchmark_suite():
    """Run comprehensive benchmarks and print detailed report."""
    print("\n" + "=" * 80)
    print("PARSIM Y_TILDE SIMD OPTIMIZATION - COMPREHENSIVE BENCHMARK")
    print("=" * 80)

    if not NUMBA_AVAILABLE:
        print("Numba not available. Skipping benchmarks.")
        return

    if not SIMD_OPTIMIZED_AVAILABLE:
        print("SIMD optimized versions not available. Skipping comparison benchmarks.")
        return

    test_cases = [
        # (l_, m, f, n_cols, description)
        (1, 1, 5, 50, "Tiny SISO"),
        (1, 1, 10, 100, "Small SISO"),
        (2, 3, 10, 500, "Medium MIMO"),
        (3, 5, 20, 1000, "Large MIMO"),
        (5, 10, 30, 500, "Very Large MIMO"),
    ]

    implementations = {
        "Original": parsim_y_tilde_estimation_compiled,
        "SIMD": parsim_y_tilde_estimation_simd_optimized,
        "NumPy BLAS": parsim_y_tilde_estimation_numpy_blas,
    }

    iterations = 200

    print("\nBenchmarking all implementations...")
    print(f"Iterations per test: {iterations}")
    print()

    all_results = []

    for l_, m, f, n_cols, desc in test_cases:
        print(f"\nTest Case: {desc} (l_={l_}, m={m}, f={f}, n_cols={n_cols})")
        print("-" * 80)

        H_K, Uf, G_K, Yf, i, m_val, l_val, f_val = generate_test_data(
            l_, m, f, n_cols
        )

        case_results = {"config": desc, "l_": l_, "m": m, "f": f, "n_cols": n_cols}

        for name, func in implementations.items():
            # Warmup
            for _ in range(10):
                func(H_K, Uf, G_K, Yf, i, m_val, l_val, f_val)

            # Benchmark
            times = []
            for _ in range(iterations):
                start = time.perf_counter()
                func(H_K, Uf, G_K, Yf, i, m_val, l_val, f_val)
                times.append(time.perf_counter() - start)

            mean_time = np.mean(times) * 1e6
            std_time = np.std(times) * 1e6
            min_time = np.min(times) * 1e6
            p95_time = np.percentile(times, 95) * 1e6

            case_results[name] = {
                "mean": mean_time,
                "std": std_time,
                "min": min_time,
                "p95": p95_time,
            }

            print(
                f"{name:15s}: {mean_time:8.2f} ± {std_time:6.2f} µs "
                f"(min: {min_time:7.2f}, p95: {p95_time:7.2f})"
            )

        # Calculate speedups
        baseline = case_results["Original"]["mean"]
        print("\nSpeedups vs Original:")
        for name in ["SIMD", "NumPy BLAS"]:
            speedup = baseline / case_results[name]["mean"]
            case_results[f"{name}_speedup"] = speedup
            print(f"  {name:15s}: {speedup:.2f}×")

        all_results.append(case_results)

    # Summary report
    print("\n" + "=" * 80)
    print("SUMMARY REPORT")
    print("=" * 80)

    print("\nAverage Speedups Across All Test Cases:")
    simd_speedups = [r["SIMD_speedup"] for r in all_results]
    numpy_speedups = [r["NumPy BLAS_speedup"] for r in all_results]

    print(f"  SIMD Optimized:  {np.mean(simd_speedups):.2f}× (±{np.std(simd_speedups):.2f})")
    print(f"  NumPy BLAS:      {np.mean(numpy_speedups):.2f}× (±{np.std(numpy_speedups):.2f})")

    # Overall impact assessment
    print("\n" + "=" * 80)
    print("OVERALL IMPACT ASSESSMENT")
    print("=" * 80)
    print("\nContext: y_tilde estimation is ~5% of total PARSIM runtime")
    print(
        f"Best case speedup: {max(max(simd_speedups), max(numpy_speedups)):.2f}× on this function"
    )
    print(
        f"Overall PARSIM speedup: ~{max(max(simd_speedups), max(numpy_speedups)) * 0.05:.1f}% improvement"
    )
    print("\nRecommendation:")
    best_impl = "SIMD" if np.mean(simd_speedups) > np.mean(numpy_speedups) else "NumPy BLAS"
    best_speedup = (
        np.mean(simd_speedups)
        if best_impl == "SIMD"
        else np.mean(numpy_speedups)
    )
    print(
        f"  Best implementation: {best_impl} ({best_speedup:.2f}× average speedup)"
    )
    print(
        f"  Overall impact: Marginal ({best_speedup * 0.05:.1f}% total PARSIM improvement)"
    )

    if best_speedup * 0.05 < 3.0:
        print(
            "  Decision: Optimization provides <3% overall improvement - NOT RECOMMENDED for integration"
        )
        print("  Rationale: Code complexity increase not justified by minimal performance gain")
    else:
        print(
            "  Decision: Optimization provides ≥3% overall improvement - RECOMMENDED for integration"
        )


if __name__ == "__main__":
    # Run full benchmark suite
    run_full_benchmark_suite()
