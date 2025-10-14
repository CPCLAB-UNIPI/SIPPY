"""
Comprehensive tests for PARSIM parallel simulation sequences.

This test suite validates:
1. Numerical correctness: parallel matches sequential to machine precision
2. Performance: benchmarks speedup vs n_simulations
3. Edge cases: small/large problems, with/without D_required
4. Integration: full PARSIM-K and PARSIM-S identification
5. Thread safety: concurrent execution produces deterministic results
6. Memory usage: monitors memory consumption during parallel execution
"""

import time
import numpy as np
import pytest

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from src.sippy.identification.algorithms.parsim_core import ParsimCoreAlgorithm


class TestParsimSimulationParallel:
    """Test suite for parallel PARSIM simulation sequences."""

    @pytest.fixture
    def simple_system(self):
        """Create a simple test system for validation."""
        np.random.seed(42)

        # System parameters
        n = 3  # order
        m = 1  # inputs
        l_ = 1  # outputs
        L = 100  # time steps

        # Create stable system matrices
        A_K = np.array([[0.5, 0.1, 0.0], [0.0, 0.6, 0.1], [0.0, 0.0, 0.7]])
        C = np.array([[1.0, 0.5, 0.2]])
        K = np.array([[0.1], [0.2], [0.1]])
        D = np.zeros((l_, m))

        # Generate test data
        u = np.random.randn(m, L)
        y = np.random.randn(l_, L)

        return {
            "A_K": A_K,
            "C": C,
            "K": K,
            "D": D,
            "u": u,
            "y": y,
            "n": n,
            "m": m,
            "l_": l_,
            "L": L,
        }

    @pytest.fixture
    def large_system(self):
        """Create a larger system for performance benchmarking."""
        np.random.seed(42)

        # Larger system to trigger parallel execution
        n = 10  # order
        m = 2  # inputs
        l_ = 2  # outputs
        L = 500  # time steps

        # Create stable system matrices
        A_K = np.random.randn(n, n) * 0.1
        np.fill_diagonal(A_K, np.random.rand(n) * 0.5 + 0.2)
        C = np.random.randn(l_, n) * 0.5
        K = np.random.randn(n, l_) * 0.1
        D = np.zeros((l_, m))

        # Generate test data
        u = np.random.randn(m, L)
        y = np.random.randn(l_, L)

        return {
            "A_K": A_K,
            "C": C,
            "K": K,
            "D": D,
            "u": u,
            "y": y,
            "n": n,
            "m": m,
            "l_": l_,
            "L": L,
        }

    def test_parsim_k_correctness_no_D(self, simple_system):
        """Test PARSIM-K parallel simulation correctness without D matrix."""
        sys = simple_system

        # Run parallel version
        y_matrix_parallel = ParsimCoreAlgorithm.simulations_sequence_k(
            sys["A_K"],
            sys["C"],
            sys["L"],
            sys["y"],
            sys["u"],
            sys["l_"],
            sys["m"],
            sys["n"],
            sys["K"],
            sys["D"],
            D_required=False,
        )

        # Force sequential version by temporarily disabling joblib
        from src.sippy.identification.algorithms import parsim_core

        joblib_available = parsim_core.JOBLIB_AVAILABLE
        parsim_core.JOBLIB_AVAILABLE = False

        y_matrix_sequential = ParsimCoreAlgorithm.simulations_sequence_k(
            sys["A_K"],
            sys["C"],
            sys["L"],
            sys["y"],
            sys["u"],
            sys["l_"],
            sys["m"],
            sys["n"],
            sys["K"],
            sys["D"],
            D_required=False,
        )

        # Restore joblib state
        parsim_core.JOBLIB_AVAILABLE = joblib_available

        # Check correctness (should match to machine precision)
        max_abs_error = np.max(np.abs(y_matrix_parallel - y_matrix_sequential))
        max_rel_error = np.max(
            np.abs(y_matrix_parallel - y_matrix_sequential)
            / (np.abs(y_matrix_sequential) + 1e-15)
        )

        print(f"\nPARSIM-K No D - Max absolute error: {max_abs_error:.3e}")
        print(f"PARSIM-K No D - Max relative error: {max_rel_error:.3e}")

        assert max_abs_error < 1e-12, f"Absolute error too large: {max_abs_error}"
        assert max_rel_error < 1e-10, f"Relative error too large: {max_rel_error}"

    def test_parsim_k_correctness_with_D(self, simple_system):
        """Test PARSIM-K parallel simulation correctness with D matrix."""
        sys = simple_system

        # Run parallel version
        y_matrix_parallel = ParsimCoreAlgorithm.simulations_sequence_k(
            sys["A_K"],
            sys["C"],
            sys["L"],
            sys["y"],
            sys["u"],
            sys["l_"],
            sys["m"],
            sys["n"],
            sys["K"],
            sys["D"],
            D_required=True,
        )

        # Force sequential version
        from src.sippy.identification.algorithms import parsim_core

        joblib_available = parsim_core.JOBLIB_AVAILABLE
        parsim_core.JOBLIB_AVAILABLE = False

        y_matrix_sequential = ParsimCoreAlgorithm.simulations_sequence_k(
            sys["A_K"],
            sys["C"],
            sys["L"],
            sys["y"],
            sys["u"],
            sys["l_"],
            sys["m"],
            sys["n"],
            sys["K"],
            sys["D"],
            D_required=True,
        )

        parsim_core.JOBLIB_AVAILABLE = joblib_available

        # Check correctness
        max_abs_error = np.max(np.abs(y_matrix_parallel - y_matrix_sequential))
        max_rel_error = np.max(
            np.abs(y_matrix_parallel - y_matrix_sequential)
            / (np.abs(y_matrix_sequential) + 1e-15)
        )

        print(f"\nPARSIM-K With D - Max absolute error: {max_abs_error:.3e}")
        print(f"PARSIM-K With D - Max relative error: {max_rel_error:.3e}")

        assert max_abs_error < 1e-12, f"Absolute error too large: {max_abs_error}"
        assert max_rel_error < 1e-10, f"Relative error too large: {max_rel_error}"

    def test_parsim_s_correctness_no_D(self, simple_system):
        """Test PARSIM-S parallel simulation correctness without D matrix."""
        sys = simple_system

        # Run parallel version
        y_matrix_parallel = ParsimCoreAlgorithm.simulations_sequence_s(
            sys["A_K"],
            sys["C"],
            sys["L"],
            sys["K"],
            sys["y"],
            sys["u"],
            sys["l_"],
            sys["m"],
            sys["n"],
            D_required=False,
        )

        # Force sequential version
        from src.sippy.identification.algorithms import parsim_core

        joblib_available = parsim_core.JOBLIB_AVAILABLE
        parsim_core.JOBLIB_AVAILABLE = False

        y_matrix_sequential = ParsimCoreAlgorithm.simulations_sequence_s(
            sys["A_K"],
            sys["C"],
            sys["L"],
            sys["K"],
            sys["y"],
            sys["u"],
            sys["l_"],
            sys["m"],
            sys["n"],
            D_required=False,
        )

        parsim_core.JOBLIB_AVAILABLE = joblib_available

        # Check correctness
        max_abs_error = np.max(np.abs(y_matrix_parallel - y_matrix_sequential))
        max_rel_error = np.max(
            np.abs(y_matrix_parallel - y_matrix_sequential)
            / (np.abs(y_matrix_sequential) + 1e-15)
        )

        print(f"\nPARSIM-S No D - Max absolute error: {max_abs_error:.3e}")
        print(f"PARSIM-S No D - Max relative error: {max_rel_error:.3e}")

        assert max_abs_error < 1e-12, f"Absolute error too large: {max_abs_error}"
        assert max_rel_error < 1e-10, f"Relative error too large: {max_rel_error}"

    def test_parsim_s_correctness_with_D(self, simple_system):
        """Test PARSIM-S parallel simulation correctness with D matrix."""
        sys = simple_system

        # Run parallel version
        y_matrix_parallel = ParsimCoreAlgorithm.simulations_sequence_s(
            sys["A_K"],
            sys["C"],
            sys["L"],
            sys["K"],
            sys["y"],
            sys["u"],
            sys["l_"],
            sys["m"],
            sys["n"],
            D_required=True,
        )

        # Force sequential version
        from src.sippy.identification.algorithms import parsim_core

        joblib_available = parsim_core.JOBLIB_AVAILABLE
        parsim_core.JOBLIB_AVAILABLE = False

        y_matrix_sequential = ParsimCoreAlgorithm.simulations_sequence_s(
            sys["A_K"],
            sys["C"],
            sys["L"],
            sys["K"],
            sys["y"],
            sys["u"],
            sys["l_"],
            sys["m"],
            sys["n"],
            D_required=True,
        )

        parsim_core.JOBLIB_AVAILABLE = joblib_available

        # Check correctness
        max_abs_error = np.max(np.abs(y_matrix_parallel - y_matrix_sequential))
        max_rel_error = np.max(
            np.abs(y_matrix_parallel - y_matrix_sequential)
            / (np.abs(y_matrix_sequential) + 1e-15)
        )

        print(f"\nPARSIM-S With D - Max absolute error: {max_abs_error:.3e}")
        print(f"PARSIM-S With D - Max relative error: {max_rel_error:.3e}")

        assert max_abs_error < 1e-12, f"Absolute error too large: {max_abs_error}"
        assert max_rel_error < 1e-10, f"Relative error too large: {max_rel_error}"

    def test_performance_parsim_k(self, large_system):
        """Benchmark PARSIM-K parallel vs sequential performance."""
        sys = large_system

        # Warm up
        _ = ParsimCoreAlgorithm.simulations_sequence_k(
            sys["A_K"],
            sys["C"],
            sys["L"],
            sys["y"],
            sys["u"],
            sys["l_"],
            sys["m"],
            sys["n"],
            sys["K"],
            sys["D"],
            D_required=False,
        )

        # Benchmark parallel version
        start_parallel = time.perf_counter()
        y_matrix_parallel = ParsimCoreAlgorithm.simulations_sequence_k(
            sys["A_K"],
            sys["C"],
            sys["L"],
            sys["y"],
            sys["u"],
            sys["l_"],
            sys["m"],
            sys["n"],
            sys["K"],
            sys["D"],
            D_required=False,
        )
        time_parallel = time.perf_counter() - start_parallel

        # Benchmark sequential version
        from src.sippy.identification.algorithms import parsim_core

        joblib_available = parsim_core.JOBLIB_AVAILABLE
        parsim_core.JOBLIB_AVAILABLE = False

        start_sequential = time.perf_counter()
        y_matrix_sequential = ParsimCoreAlgorithm.simulations_sequence_k(
            sys["A_K"],
            sys["C"],
            sys["L"],
            sys["y"],
            sys["u"],
            sys["l_"],
            sys["m"],
            sys["n"],
            sys["K"],
            sys["D"],
            D_required=False,
        )
        time_sequential = time.perf_counter() - start_sequential

        parsim_core.JOBLIB_AVAILABLE = joblib_available

        speedup = time_sequential / time_parallel
        n_simulations = sys["n"] * sys["m"] + sys["n"] * sys["l_"] + sys["n"]

        print(f"\n--- PARSIM-K Performance (n_simulations={n_simulations}) ---")
        print(f"Sequential time: {time_sequential:.4f}s")
        print(f"Parallel time:   {time_parallel:.4f}s")
        print(f"Speedup:         {speedup:.2f}x")

        # Verify correctness wasn't sacrificed
        max_error = np.max(np.abs(y_matrix_parallel - y_matrix_sequential))
        print(f"Max error:       {max_error:.3e}")

        assert max_error < 1e-12, "Parallel version changed results"
        # Performance check: expect at least 1.5x speedup for large problems
        # (conservative threshold accounting for overhead and machine variation)
        assert speedup >= 1.5, f"Expected speedup >= 1.5x, got {speedup:.2f}x"

    def test_performance_parsim_s(self, large_system):
        """Benchmark PARSIM-S parallel vs sequential performance."""
        sys = large_system

        # Warm up
        _ = ParsimCoreAlgorithm.simulations_sequence_s(
            sys["A_K"],
            sys["C"],
            sys["L"],
            sys["K"],
            sys["y"],
            sys["u"],
            sys["l_"],
            sys["m"],
            sys["n"],
            D_required=False,
        )

        # Benchmark parallel version
        start_parallel = time.perf_counter()
        y_matrix_parallel = ParsimCoreAlgorithm.simulations_sequence_s(
            sys["A_K"],
            sys["C"],
            sys["L"],
            sys["K"],
            sys["y"],
            sys["u"],
            sys["l_"],
            sys["m"],
            sys["n"],
            D_required=False,
        )
        time_parallel = time.perf_counter() - start_parallel

        # Benchmark sequential version
        from src.sippy.identification.algorithms import parsim_core

        joblib_available = parsim_core.JOBLIB_AVAILABLE
        parsim_core.JOBLIB_AVAILABLE = False

        start_sequential = time.perf_counter()
        y_matrix_sequential = ParsimCoreAlgorithm.simulations_sequence_s(
            sys["A_K"],
            sys["C"],
            sys["L"],
            sys["K"],
            sys["y"],
            sys["u"],
            sys["l_"],
            sys["m"],
            sys["n"],
            D_required=False,
        )
        time_sequential = time.perf_counter() - start_sequential

        parsim_core.JOBLIB_AVAILABLE = joblib_available

        speedup = time_sequential / time_parallel
        n_simulations = sys["n"] * sys["m"] + sys["n"]

        print(f"\n--- PARSIM-S Performance (n_simulations={n_simulations}) ---")
        print(f"Sequential time: {time_sequential:.4f}s")
        print(f"Parallel time:   {time_parallel:.4f}s")
        print(f"Speedup:         {speedup:.2f}x")

        # Verify correctness
        max_error = np.max(np.abs(y_matrix_parallel - y_matrix_sequential))
        print(f"Max error:       {max_error:.3e}")

        assert max_error < 1e-12, "Parallel version changed results"
        assert speedup >= 1.5, f"Expected speedup >= 1.5x, got {speedup:.2f}x"

    def test_edge_case_small_system(self):
        """Test parallel execution with small system (below threshold)."""
        np.random.seed(42)

        # Very small system - should use sequential fallback
        n = 1  # order
        m = 1  # inputs
        l_ = 1  # outputs
        L = 50  # time steps

        A_K = np.array([[0.5]])
        C = np.array([[1.0]])
        K = np.array([[0.1]])
        D = np.zeros((l_, m))
        u = np.random.randn(m, L)
        y = np.random.randn(l_, L)

        # Should execute without errors (likely uses sequential path)
        y_matrix = ParsimCoreAlgorithm.simulations_sequence_k(
            A_K, C, L, y, u, l_, m, n, K, D, D_required=False
        )

        # Check output shape
        n_simulations = n * m + n * l_ + n  # 1 + 1 + 1 = 3
        expected_shape = (L * l_, n_simulations)
        assert (
            y_matrix.shape == expected_shape
        ), f"Expected shape {expected_shape}, got {y_matrix.shape}"

    def test_edge_case_large_order(self):
        """Test parallel execution with large order system."""
        np.random.seed(42)

        # Large order system - should definitely use parallel
        n = 20  # large order
        m = 3  # inputs
        l_ = 2  # outputs
        L = 300  # time steps

        A_K = np.random.randn(n, n) * 0.05
        np.fill_diagonal(A_K, np.random.rand(n) * 0.3 + 0.1)
        C = np.random.randn(l_, n) * 0.3
        K = np.random.randn(n, l_) * 0.1
        D = np.zeros((l_, m))
        u = np.random.randn(m, L)
        y = np.random.randn(l_, L)

        # Should execute efficiently with parallel
        start = time.perf_counter()
        y_matrix = ParsimCoreAlgorithm.simulations_sequence_k(
            A_K, C, L, y, u, l_, m, n, K, D, D_required=True
        )
        elapsed = time.perf_counter() - start

        n_simulations = n * m + l_ * m + n * l_ + n  # 60 + 6 + 40 + 20 = 126
        print(
            f"\nLarge order system (n={n}, n_simulations={n_simulations}): {elapsed:.4f}s"
        )

        # Check output shape
        expected_shape = (L * l_, n_simulations)
        assert (
            y_matrix.shape == expected_shape
        ), f"Expected shape {expected_shape}, got {y_matrix.shape}"

    @pytest.mark.skipif(not PSUTIL_AVAILABLE, reason="psutil not available")
    def test_memory_usage(self, large_system):
        """Monitor memory usage during parallel execution."""
        import psutil

        sys = large_system
        process = psutil.Process()

        # Measure memory before
        mem_before = process.memory_info().rss / 1024**2  # MB

        # Run parallel version
        y_matrix = ParsimCoreAlgorithm.simulations_sequence_k(
            sys["A_K"],
            sys["C"],
            sys["L"],
            sys["y"],
            sys["u"],
            sys["l_"],
            sys["m"],
            sys["n"],
            sys["K"],
            sys["D"],
            D_required=False,
        )

        # Measure memory after
        mem_after = process.memory_info().rss / 1024**2  # MB
        mem_increase = mem_after - mem_before

        print(f"\n--- Memory Usage ---")
        print(f"Before:   {mem_before:.2f} MB")
        print(f"After:    {mem_after:.2f} MB")
        print(f"Increase: {mem_increase:.2f} MB")

        # Memory increase should be reasonable (< 500 MB for this test)
        assert (
            mem_increase < 500
        ), f"Memory increase too large: {mem_increase:.2f} MB"

    def test_thread_safety(self, simple_system):
        """Test that parallel execution is deterministic and thread-safe."""
        sys = simple_system

        # Run multiple times and check for determinism
        results = []
        for _ in range(5):
            y_matrix = ParsimCoreAlgorithm.simulations_sequence_k(
                sys["A_K"],
                sys["C"],
                sys["L"],
                sys["y"],
                sys["u"],
                sys["l_"],
                sys["m"],
                sys["n"],
                sys["K"],
                sys["D"],
                D_required=False,
            )
            results.append(y_matrix)

        # All results should be identical (deterministic)
        for i in range(1, len(results)):
            max_diff = np.max(np.abs(results[i] - results[0]))
            assert (
                max_diff < 1e-15
            ), f"Run {i} differs from run 0 by {max_diff} (not deterministic)"

        print("\nThread safety check passed - all runs identical")

    def test_integration_parsim_k_full(self):
        """Integration test: Full PARSIM-K identification with parallel simulations."""
        np.random.seed(42)

        # Generate realistic test data
        from src.sippy.utils.signal_utils import GBN_seq

        # System dimensions
        m = 1  # inputs
        l_ = 1  # outputs
        L = 500  # time steps

        # Generate input signal
        u, _, _ = GBN_seq(L, 0.05)
        u = u.reshape(1, -1)

        # Create true system and generate output
        A_true = np.array([[0.7, 0.2], [0.0, 0.5]])
        B_true = np.array([[1.0], [0.5]])
        C_true = np.array([[1.0, 0.5]])
        D_true = np.zeros((l_, m))

        # Simulate true system
        from src.sippy.utils.simulation_utils import simulate_ss_system

        x0 = np.zeros((2, 1))
        _, y = simulate_ss_system(A_true, B_true, C_true, D_true, u, x0=x0)

        # Add noise
        y = y + np.random.randn(*y.shape) * 0.01

        # Run PARSIM-K identification (will use parallel simulations)
        start = time.perf_counter()
        A_K, C, B_K, D, K, A, B, x0, Vn = ParsimCoreAlgorithm.parsim_k(
            y, u, f=10, p=10, threshold=0.1
        )
        elapsed = time.perf_counter() - start

        print(f"\n--- PARSIM-K Integration Test ---")
        print(f"Identification time: {elapsed:.4f}s")
        print(f"Model order: {A.shape[0]}")
        print(f"Fit variance: {Vn:.6f}")

        # Check that we got reasonable results
        assert A.shape[0] >= 1, "Model order should be at least 1"
        assert np.isfinite(Vn), "Variance should be finite"
        assert not np.any(np.isnan(A)), "A matrix should not contain NaN"
        assert not np.any(np.isnan(B)), "B matrix should not contain NaN"

    def test_integration_parsim_s_full(self):
        """Integration test: Full PARSIM-S identification with parallel simulations."""
        np.random.seed(42)

        # Generate realistic test data
        from src.sippy.utils.signal_utils import GBN_seq

        # System dimensions
        m = 1  # inputs
        l_ = 1  # outputs
        L = 500  # time steps

        # Generate input signal
        u, _, _ = GBN_seq(L, 0.05)
        u = u.reshape(1, -1)

        # Create true system and generate output
        A_true = np.array([[0.7, 0.2], [0.0, 0.5]])
        B_true = np.array([[1.0], [0.5]])
        C_true = np.array([[1.0, 0.5]])
        D_true = np.zeros((l_, m))

        # Simulate true system
        from src.sippy.utils.simulation_utils import simulate_ss_system

        x0 = np.zeros((2, 1))
        _, y = simulate_ss_system(A_true, B_true, C_true, D_true, u, x0=x0)

        # Add noise
        y = y + np.random.randn(*y.shape) * 0.01

        # Run PARSIM-S identification (will use parallel simulations)
        start = time.perf_counter()
        A_K, C, B_K, D, K, A, B, x0, Vn = ParsimCoreAlgorithm.parsim_s(
            y, u, f=10, p=10, threshold=0.1
        )
        elapsed = time.perf_counter() - start

        print(f"\n--- PARSIM-S Integration Test ---")
        print(f"Identification time: {elapsed:.4f}s")
        print(f"Model order: {A.shape[0]}")
        print(f"Fit variance: {Vn:.6f}")

        # Check that we got reasonable results
        assert A.shape[0] >= 1, "Model order should be at least 1"
        assert np.isfinite(Vn), "Variance should be finite"
        assert not np.any(np.isnan(A)), "A matrix should not contain NaN"
        assert not np.any(np.isnan(B)), "B matrix should not contain NaN"


def benchmark_varying_simulations():
    """
    Comprehensive benchmark across different n_simulations values.
    Run manually for detailed performance analysis.
    """
    print("\n" + "=" * 80)
    print("COMPREHENSIVE PERFORMANCE BENCHMARK")
    print("=" * 80)

    # Test different system sizes
    test_configs = [
        {"n": 2, "m": 1, "l_": 1, "L": 200, "label": "Tiny (n=2, m=1)"},
        {"n": 5, "m": 1, "l_": 1, "L": 300, "label": "Small (n=5, m=1)"},
        {"n": 8, "m": 2, "l_": 1, "L": 400, "label": "Medium (n=8, m=2)"},
        {"n": 10, "m": 2, "l_": 2, "L": 500, "label": "Large (n=10, m=2, l_=2)"},
        {"n": 15, "m": 3, "l_": 2, "L": 600, "label": "X-Large (n=15, m=3, l_=2)"},
    ]

    results = []

    for config in test_configs:
        np.random.seed(42)
        n, m, l_, L = config["n"], config["m"], config["l_"], config["L"]

        # Create system
        A_K = np.random.randn(n, n) * 0.1
        np.fill_diagonal(A_K, np.random.rand(n) * 0.5 + 0.2)
        C = np.random.randn(l_, n) * 0.5
        K = np.random.randn(n, l_) * 0.1
        D = np.zeros((l_, m))
        u = np.random.randn(m, L)
        y = np.random.randn(l_, L)

        n_simulations = n * m + n * l_ + n

        # Warm up
        _ = ParsimCoreAlgorithm.simulations_sequence_k(
            A_K, C, L, y, u, l_, m, n, K, D, D_required=False
        )

        # Benchmark parallel
        start = time.perf_counter()
        for _ in range(3):
            _ = ParsimCoreAlgorithm.simulations_sequence_k(
                A_K, C, L, y, u, l_, m, n, K, D, D_required=False
            )
        time_parallel = (time.perf_counter() - start) / 3

        # Benchmark sequential
        from src.sippy.identification.algorithms import parsim_core

        joblib_available = parsim_core.JOBLIB_AVAILABLE
        parsim_core.JOBLIB_AVAILABLE = False

        start = time.perf_counter()
        for _ in range(3):
            _ = ParsimCoreAlgorithm.simulations_sequence_k(
                A_K, C, L, y, u, l_, m, n, K, D, D_required=False
            )
        time_sequential = (time.perf_counter() - start) / 3

        parsim_core.JOBLIB_AVAILABLE = joblib_available

        speedup = time_sequential / time_parallel

        results.append(
            {
                "config": config["label"],
                "n_simulations": n_simulations,
                "time_sequential": time_sequential,
                "time_parallel": time_parallel,
                "speedup": speedup,
            }
        )

        print(f"\n{config['label']}:")
        print(f"  n_simulations:   {n_simulations}")
        print(f"  Sequential time: {time_sequential:.4f}s")
        print(f"  Parallel time:   {time_parallel:.4f}s")
        print(f"  Speedup:         {speedup:.2f}x")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Config':<30} {'n_sim':<10} {'Seq (s)':<12} {'Par (s)':<12} {'Speedup':<10}")
    print("-" * 80)
    for r in results:
        print(
            f"{r['config']:<30} {r['n_simulations']:<10} "
            f"{r['time_sequential']:<12.4f} {r['time_parallel']:<12.4f} "
            f"{r['speedup']:<10.2f}x"
        )

    return results


if __name__ == "__main__":
    # Run comprehensive benchmark
    benchmark_varying_simulations()
