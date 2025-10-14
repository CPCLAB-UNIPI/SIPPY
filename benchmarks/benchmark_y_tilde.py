"""
Benchmark script to measure performance improvement of the loop-based
parsim_y_tilde_estimation_compiled function.
"""

import numpy as np
import time

# Import the compiled function
try:
    from src.sippy.utils.compiled_utils import (
        parsim_y_tilde_estimation_compiled,
        NUMBA_AVAILABLE,
    )
    print(f"Numba available: {NUMBA_AVAILABLE}")
except ImportError:
    print("ERROR: Cannot import compiled utilities")
    exit(1)


def create_test_matrices(l_=2, m=3, f=20, n_cols=100):
    """Create test matrices for benchmarking."""
    # H_K grows with iterations, max size is (l_*f, m)
    H_K = np.random.randn(l_ * f, m)

    # G_K grows with iterations, max size is (l_*f, l_)
    G_K = np.random.randn(l_ * f, l_)

    # Uf: input future sequence (m*f, n_cols)
    Uf = np.random.randn(m * f, n_cols)

    # Yf: output future sequence (l_*f, n_cols)
    Yf = np.random.randn(l_ * f, n_cols)

    return H_K, G_K, Uf, Yf


def benchmark_function(H_K, G_K, Uf, Yf, m, l_, f, n_iterations=100):
    """Benchmark the y_tilde estimation function."""
    times = []

    # Warm-up (important for JIT compilation)
    for i in range(1, min(f, 5)):
        _ = parsim_y_tilde_estimation_compiled(H_K, Uf, G_K, Yf, i, m, l_, f)

    # Actual benchmark
    for _ in range(n_iterations):
        start = time.perf_counter()
        # Test with i=f-1 (most expensive case)
        i = f - 1
        result = parsim_y_tilde_estimation_compiled(H_K, Uf, G_K, Yf, i, m, l_, f)
        end = time.perf_counter()
        times.append(end - start)

    return np.array(times), result


def main():
    print("=" * 70)
    print("PARSIM Y_TILDE ESTIMATION BENCHMARK")
    print("=" * 70)

    # Test configurations
    configs = [
        {"name": "Small (l=2, m=2, f=10)", "l_": 2, "m": 2, "f": 10, "n_cols": 50},
        {"name": "Medium (l=2, m=3, f=20)", "l_": 2, "m": 3, "f": 20, "n_cols": 100},
        {"name": "Large (l=3, m=3, f=50)", "l_": 3, "m": 3, "f": 50, "n_cols": 200},
    ]

    for config in configs:
        print(f"\n{config['name']}")
        print("-" * 70)

        # Create test data
        H_K, G_K, Uf, Yf = create_test_matrices(
            config["l_"], config["m"], config["f"], config["n_cols"]
        )

        # Benchmark
        times, result = benchmark_function(
            H_K, G_K, Uf, Yf, config["m"], config["l_"], config["f"], n_iterations=100
        )

        # Statistics
        mean_time = np.mean(times) * 1000  # Convert to milliseconds
        std_time = np.std(times) * 1000
        min_time = np.min(times) * 1000
        max_time = np.max(times) * 1000

        print(f"  Result shape: {result.shape}")
        print(f"  Mean time:    {mean_time:.4f} ms")
        print(f"  Std dev:      {std_time:.4f} ms")
        print(f"  Min time:     {min_time:.4f} ms")
        print(f"  Max time:     {max_time:.4f} ms")
        print(f"  Throughput:   {1000.0/mean_time:.1f} calls/sec")

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)
    print("\nNote: The optimized loop-based implementation should provide")
    print("4-5x speedup compared to the original matrix slicing + np.dot")
    print("approach, especially for larger problem sizes.")


if __name__ == "__main__":
    main()
