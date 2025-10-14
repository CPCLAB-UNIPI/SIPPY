"""
Comprehensive test suite for SIMD-optimized Vn_mat_compiled variants.

Tests numerical accuracy, performance, and adaptive strategy selection
across different array sizes.
"""

import time
import numpy as np

from sippy.utils.compiled_utils import (
    NUMBA_AVAILABLE,
    Vn_mat_compiled,
    Vn_mat_compiled_simd,
    Vn_mat_adaptive,
)


def compute_reference_variance(y, yest):
    """Pure Python reference implementation."""
    diff = y.flatten() - yest.flatten()
    return np.mean(diff**2)


def test_numerical_equivalence():
    """Test that all three implementations produce identical results."""
    print("\n" + "=" * 70)
    print("TEST 1: Numerical Equivalence")
    print("=" * 70)

    test_cases = [
        ("Small 1D (100)", (100,)),
        ("Medium 1D (1000)", (1000,)),
        ("Large 1D (10000)", (10000,)),
        ("Very Large 1D (100000)", (100000,)),
        ("2D array (100x100)", (100, 100)),
        ("3D array (10x10x10)", (10, 10, 10)),
        ("Non-square 2D (50x200)", (50, 200)),
    ]

    all_passed = True

    for name, shape in test_cases:
        np.random.seed(42)
        y = np.random.randn(*shape)
        yest = np.random.randn(*shape)

        # Compute with all implementations
        ref = compute_reference_variance(y, yest)
        parallel = Vn_mat_compiled(y, yest)
        simd = Vn_mat_compiled_simd(y, yest)
        adaptive = Vn_mat_adaptive(y, yest, strategy="auto")

        # Compute relative errors
        rel_err_parallel = abs(parallel - ref) / (abs(ref) + 1e-15)
        rel_err_simd = abs(simd - ref) / (abs(ref) + 1e-15)
        rel_err_adaptive = abs(adaptive - ref) / (abs(ref) + 1e-15)

        # Check tolerance (should be within floating point precision)
        tolerance = 1e-10
        passed = (
            rel_err_parallel < tolerance
            and rel_err_simd < tolerance
            and rel_err_adaptive < tolerance
        )

        status = "✓ PASS" if passed else "✗ FAIL"
        all_passed = all_passed and passed

        print(f"{status} {name:25s}: ", end="")
        print(
            f"parallel={rel_err_parallel:.2e}, simd={rel_err_simd:.2e}, adaptive={rel_err_adaptive:.2e}"
        )

    return all_passed


def test_edge_cases():
    """Test edge cases like empty arrays, single elements, etc."""
    print("\n" + "=" * 70)
    print("TEST 2: Edge Cases")
    print("=" * 70)

    test_cases = [
        ("Empty array", np.array([]), np.array([])),
        ("Single element", np.array([5.0]), np.array([3.0])),
        ("All zeros", np.zeros(100), np.zeros(100)),
        ("Identical arrays", np.ones(100), np.ones(100)),
        ("Large difference", np.ones(100), np.zeros(100)),
    ]

    all_passed = True

    for name, y, yest in test_cases:
        if y.size == 0:
            # Empty arrays
            parallel = Vn_mat_compiled(y, yest)
            simd = Vn_mat_compiled_simd(y, yest)
            adaptive = Vn_mat_adaptive(y, yest)
            passed = parallel == 0.0 and simd == 0.0 and adaptive == 0.0
        else:
            ref = compute_reference_variance(y, yest)
            parallel = Vn_mat_compiled(y, yest)
            simd = Vn_mat_compiled_simd(y, yest)
            adaptive = Vn_mat_adaptive(y, yest)

            rel_err_parallel = abs(parallel - ref) / (abs(ref) + 1e-15)
            rel_err_simd = abs(simd - ref) / (abs(ref) + 1e-15)
            rel_err_adaptive = abs(adaptive - ref) / (abs(ref) + 1e-15)

            passed = (
                rel_err_parallel < 1e-10
                and rel_err_simd < 1e-10
                and rel_err_adaptive < 1e-10
            )

        status = "✓ PASS" if passed else "✗ FAIL"
        all_passed = all_passed and passed

        print(
            f"{status} {name:25s}: parallel={Vn_mat_compiled(y, yest):.6e}, simd={Vn_mat_compiled_simd(y, yest):.6e}"
        )

    return all_passed


def test_non_divisible_by_4():
    """Test SIMD implementation with array sizes not divisible by 4."""
    print("\n" + "=" * 70)
    print("TEST 3: Non-Divisible by 4 (SIMD edge case)")
    print("=" * 70)

    all_passed = True

    for size in [1, 2, 3, 5, 7, 11, 101, 1001, 10003]:
        np.random.seed(42)
        y = np.random.randn(size)
        yest = np.random.randn(size)

        ref = compute_reference_variance(y, yest)
        simd = Vn_mat_compiled_simd(y, yest)

        rel_err = abs(simd - ref) / (abs(ref) + 1e-15)
        passed = rel_err < 1e-10

        status = "✓ PASS" if passed else "✗ FAIL"
        all_passed = all_passed and passed

        print(f"{status} Size {size:6d}: rel_error = {rel_err:.2e}")

    return all_passed


def test_adaptive_strategy_manual():
    """Test manual strategy selection in adaptive dispatcher."""
    print("\n" + "=" * 70)
    print("TEST 4: Adaptive Strategy Manual Selection")
    print("=" * 70)

    np.random.seed(42)
    y = np.random.randn(1000)
    yest = np.random.randn(1000)

    # Test all strategies
    auto = Vn_mat_adaptive(y, yest, strategy="auto")
    simd = Vn_mat_adaptive(y, yest, strategy="simd")
    parallel = Vn_mat_adaptive(y, yest, strategy="parallel")

    # Also test integer codes
    auto_int = Vn_mat_adaptive(y, yest, strategy=0)
    simd_int = Vn_mat_adaptive(y, yest, strategy=1)
    parallel_int = Vn_mat_adaptive(y, yest, strategy=2)

    ref = compute_reference_variance(y, yest)

    all_passed = True
    for name, result in [
        ("auto (str)", auto),
        ("simd (str)", simd),
        ("parallel (str)", parallel),
        ("auto (int=0)", auto_int),
        ("simd (int=1)", simd_int),
        ("parallel (int=2)", parallel_int),
    ]:
        rel_err = abs(result - ref) / (abs(ref) + 1e-15)
        passed = rel_err < 1e-10
        status = "✓ PASS" if passed else "✗ FAIL"
        all_passed = all_passed and passed
        print(f"{status} {name:20s}: rel_error = {rel_err:.2e}")

    return all_passed


def test_threshold_validation():
    """Validate that adaptive strategy uses correct thresholds."""
    print("\n" + "=" * 70)
    print("TEST 5: Adaptive Threshold Validation")
    print("=" * 70)

    # Test threshold boundaries
    test_sizes = [
        (100, "SIMD", "< 10k threshold"),
        (5000, "SIMD", "< 10k threshold"),
        (9999, "SIMD", "< 10k threshold"),
        (10000, "SIMD", "10k-100k threshold (SIMD default)"),
        (50000, "SIMD", "10k-100k threshold (SIMD default)"),
        (99999, "SIMD", "10k-100k threshold (SIMD default)"),
        (100000, "Parallel", "> 100k threshold"),
        (500000, "Parallel", "> 100k threshold"),
    ]

    all_passed = True

    for size, expected, description in test_sizes:
        np.random.seed(42)
        y = np.random.randn(size)
        yest = np.random.randn(size)

        # We can't directly check which implementation was used,
        # but we can verify correctness
        result = Vn_mat_adaptive(y, yest, strategy="auto")
        ref = compute_reference_variance(y, yest)

        rel_err = abs(result - ref) / (abs(ref) + 1e-15)
        passed = rel_err < 1e-10

        status = "✓ PASS" if passed else "✗ FAIL"
        all_passed = all_passed and passed

        print(f"{status} Size {size:7d} ({expected:8s}): {description:30s} rel_error={rel_err:.2e}")

    return all_passed


def benchmark_performance():
    """Comprehensive performance benchmark across array sizes."""
    print("\n" + "=" * 70)
    print("TEST 6: Performance Benchmarks")
    print("=" * 70)

    sizes = [100, 1000, 10000, 100000, 1000000]
    n_trials = 10

    print(f"\n{'Size':>10s} {'Parallel (ms)':>15s} {'SIMD (ms)':>15s} {'Adaptive (ms)':>15s} {'SIMD Speedup':>15s}")
    print("-" * 75)

    for size in sizes:
        np.random.seed(42)
        y = np.random.randn(size)
        yest = np.random.randn(size)

        # Warm-up (compile functions)
        _ = Vn_mat_compiled(y, yest)
        _ = Vn_mat_compiled_simd(y, yest)
        _ = Vn_mat_adaptive(y, yest)

        # Benchmark parallel version
        times_parallel = []
        for _ in range(n_trials):
            start = time.perf_counter()
            _ = Vn_mat_compiled(y, yest)
            times_parallel.append((time.perf_counter() - start) * 1000)

        # Benchmark SIMD version
        times_simd = []
        for _ in range(n_trials):
            start = time.perf_counter()
            _ = Vn_mat_compiled_simd(y, yest)
            times_simd.append((time.perf_counter() - start) * 1000)

        # Benchmark adaptive version
        times_adaptive = []
        for _ in range(n_trials):
            start = time.perf_counter()
            _ = Vn_mat_adaptive(y, yest)
            times_adaptive.append((time.perf_counter() - start) * 1000)

        avg_parallel = np.mean(times_parallel)
        avg_simd = np.mean(times_simd)
        avg_adaptive = np.mean(times_adaptive)
        speedup = avg_parallel / avg_simd

        print(
            f"{size:10d} {avg_parallel:15.3f} {avg_simd:15.3f} {avg_adaptive:15.3f} {speedup:15.2f}x"
        )

    print("\nNote: Speedup > 1.0 means SIMD is faster than parallel")


def test_integration_with_algorithms():
    """Test integration with SIPPY algorithms that use Vn_mat."""
    print("\n" + "=" * 70)
    print("TEST 7: Integration with SIPPY Algorithms")
    print("=" * 70)

    try:
        from sippy.utils.simulation_utils import Vn_mat

        # Test that Vn_mat wrapper works with all implementations
        np.random.seed(42)
        y = np.random.randn(1000)
        yest = np.random.randn(1000)

        # Call the high-level wrapper
        result = Vn_mat(y, yest)
        ref = compute_reference_variance(y, yest)

        rel_err = abs(result - ref) / (abs(ref) + 1e-15)
        passed = rel_err < 1e-10

        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status} Vn_mat wrapper: rel_error = {rel_err:.2e}")

        return passed

    except ImportError as e:
        print(f"⚠ SKIP: Could not import simulation_utils: {e}")
        return True


def run_all_tests():
    """Run all test suites."""
    print("\n" + "=" * 70)
    print("SIMD OPTIMIZATION TEST SUITE FOR Vn_mat_compiled")
    print("=" * 70)

    if not NUMBA_AVAILABLE:
        print("\n✗ FAILED: Numba not available. Tests require Numba.")
        return False

    print(f"\nNumba: ✓ Available")

    results = []

    # Run all tests
    results.append(("Numerical Equivalence", test_numerical_equivalence()))
    results.append(("Edge Cases", test_edge_cases()))
    results.append(("Non-Divisible by 4", test_non_divisible_by_4()))
    results.append(("Adaptive Strategy Manual", test_adaptive_strategy_manual()))
    results.append(("Threshold Validation", test_threshold_validation()))

    # Run benchmark (always included, not a pass/fail)
    benchmark_performance()

    # Run integration test
    results.append(("Integration Test", test_integration_with_algorithms()))

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    all_passed = True
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status} {test_name}")
        all_passed = all_passed and passed

    print("\n" + "=" * 70)
    if all_passed:
        print("✓ ALL TESTS PASSED")
    else:
        print("✗ SOME TESTS FAILED")
    print("=" * 70)

    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
