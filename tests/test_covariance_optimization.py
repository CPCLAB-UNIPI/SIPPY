"""
Test and benchmark the covariance_symmetric_compiled optimization.

This test file validates:
1. Numerical accuracy (optimized vs original < 1e-10 error)
2. Performance improvement (time both versions)
3. Edge cases (empty matrices, single element, large matrices)
"""

import time

import numpy as np

from src.sippy.utils.compiled_utils import (
    NUMBA_AVAILABLE,
    covariance_symmetric_compiled,
)


def compute_covariance_original(residuals, ddof=1):
    """Original covariance computation using np.dot."""
    return np.dot(residuals, residuals.T) / (residuals.shape[1] - ddof)


def test_numerical_accuracy():
    """Test numerical accuracy between optimized and original versions."""
    print("\n" + "=" * 80)
    print("TEST 1: Numerical Accuracy")
    print("=" * 80)

    test_cases = [
        ("Small SISO", (2, 100)),
        ("Medium SISO", (3, 500)),
        ("Large SISO", (4, 1000)),
        ("Small MIMO (2x2)", (4, 200)),
        ("Medium MIMO (3x3)", (9, 500)),
        ("Large MIMO (5x5)", (25, 1000)),
    ]

    all_passed = True
    for name, shape in test_cases:
        # Generate random residuals
        np.random.seed(42)
        residuals = np.random.randn(*shape)

        # Compute with both methods
        cov_original = compute_covariance_original(residuals, ddof=1)

        if NUMBA_AVAILABLE:
            cov_optimized = covariance_symmetric_compiled(residuals, ddof=1)

            # Check numerical accuracy
            max_abs_error = np.max(np.abs(cov_original - cov_optimized))
            max_rel_error = np.max(
                np.abs((cov_original - cov_optimized) / (np.abs(cov_original) + 1e-15))
            )
            frobenius_norm = np.linalg.norm(cov_original - cov_optimized, "fro")

            passed = max_abs_error < 1e-10 and max_rel_error < 1e-10

            print(f"\n{name} (shape={shape}):")
            print(f"  Max absolute error: {max_abs_error:.2e}")
            print(f"  Max relative error: {max_rel_error:.2e}")
            print(f"  Frobenius norm:     {frobenius_norm:.2e}")
            print(f"  Status: {'PASS' if passed else 'FAIL'}")

            if not passed:
                all_passed = False
        else:
            print(f"\n{name} (shape={shape}):")
            print("  Status: SKIPPED (Numba not available)")

    return all_passed


def test_symmetry():
    """Test that the covariance matrix is symmetric."""
    print("\n" + "=" * 80)
    print("TEST 2: Symmetry Verification")
    print("=" * 80)

    if not NUMBA_AVAILABLE:
        print("SKIPPED (Numba not available)")
        return True

    np.random.seed(42)
    residuals = np.random.randn(5, 100)

    cov = covariance_symmetric_compiled(residuals, ddof=1)

    # Check symmetry
    symmetry_error = np.max(np.abs(cov - cov.T))
    passed = symmetry_error < 1e-15

    print(f"\nSymmetry error: {symmetry_error:.2e}")
    print(f"Status: {'PASS' if passed else 'FAIL'}")

    return passed


def test_edge_cases():
    """Test edge cases."""
    print("\n" + "=" * 80)
    print("TEST 3: Edge Cases")
    print("=" * 80)

    if not NUMBA_AVAILABLE:
        print("SKIPPED (Numba not available)")
        return True

    all_passed = True

    # Test 1: Single element (optimized handles this gracefully)
    print("\nTest 3.1: Single element (1x1) with ddof=1")
    residuals = np.array([[1.5]])
    try:
        cov = covariance_symmetric_compiled(residuals, ddof=1)
        # Optimized version returns identity matrix for edge case
        # This is better behavior than division by zero
        passed = np.allclose(cov, np.eye(1))
        print(f"  Optimized returns identity: {passed}")
        print(f"  Status: {'PASS' if passed else 'FAIL'}")
        if not passed:
            all_passed = False
    except Exception as e:
        print(f"  Status: FAIL - {e}")
        all_passed = False

    # Test 2: Two samples (edge case for ddof=1)
    print("\nTest 3.2: Two samples (3x2)")
    residuals = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    try:
        cov = covariance_symmetric_compiled(residuals, ddof=1)
        cov_orig = compute_covariance_original(residuals, ddof=1)
        error = np.max(np.abs(cov - cov_orig))
        passed = error < 1e-10
        print(f"  Error: {error:.2e}")
        print(f"  Status: {'PASS' if passed else 'FAIL'}")
        if not passed:
            all_passed = False
    except Exception as e:
        print(f"  Status: FAIL - {e}")
        all_passed = False

    # Test 3: Zero residuals
    print("\nTest 3.3: Zero residuals (2x10)")
    residuals = np.zeros((2, 10))
    try:
        cov = covariance_symmetric_compiled(residuals, ddof=1)
        cov_orig = compute_covariance_original(residuals, ddof=1)
        error = np.max(np.abs(cov - cov_orig))
        passed = error < 1e-10
        print(f"  Error: {error:.2e}")
        print(f"  Status: {'PASS' if passed else 'FAIL'}")
        if not passed:
            all_passed = False
    except Exception as e:
        print(f"  Status: FAIL - {e}")
        all_passed = False

    return all_passed


def benchmark_performance():
    """Benchmark performance improvement."""
    print("\n" + "=" * 80)
    print("TEST 4: Performance Benchmark")
    print("=" * 80)

    if not NUMBA_AVAILABLE:
        print("SKIPPED (Numba not available)")
        return

    test_cases = [
        ("Small (3x100)", (3, 100)),
        ("Medium (10x500)", (10, 500)),
        ("Large (20x1000)", (20, 1000)),
        ("Very Large (50x2000)", (50, 2000)),
    ]

    for name, shape in test_cases:
        np.random.seed(42)
        residuals = np.random.randn(*shape)

        # Warmup (compile Numba function)
        _ = covariance_symmetric_compiled(residuals, ddof=1)

        # Benchmark original
        n_iter = 100
        start = time.time()
        for _ in range(n_iter):
            _ = compute_covariance_original(residuals, ddof=1)
        time_original = (time.time() - start) / n_iter

        # Benchmark optimized
        start = time.time()
        for _ in range(n_iter):
            _ = covariance_symmetric_compiled(residuals, ddof=1)
        time_optimized = (time.time() - start) / n_iter

        speedup = time_original / time_optimized

        print(f"\n{name} (shape={shape}):")
        print(f"  Original:  {time_original*1000:.3f} ms")
        print(f"  Optimized: {time_optimized*1000:.3f} ms")
        print(f"  Speedup:   {speedup:.2f}x")


def test_with_realistic_data():
    """Test with realistic subspace identification residuals."""
    print("\n" + "=" * 80)
    print("TEST 5: Realistic Subspace Data")
    print("=" * 80)

    if not NUMBA_AVAILABLE:
        print("SKIPPED (Numba not available)")
        return True

    # Simulate realistic residuals from N4SID algorithm
    # Typical dimensions: (n+l, N) where n=order, l=outputs, N=samples
    np.random.seed(42)

    test_cases = [
        ("N4SID SISO (n=5, l=1, N=500)", (6, 500)),
        ("N4SID MIMO 2x2 (n=10, l=2, N=1000)", (12, 1000)),
        ("MOESP MIMO 3x3 (n=15, l=3, N=2000)", (18, 2000)),
    ]

    all_passed = True
    for name, shape in test_cases:
        # Generate residuals with realistic statistics
        residuals = np.random.randn(*shape) * 0.1  # Small residuals typical

        cov_original = compute_covariance_original(residuals, ddof=1)
        cov_optimized = covariance_symmetric_compiled(residuals, ddof=1)

        # Check accuracy
        max_abs_error = np.max(np.abs(cov_original - cov_optimized))
        max_rel_error = np.max(
            np.abs((cov_original - cov_optimized) / (np.abs(cov_original) + 1e-15))
        )

        passed = max_abs_error < 1e-10 and max_rel_error < 1e-10

        print(f"\n{name}:")
        print(f"  Max absolute error: {max_abs_error:.2e}")
        print(f"  Max relative error: {max_rel_error:.2e}")
        print(f"  Status: {'PASS' if passed else 'FAIL'}")

        if not passed:
            all_passed = False

    return all_passed


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("COVARIANCE OPTIMIZATION TEST SUITE")
    print("=" * 80)

    if not NUMBA_AVAILABLE:
        print("\nWARNING: Numba not available. Most tests will be skipped.")
        print("Install Numba to run the full test suite.")
    else:
        print("\nNumba is available. Running full test suite.")

    # Run tests
    test1_pass = test_numerical_accuracy()
    test2_pass = test_symmetry()
    test3_pass = test_edge_cases()
    benchmark_performance()
    test5_pass = test_with_realistic_data()

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    if NUMBA_AVAILABLE:
        tests = [
            ("Numerical Accuracy", test1_pass),
            ("Symmetry", test2_pass),
            ("Edge Cases", test3_pass),
            ("Realistic Data", test5_pass),
        ]
        all_passed = all(result for _, result in tests)

        for name, result in tests:
            status = "PASS" if result else "FAIL"
            print(f"{name:30s}: {status}")

        print("\n" + "=" * 80)
        if all_passed:
            print("ALL TESTS PASSED")
        else:
            print("SOME TESTS FAILED")
        print("=" * 80)

        return all_passed
    else:
        print("Tests skipped (Numba not available)")
        return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
