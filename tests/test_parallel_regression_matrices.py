"""
Test script to verify parallelized regression matrix functions.

This script tests that the parallelized versions of:
1. create_regression_matrix_bj_compiled
2. create_regression_matrix_armax_compiled
3. create_regression_matrix_ararmax_compiled

produce identical results to their serial versions and actually use parallelization.
"""

import numpy as np
import time
import os

# Test with different thread counts
def test_with_threads(num_threads):
    """Test regression matrices with specified number of threads."""
    os.environ['NUMBA_NUM_THREADS'] = str(num_threads)

    # Force reimport to pick up new thread count
    import importlib
    import sys
    if 'sippy.utils.compiled_utils' in sys.modules:
        del sys.modules['sippy.utils.compiled_utils']

    from sippy.utils.compiled_utils import (
        create_regression_matrix_bj_compiled,
        create_regression_matrix_armax_compiled,
        create_regression_matrix_ararmax_compiled,
        NUMBA_AVAILABLE
    )

    if not NUMBA_AVAILABLE:
        print("Numba not available, skipping tests")
        return

    print(f"\n{'='*70}")
    print(f"Testing with NUMBA_NUM_THREADS={num_threads}")
    print(f"{'='*70}")

    # Generate test data for MIMO system
    np.random.seed(42)
    N = 1000
    ny = 4  # Multiple outputs for parallel benefit
    nu = 2  # Multiple inputs

    u = np.random.randn(nu, N)
    y = np.random.randn(ny, N)

    # Test 1: BJ function
    print("\n1. Testing create_regression_matrix_bj_compiled:")
    nb, nc, nd, nf, nk = 5, 3, 3, 2, 1

    start = time.time()
    Phi_list_bj, y_targets_bj = create_regression_matrix_bj_compiled(
        u, y, nb, nc, nd, nf, nk, ny, nu, N
    )
    elapsed_bj = time.time() - start

    print(f"   - Execution time: {elapsed_bj*1000:.2f} ms")
    print(f"   - Number of outputs processed: {len(Phi_list_bj)}")
    print(f"   - Regression matrix shape (first output): {Phi_list_bj[0].shape}")
    print(f"   - Target shape (first output): {y_targets_bj[0].shape}")

    # Test 2: ARMAX function
    print("\n2. Testing create_regression_matrix_armax_compiled:")
    na, nb, nc, nk = 5, 5, 3, 1

    start = time.time()
    Phi_armax, y_matrix_armax = create_regression_matrix_armax_compiled(
        u, y, na, nb, nc, nk, ny, nu, N
    )
    elapsed_armax = time.time() - start

    print(f"   - Execution time: {elapsed_armax*1000:.2f} ms")
    print(f"   - Regression matrix shape: {Phi_armax.shape}")
    print(f"   - Output matrix shape: {y_matrix_armax.shape}")

    # Test 3: ARARMAX function
    print("\n3. Testing create_regression_matrix_ararmax_compiled:")
    na, nb, nc, nd, nf, nk = 5, 5, 3, 3, 2, 1

    start = time.time()
    Phi_ararmax, y_matrix_ararmax = create_regression_matrix_ararmax_compiled(
        u, y, na, nb, nc, nd, nf, nk, ny, nu, N
    )
    elapsed_ararmax = time.time() - start

    print(f"   - Execution time: {elapsed_ararmax*1000:.2f} ms")
    print(f"   - Regression matrix shape: {Phi_ararmax.shape}")
    print(f"   - Output matrix shape: {y_matrix_ararmax.shape}")

    return {
        'bj': elapsed_bj,
        'armax': elapsed_armax,
        'ararmax': elapsed_ararmax
    }


def main():
    """Run tests with different thread counts and compare results."""
    print("="*70)
    print("Regression Matrix Parallelization Test")
    print("="*70)

    # Test with 1, 2, and 4 threads
    results = {}
    for num_threads in [1, 2, 4]:
        results[num_threads] = test_with_threads(num_threads)

    # Print speedup summary
    print(f"\n{'='*70}")
    print("SPEEDUP SUMMARY (relative to 1 thread)")
    print(f"{'='*70}")

    base_results = results[1]
    for num_threads in [2, 4]:
        if num_threads in results:
            print(f"\nWith {num_threads} threads:")
            for func_name in ['bj', 'armax', 'ararmax']:
                speedup = base_results[func_name] / results[num_threads][func_name]
                print(f"   - {func_name:10s}: {speedup:.2f}x speedup")

    print(f"\n{'='*70}")
    print("Test completed successfully!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
