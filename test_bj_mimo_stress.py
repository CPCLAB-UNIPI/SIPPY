#!/usr/bin/env python3
"""
Stress test for BJ MIMO algorithm to verify race condition fix.

This test runs the BJ MIMO identification multiple times to ensure
the race condition in create_regression_matrix_bj_compiled is fixed.
"""

import numpy as np
import pandas as pd
import sys
from sippy.identification.algorithms.bj import BJAlgorithm
from sippy.identification.base import SystemIdentificationConfig, StateSpaceModel
from sippy.identification.iddata import IDData


def create_mimo_test_data(ny=2, nu=2, n_samples=500):
    """Create MIMO test data for BJ algorithm."""
    np.random.seed(42)

    u = np.random.randn(nu, n_samples)
    y = np.zeros((ny, n_samples))

    # Simple input-output relationships with noise
    for k in range(2, n_samples):
        # First output
        y[0, k] = (
            0.3 * u[0, k - 1]
            + 0.2 * u[1, k - 1] if nu > 1 else 0
            + 0.1 * y[0, k - 1]
            + 0.05 * np.random.randn()
        )

        # Second output (if MIMO)
        if ny > 1:
            y[1, k] = (
                0.4 * u[0, k - 1]
                + 0.1 * u[1, k - 1] if nu > 1 else 0
                + 0.2 * y[0, k - 1]
                + 0.05 * y[1, k - 1]
                + 0.05 * np.random.randn()
            )

    return u, y


def test_bj_mimo_once(ny=2, nu=2, verbose=False):
    """Run one iteration of BJ MIMO test."""
    # Create test data
    u, y = create_mimo_test_data(ny, nu, n_samples=500)

    # Create IDData
    time_index = pd.date_range("2023-01-01", periods=500, freq="1s")

    # Build dataframe columns dynamically
    data_dict = {}
    for i in range(nu):
        data_dict[f"u{i+1}"] = u[i, :]
    for i in range(ny):
        data_dict[f"y{i+1}"] = y[i, :]

    data_df = pd.DataFrame(data_dict, index=time_index)

    inputs = [f"u{i+1}" for i in range(nu)]
    outputs = [f"y{i+1}" for i in range(ny)]

    data = IDData(data=data_df, inputs=inputs, outputs=outputs, tsample=1.0)

    # Configure BJ algorithm
    config = SystemIdentificationConfig(method="BJ")
    config.nb = 1
    config.nc = 1
    config.nd = 1
    config.nf = 1
    config.nk = 0

    # Run identification
    algorithm = BJAlgorithm()
    result = algorithm.identify(data, config)

    # Verify result
    assert result is not None, "BJ identification returned None"
    assert isinstance(result, StateSpaceModel), "Result is not a StateSpaceModel"

    if verbose:
        print(f"  ✓ MIMO ({ny}x{nu}) test passed")

    return True


def run_stress_test(n_iterations=100, verbose=True):
    """
    Run stress test for BJ MIMO algorithm.

    Parameters:
    -----------
    n_iterations : int
        Number of test iterations
    verbose : bool
        Print progress information
    """
    print(f"\n{'='*60}")
    print("BJ MIMO Stress Test")
    print(f"{'='*60}")
    print(f"Running {n_iterations} iterations to verify race condition fix...")
    print()

    # Test different MIMO configurations
    configurations = [
        (2, 2),  # 2x2 MIMO
        (3, 2),  # 3x2 MIMO
        (2, 3),  # 2x3 MIMO
        (3, 3),  # 3x3 MIMO
    ]

    total_tests = 0
    failed_tests = 0

    for ny, nu in configurations:
        print(f"\nTesting {ny}x{nu} MIMO system:")
        print("-" * 40)

        success_count = 0
        for i in range(n_iterations):
            try:
                test_bj_mimo_once(ny, nu, verbose=False)
                success_count += 1
                total_tests += 1

                # Progress indicator
                if (i + 1) % 10 == 0:
                    print(f"  [{i+1}/{n_iterations}] iterations completed", end="\r")

            except Exception as e:
                failed_tests += 1
                total_tests += 1
                print(f"\n  ✗ Failed at iteration {i+1}: {str(e)}")
                # Don't stop, continue testing

        print(f"  ✓ {success_count}/{n_iterations} tests passed for {ny}x{nu} MIMO")

    # Final summary
    print(f"\n{'='*60}")
    print("Test Summary:")
    print(f"{'='*60}")
    print(f"Total tests run: {total_tests}")
    print(f"Successful: {total_tests - failed_tests}")
    print(f"Failed: {failed_tests}")

    if failed_tests == 0:
        print("\n✅ ALL TESTS PASSED - Race condition appears to be fixed!")
        return 0
    else:
        print(f"\n❌ {failed_tests} TESTS FAILED - Issue may persist")
        return 1


if __name__ == "__main__":
    # Run with fewer iterations for quick test, or specify number
    n_iter = 10 if len(sys.argv) == 1 else int(sys.argv[1])
    exit_code = run_stress_test(n_iterations=n_iter)
    sys.exit(exit_code)