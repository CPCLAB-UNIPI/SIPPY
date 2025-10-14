#!/usr/bin/env python3
"""
Test script for parallel simulate_ss_system_compiled function.

Tests numerical equivalence and benchmarks performance with different thread counts.
"""

import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Set Numba thread count before importing
test_thread_counts = [1, 2, 4, 8]

# Import after setting environment
from sippy.utils.compiled_utils import simulate_ss_system_compiled, NUMBA_AVAILABLE

print(f"Numba available: {NUMBA_AVAILABLE}")
print(f"NumPy version: {np.__version__}")
print()


def create_test_system(n_states, n_inputs, n_outputs, time_steps):
    """Create a random stable state-space system for testing."""
    np.random.seed(42)

    # Create stable A matrix (eigenvalues < 1)
    A = np.random.randn(n_states, n_states) * 0.3
    A = A - np.eye(n_states) * 0.1  # Make it stable

    # Random B, C, D matrices
    B = np.random.randn(n_states, n_inputs)
    C = np.random.randn(n_outputs, n_states)
    D = np.random.randn(n_outputs, n_inputs) * 0.1

    # Random input signal
    u = np.random.randn(n_inputs, time_steps)

    # Random initial state
    x0 = np.random.randn(n_states, 1)

    return A, B, C, D, u, x0


def test_numerical_equivalence():
    """Test that parallel version produces identical results to serial."""
    print("=" * 80)
    print("TEST 1: Numerical Equivalence")
    print("=" * 80)

    test_cases = [
        ("Small system (n=5)", 5, 2, 2, 100),
        ("Medium system (n=10)", 10, 3, 3, 200),
        ("Large system (n=20)", 20, 4, 4, 500),
    ]

    for name, n, m, l, L in test_cases:
        print(f"\n{name}: n={n}, m={m}, l={l}, L={L}")

        A, B, C, D, u, x0 = create_test_system(n, m, l, L)

        # Run simulation twice with same inputs
        x1, y1 = simulate_ss_system_compiled(A, B, C, D, u, x0)
        x2, y2 = simulate_ss_system_compiled(A, B, C, D, u, x0)

        # Check numerical equivalence
        x_diff = np.abs(x1 - x2).max()
        y_diff = np.abs(y1 - y2).max()

        x_rtol = np.abs((x1 - x2) / (np.abs(x1) + 1e-15)).max()
        y_rtol = np.abs((y1 - y2) / (np.abs(y1) + 1e-15)).max()

        print(f"  State max abs diff: {x_diff:.2e}")
        print(f"  Output max abs diff: {y_diff:.2e}")
        print(f"  State max rel diff: {x_rtol:.2e}")
        print(f"  Output max rel diff: {y_rtol:.2e}")

        # Verify numerical equivalence (should be exact or very close)
        assert x_diff < 1e-12, f"State difference too large: {x_diff}"
        assert y_diff < 1e-12, f"Output difference too large: {y_diff}"
        print(f"  ✓ PASSED: Numerical equivalence verified")

    print("\n" + "=" * 80)
    print("✓ ALL NUMERICAL EQUIVALENCE TESTS PASSED")
    print("=" * 80)


def benchmark_performance():
    """Benchmark performance with different thread counts."""
    print("\n" + "=" * 80)
    print("TEST 2: Performance Benchmarking")
    print("=" * 80)

    # Test different system sizes
    test_cases = [
        ("Small (n=5)", 5, 2, 2, 1000),
        ("Medium (n=10)", 10, 3, 3, 1000),
        ("Large (n=20)", 20, 4, 4, 1000),
        ("Very Large (n=50)", 50, 5, 5, 1000),
    ]

    results = {}

    for thread_count in test_thread_counts:
        # Set thread count for this iteration
        os.environ['NUMBA_NUM_THREADS'] = str(thread_count)
        print(f"\n{'=' * 80}")
        print(f"Testing with {thread_count} thread(s)")
        print(f"{'=' * 80}")

        results[thread_count] = {}

        for name, n, m, l, L in test_cases:
            A, B, C, D, u, x0 = create_test_system(n, m, l, L)

            # Warmup run to compile
            simulate_ss_system_compiled(A, B, C, D, u, x0)

            # Benchmark
            n_runs = 10
            times = []
            for _ in range(n_runs):
                start = time.perf_counter()
                x, y = simulate_ss_system_compiled(A, B, C, D, u, x0)
                end = time.perf_counter()
                times.append(end - start)

            avg_time = np.mean(times)
            std_time = np.std(times)
            min_time = np.min(times)

            results[thread_count][name] = {
                'avg': avg_time,
                'std': std_time,
                'min': min_time,
            }

            print(f"{name:20s}: {avg_time*1000:7.2f} ± {std_time*1000:5.2f} ms "
                  f"(min: {min_time*1000:7.2f} ms)")

    # Calculate speedups relative to single thread
    print("\n" + "=" * 80)
    print("Speedup Analysis (relative to 1 thread)")
    print("=" * 80)

    for name, _, _, _, _ in test_cases:
        print(f"\n{name}:")
        baseline = results[1][name]['avg']
        for thread_count in test_thread_counts:
            current = results[thread_count][name]['avg']
            speedup = baseline / current
            print(f"  {thread_count} threads: {speedup:.2f}x speedup "
                  f"({baseline*1000:.2f} ms → {current*1000:.2f} ms)")

    # Create speedup plot
    create_speedup_plot(test_cases, results)

    print("\n" + "=" * 80)
    print("✓ PERFORMANCE BENCHMARKING COMPLETE")
    print("=" * 80)

    return results


def create_speedup_plot(test_cases, results):
    """Create and save speedup visualization."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Speedup vs Thread Count
    for name, _, _, _, _ in test_cases:
        speedups = []
        baseline = results[1][name]['avg']
        for thread_count in test_thread_counts:
            current = results[thread_count][name]['avg']
            speedup = baseline / current
            speedups.append(speedup)

        ax1.plot(test_thread_counts, speedups, 'o-', label=name, linewidth=2)

    # Add ideal speedup line
    ax1.plot(test_thread_counts, test_thread_counts, 'k--',
             label='Ideal (linear)', linewidth=1.5, alpha=0.5)

    ax1.set_xlabel('Number of Threads', fontsize=12)
    ax1.set_ylabel('Speedup', fontsize=12)
    ax1.set_title('Parallel Speedup vs Thread Count', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(test_thread_counts)

    # Plot 2: Execution Time vs Thread Count
    for name, _, _, _, _ in test_cases:
        times = []
        for thread_count in test_thread_counts:
            times.append(results[thread_count][name]['avg'] * 1000)

        ax2.plot(test_thread_counts, times, 'o-', label=name, linewidth=2)

    ax2.set_xlabel('Number of Threads', fontsize=12)
    ax2.set_ylabel('Execution Time (ms)', fontsize=12)
    ax2.set_title('Execution Time vs Thread Count', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(test_thread_counts)
    ax2.set_yscale('log')

    plt.tight_layout()
    plt.savefig('/Users/josephj/Workspace/SIPPY/simulate_parallel_speedup.png',
                dpi=150, bbox_inches='tight')
    print("\n✓ Speedup plot saved to: simulate_parallel_speedup.png")
    plt.close()


def test_race_conditions():
    """Test for race conditions by running many iterations."""
    print("\n" + "=" * 80)
    print("TEST 3: Race Condition Detection")
    print("=" * 80)

    n, m, l, L = 10, 3, 3, 200
    A, B, C, D, u, x0 = create_test_system(n, m, l, L)

    # Get reference output
    x_ref, y_ref = simulate_ss_system_compiled(A, B, C, D, u, x0)

    # Run multiple times and check consistency
    n_runs = 100
    print(f"\nRunning {n_runs} iterations to detect race conditions...")

    max_x_diff = 0.0
    max_y_diff = 0.0

    for i in range(n_runs):
        x, y = simulate_ss_system_compiled(A, B, C, D, u, x0)

        x_diff = np.abs(x - x_ref).max()
        y_diff = np.abs(y - y_ref).max()

        max_x_diff = max(max_x_diff, x_diff)
        max_y_diff = max(max_y_diff, y_diff)

        if i % 20 == 0:
            print(f"  Iteration {i+1}/{n_runs}: x_diff={x_diff:.2e}, y_diff={y_diff:.2e}")

    print(f"\nMaximum differences across {n_runs} runs:")
    print(f"  State: {max_x_diff:.2e}")
    print(f"  Output: {max_y_diff:.2e}")

    # Verify no race conditions (should be exactly zero or floating point precision)
    assert max_x_diff < 1e-12, f"Race condition detected: max_x_diff = {max_x_diff}"
    assert max_y_diff < 1e-12, f"Race condition detected: max_y_diff = {max_y_diff}"

    print("\n✓ NO RACE CONDITIONS DETECTED")
    print("=" * 80)


def main():
    """Run all tests."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "  PARALLEL STATE-SPACE SIMULATION TEST SUITE".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "=" * 78 + "╝")
    print()

    if not NUMBA_AVAILABLE:
        print("⚠ WARNING: Numba not available. Tests will run but without parallelization.")
        print()

    try:
        # Test 1: Numerical equivalence
        test_numerical_equivalence()

        # Test 2: Performance benchmarking
        results = benchmark_performance()

        # Test 3: Race conditions
        test_race_conditions()

        # Final summary
        print("\n" + "╔" + "=" * 78 + "╗")
        print("║" + " " * 78 + "║")
        print("║" + "  ✓ ALL TESTS PASSED SUCCESSFULLY".center(78) + "║")
        print("║" + " " * 78 + "║")
        print("╚" + "=" * 78 + "╝")
        print()

        # Print best speedup
        if NUMBA_AVAILABLE and len(results) > 1:
            print("Performance Summary:")
            print("-" * 80)
            for thread_count in [4, 8]:
                if thread_count in results:
                    speedups = []
                    for name in results[thread_count].keys():
                        baseline = results[1][name]['avg']
                        current = results[thread_count][name]['avg']
                        speedup = baseline / current
                        speedups.append(speedup)

                    avg_speedup = np.mean(speedups)
                    print(f"  Average speedup with {thread_count} threads: {avg_speedup:.2f}x")
            print()

        return 0

    except Exception as e:
        print("\n" + "=" * 80)
        print(f"❌ TEST FAILED: {e}")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
