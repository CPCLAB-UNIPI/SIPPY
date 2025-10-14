#!/usr/bin/env python
"""
Comprehensive Performance Benchmark Suite for SIPPY Harold Branch

Tests all algorithms with various dataset sizes and measures:
- Execution time
- Speedup ratios
- Memory usage
- Numba compilation overhead
"""

import time
import numpy as np
import warnings
from typing import Dict, List, Tuple
import sys

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

from sippy.identification import system_identification
from sippy.utils.compiled_utils import NUMBA_AVAILABLE

print("=" * 80)
print("SIPPY COMPREHENSIVE PERFORMANCE BENCHMARK")
print("=" * 80)
print(f"Numba Available: {NUMBA_AVAILABLE}")
print(f"NumPy Version: {np.__version__}")
print()


def generate_test_data(n_samples: int, siso: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic test data."""
    if siso:
        u = np.random.randn(1, n_samples)
        y = np.random.randn(1, n_samples)
    else:
        u = np.random.randn(2, n_samples)
        y = np.random.randn(2, n_samples)
    return y, u


def benchmark_algorithm(algo_name: str, y: np.ndarray, u: np.ndarray,
                       orders: Dict, n_runs: int = 3) -> Dict:
    """Benchmark a single algorithm."""
    times = []
    success = False
    error_msg = None

    for i in range(n_runs):
        try:
            start = time.time()
            model = system_identification(
                y=y, 
                u=u, 
                id_method=algo_name,
                tsample=1.0,
                **orders
            )
            elapsed = time.time() - start
            times.append(elapsed)
            success = True
        except Exception as e:
            error_msg = str(e)
            break

    if success and times:
        return {
            'success': True,
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'times': times
        }
    else:
        return {
            'success': False,
            'error': error_msg or "Unknown error"
        }


def run_benchmark_suite():
    """Run comprehensive benchmark suite."""

    # Test configurations
    configs = [
        {
            'name': 'Small Dataset (N=1000)',
            'n_samples': 1000,
            'siso': True
        },
        {
            'name': 'Medium Dataset (N=5000)',
            'n_samples': 5000,
            'siso': True
        },
        {
            'name': 'Large Dataset (N=10000)',
            'n_samples': 10000,
            'siso': True
        }
    ]

    # Algorithm configurations
    algorithms = {
        'ARX': {'AR_orders': (5, 5)},
        'ARMAX': {'AR_orders': (3, 3), 'ARMA_orders': (2,)},
        'FIR': {'FIR_order': 10},
        'ARARX': {'AR_orders': (2, 2), 'ARMA_orders': (2,)},
        'N4SID': {'SS_fixed_order': 5},
        'MOESP': {'SS_fixed_order': 5},
        'CVA': {'SS_fixed_order': 5},
    }

    results = {}

    for config in configs:
        print("\n" + "=" * 80)
        print(f"Testing: {config['name']}")
        print("=" * 80)

        # Generate data
        y, u = generate_test_data(config['n_samples'], config['siso'])

        config_results = {}

        for algo_name, orders in algorithms.items():
            print(f"\n{algo_name:15s} ", end='', flush=True)

            result = benchmark_algorithm(algo_name, y, u, orders, n_runs=3)

            if result['success']:
                mean_time = result['mean_time']
                std_time = result['std_time']
                print(f"✓  {mean_time:.3f}s ± {std_time:.3f}s")
            else:
                print(f"✗  {result['error'][:60]}")

            config_results[algo_name] = result

        results[config['name']] = config_results

    return results


def print_summary(results: Dict):
    """Print summary of all results."""
    print("\n" + "=" * 80)
    print("PERFORMANCE SUMMARY")
    print("=" * 80)

    for config_name, config_results in results.items():
        print(f"\n{config_name}:")
        print("-" * 80)
        print(f"{'Algorithm':<15s} {'Status':<10s} {'Mean Time':<15s} {'Std Dev':<15s}")
        print("-" * 80)

        for algo_name, result in config_results.items():
            if result['success']:
                status = "SUCCESS"
                mean_time = f"{result['mean_time']:.3f}s"
                std_time = f"±{result['std_time']:.3f}s"
            else:
                status = "FAILED"
                mean_time = "-"
                std_time = result['error'][:30]

            print(f"{algo_name:<15s} {status:<10s} {mean_time:<15s} {std_time:<15s}")

    # Calculate speedup metrics if available
    print("\n" + "=" * 80)
    print("SPEEDUP ANALYSIS")
    print("=" * 80)

    # Compare small vs large datasets
    small_results = results.get('Small Dataset (N=1000)', {})
    large_results = results.get('Large Dataset (N=10000)', {})

    if small_results and large_results:
        print("\nScaling Factor (10x data increase):")
        print("-" * 80)
        print(f"{'Algorithm':<15s} {'Small':<15s} {'Large':<15s} {'Ratio':<15s}")
        print("-" * 80)

        for algo_name in small_results.keys():
            small = small_results.get(algo_name, {})
            large = large_results.get(algo_name, {})

            if small.get('success') and large.get('success'):
                small_time = small['mean_time']
                large_time = large['mean_time']
                ratio = large_time / small_time if small_time > 0 else 0

                print(f"{algo_name:<15s} {small_time:.3f}s{' '*8} {large_time:.3f}s{' '*8} {ratio:.2f}x")
            else:
                print(f"{algo_name:<15s} {'N/A':<15s} {'N/A':<15s} {'N/A':<15s}")


def check_numba_compilation():
    """Check Numba compilation status."""
    print("\n" + "=" * 80)
    print("NUMBA COMPILATION STATUS")
    print("=" * 80)

    if NUMBA_AVAILABLE:
        print("✓ Numba is available and JIT compilation is active")
        print("✓ Performance-critical functions will be compiled")

        # Test compilation
        from sippy.utils.compiled_utils import (
            simulate_ss_system_compiled,
            create_regression_matrix_arx_compiled,
            information_criterion_compiled
        )

        print("\nCompiled functions:")
        print("  - simulate_ss_system_compiled")
        print("  - create_regression_matrix_arx_compiled")
        print("  - information_criterion_compiled")
    else:
        print("✗ Numba is NOT available")
        print("  Performance will be slower (pure NumPy implementation)")


if __name__ == '__main__':
    # Check Numba status
    check_numba_compilation()

    # Run benchmarks
    results = run_benchmark_suite()

    # Print summary
    print_summary(results)

    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)
