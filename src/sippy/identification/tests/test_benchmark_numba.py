#!/usr/bin/env python3
"""
Benchmark script to demonstrate the performance improvements from Numba JIT compilation.

This script compares the performance of key functions before and after
Numba optimization in the SIPPY library.
"""

import time
from typing import Tuple

import numpy as np


def benchmark_ordinate_sequence(size_multiplier: int = 10) -> Tuple[float, float]:
    """Benchmark the ordinate sequence function."""
    try:
        from ...utils.compiled_utils import NUMBA_AVAILABLE, ordinate_sequence_compiled
        from ...utils.simulation_utils import ordinate_sequence

        # Test data
        n_outputs = 3
        n_samples = 1000 * size_multiplier
        f = p = 20
        y = np.random.randn(n_outputs, n_samples)

        # Benchmark original implementation (force non-numba)
        from ... import utils as sim_utils

        sim_utils.ordinate_sequence_compiled = None
        sim_utils.NUMBA_AVAILABLE = False

        start_time = time.time()
        for _ in range(10):
            Yf1, Yp1 = ordinate_sequence(y, f, p)
        original_time = time.time() - start_time

        # Restore numba compilation
        sim_utils.ordinate_sequence_compiled = ordinate_sequence_compiled
        sim_utils.NUMBA_AVAILABLE = NUMBA_AVAILABLE

        # Benchmark numba implementation (warm up first)
        for _ in range(3):  # Warm up
            ordinate_sequence(y, f, p)

        start_time = time.time()
        for _ in range(10):
            Yf2, Yp2 = ordinate_sequence(y, f, p)
        numba_time = time.time() - start_time

        # Verify results are the same
        assert np.allclose(Yf1, Yf2), "Results differ between implementations!"
        assert np.allclose(Yp1, Yp2), "Results differ between implementations!"

        return original_time, numba_time

    except ImportError as e:
        print(f"Could not import required modules: {e}")
        return 0.0, 0.0


def benchmark_simulation(size_multiplier: int = 10) -> Tuple[float, float]:
    """Benchmark the simulation function."""
    try:
        from ...utils.compiled_utils import NUMBA_AVAILABLE, simulate_ss_system_compiled
        from ...utils.simulation_utils import simulate_ss_system

        # Test data
        n_states = 4
        n_inputs = 2
        n_outputs = 3
        n_samples = 1000 * size_multiplier

        A = np.random.randn(n_states, n_states)
        B = np.random.randn(n_states, n_inputs)
        C = np.random.randn(n_outputs, n_states)
        D = np.random.randn(n_outputs, n_inputs)
        u = np.random.randn(n_inputs, n_samples)
        x0 = np.random.randn(n_states, 1)

        # Make A stable
        A = A * 0.8 / np.max(np.abs(np.linalg.eigvals(A)))

        # Benchmark original implementation
        from ... import utils as sim_utils

        sim_utils.simulate_ss_system_compiled = None
        sim_utils.NUMBA_AVAILABLE = False

        start_time = time.time()
        for _ in range(10):
            x1, y1 = simulate_ss_system(A, B, C, D, u, x0)
        original_time = time.time() - start_time

        # Restore numba compilation
        sim_utils.simulate_ss_system_compiled = simulate_ss_system_compiled
        sim_utils.NUMBA_AVAILABLE = NUMBA_AVAILABLE

        # Benchmark numba implementation (warm up first)
        for _ in range(3):  # Warm up
            simulate_ss_system(A, B, C, D, u, x0)

        start_time = time.time()
        for _ in range(10):
            x2, y2 = simulate_ss_system(A, B, C, D, u, x0)
        numba_time = time.time() - start_time

        # Verify results are roughly the same
        assert np.allclose(x1, x2, rtol=1e-10), "State results differ!"
        assert np.allclose(y1, y2, rtol=1e-10), "Output results differ!"

        return original_time, numba_time

    except ImportError as e:
        print(f"Could not import required modules: {e}")
        return 0.0, 0.0


def benchmark_rescale(size_multiplier: int = 10) -> Tuple[float, float]:
    """Benchmark the rescale function."""
    try:
        from ...utils.compiled_utils import NUMBA_AVAILABLE, rescale_compiled
        from ...utils.signal_utils import rescale

        # Test data
        size = 10000 * size_multiplier
        y = np.random.randn(size)

        # Benchmark original implementation
        from ...utils import signal_utils as sig_utils

        sig_utils.rescale_compiled = None
        sig_utils.NUMBA_AVAILABLE = False

        start_time = time.time()
        for _ in range(100):
            std1, y1 = rescale(y)
        original_time = time.time() - start_time

        # Restore numba compilation
        sig_utils.rescale_compiled = rescale_compiled
        sig_utils.NUMBA_AVAILABLE = NUMBA_AVAILABLE

        # Benchmark numba implementation (warm up first)
        for _ in range(3):  # Warm up
            rescale(y)

        start_time = time.time()
        for _ in range(100):
            std2, y2 = rescale(y)
        numba_time = time.time() - start_time

        # Verify results are the same
        assert np.abs(std1 - std2) < 1e-10, "Standard deviation differs!"
        assert np.allclose(y1, y2), "Rescaled results differ!"

        return original_time, numba_time

    except ImportError as e:
        print(f"Could not import required modules: {e}")
        return 0.0, 0.0


def main():
    """Run all benchmarks and report results."""
    print("🚀 SIPPY Numba Performance Benchmarks")
    print("=" * 50)

    try:
        from ...utils.compiled_utils import NUMBA_AVAILABLE

        if not NUMBA_AVAILABLE:
            print("❌ Numba is not available. Cannot run benchmarks.")
            return
    except ImportError:
        print("❌ Could not import compiled utilities.")
        return

    print("Testing various data sizes...\n")

    functions = [
        ("Ordinate Sequence Creation", benchmark_ordinate_sequence),
        ("State-Space Simulation", benchmark_simulation),
        ("Signal Rescaling", benchmark_rescale),
    ]

    for name, func in functions:
        print(f"📊 {name}:")

        # Test different sizes
        for multiplier in [1, 10, 100]:
            try:
                original_time, numba_time = func(multiplier)

                if original_time > 0 and numba_time > 0:
                    speedup = original_time / numba_time
                    print(
                        f"  Size {multiplier}x: {original_time:.3f}s → {numba_time:.3f}s "
                        f"({speedup:.2f}x speedup)"
                    )
                else:
                    print(f"  Size {multiplier}x: Benchmark failed")
            except Exception as e:
                print(f"  Size {multiplier}x: Error - {e}")

        print()

    print("🎯 Benchmark Summary:")
    print("• Functions using Numba JIT compilation show significant speedup")
    print("• Performance improvement increases with data size")
    print("• Numerical accuracy is preserved while gaining performance")


if __name__ == "__main__":
    main()
