#!/usr/bin/env python
"""
Debug script to check number of iterations in ARMA ILLS.

Author: Claude Code
Date: 2025-10-13
"""

import numpy as np
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from sippy.identification.algorithms.arma import ARMAAlgorithm

def generate_test_data(N=1000, na=2, nc=1, seed=42):
    """Generate synthetic ARMA time series data."""
    np.random.seed(seed)

    # Generate white noise
    e = np.random.normal(0, 0.1, N)

    # Generate ARMA process
    y = np.zeros(N)
    for k in range(max(na, nc), N):
        # AR component: y[k] = -a1*y[k-1] - a2*y[k-2] + ...
        y[k] = 0.6 * y[k-1] + 0.2 * y[k-2]

        # MA component: ... + e[k] + c1*e[k-1] + ...
        y[k] += e[k] + 0.3 * e[k-1]

    return y.reshape(1, -1)

def main():
    configs = [
        {"N": 500, "na": 2, "nc": 1, "name": "Small dataset"},
        {"N": 1000, "na": 2, "nc": 1, "name": "Medium dataset"},
        {"N": 2000, "na": 3, "nc": 2, "name": "Large dataset"},
    ]

    algorithm = ARMAAlgorithm()

    for config in configs:
        N = config["N"]
        na = config["na"]
        nc = config["nc"]
        name = config["name"]

        print(f"\n{name}: N={N}, na={na}, nc={nc}")

        # Generate test data
        y = generate_test_data(N, na, nc)

        # Add iteration counter by patching
        import time
        start = time.perf_counter()
        result = algorithm._identify_ills(y, np.zeros((1, y.shape[1])), na, nc, 1.0,
                                          max_iterations=100)
        end = time.perf_counter()

        print(f"  Time: {(end - start)*1000:.2f} ms")
        print(f"  AR coeffs: {result.AR_coeffs}")
        print(f"  MA coeffs: {result.MA_coeffs}")

if __name__ == "__main__":
    main()
