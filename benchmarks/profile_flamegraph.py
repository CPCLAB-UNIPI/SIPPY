#!/usr/bin/env python
"""
Test script for py-spy flamegraph generation
"""

import time
import numpy as np
from sippy.identification import system_identification


def generate_test_data(n_samples: int = 2000):
    """Generate test data."""
    u = np.random.randn(1, n_samples)
    y = np.random.randn(1, n_samples)
    return y, u


def main():
    """Main function to profile."""
    print("Running algorithm for flamegraph profiling...")
    
    # Generate data
    y, u = generate_test_data(1000)
    
    # Run N4SID algorithm (the fastest one from benchmark)
    print("Running N4SID...")
    model = system_identification(
        y=y, u=u, 
        id_method='N4SID', 
        tsample=1.0, 
        SS_fixed_order=3
    )
    
    print("Profiling complete.")


if __name__ == '__main__':
    main()
