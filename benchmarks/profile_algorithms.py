#!/usr/bin/env python
"""
Detailed line profiling for SIPPY algorithm performance analysis
"""

import time
import numpy as np
from sippy.identification import system_identification
from sippy.utils.compiled_utils import NUMBA_AVAILABLE

# Decorator for line profiling
try:
    from line_profiler import LineProfiler
    lp = LineProfiler()
except ImportError:
    lp = None
    print("line_profiler not available")


def generate_test_data(n_samples: int = 1000):
    """Generate test data."""
    u = np.random.randn(1, n_samples)
    y = np.random.randn(1, n_samples)
    return y, u


def profile_n4sid():
    """Profile N4SID algorithm."""
    print("\n=== Profiling N4SID ===")
    
    y, u = generate_test_data(2000)
    
    if lp:
        try:
            from sippy.identification.algorithms.n4sid import N4SIDAlgorithm
            from sippy.utils.compiled_utils import ordinate_sequence_compiled
            
            # Profile key functions
            lp.add_function(N4SIDAlgorithm.identify)
            lp.add_function(ordinate_sequence_compiled)
            lp_wrapper = lp(system_identification)
            
            lp_wrapper(y=y, u=u, id_method='N4SID', tsample=1.0, SS_fixed_order=5)
            lp.print_stats()
        except Exception as e:
            print(f"N4SID profiling failed: {e}")
            
            # Fallback to simple timing
            start = time.time()
            system_identification(y=y, u=u, id_method='N4SID', tsample=1.0, SS_fixed_order=5)
            elapsed = time.time() - start
            print(f"N4SID took {elapsed:.3f}s")
    else:
        # Simple timing fallback
        start = time.time()
        system_identification(y=y, u=u, id_method='N4SID', tsample=1.0, SS_fixed_order=5)
        elapsed = time.time() - start
        print(f"N4SID took {elapsed:.3f}s")


def profile_arma():
    """Profile ARMA algorithm."""
    print("\n=== Profiling ARMA ===")
    
    y, u = generate_test_data(1000)
    
    if lp:
        try:
            from sippy.identification.algorithms.arma import ARMAAlgorithm
            from sippy.utils.compiled_utils import create_regression_matrix_arx_compiled
            
            lp.add_function(ARMAAlgorithm.identify)
            lp.add_function(create_regression_matrix_arx_compiled)
            lp_wrapper = lp(system_identification)
            
            lp_wrapper(y=y, id_method='ARMA', tsample=1.0, ARMA_orders=(2, 2))
            lp.print_stats()
        except Exception as e:
            print(f"ARMA profiling failed: {e}")
    else:
        # Simple timing fallback
        try:
            start = time.time()
            system_identification(y=y, id_method='ARMA', tsample=1.0, ARMA_orders=(2, 2))
            elapsed = time.time() - start
            print(f"ARMA took {elapsed:.3f}s")
        except Exception as e:
            print(f"ARMA failed: {e}")


def profile_armax():
    """Profile ARMAX algorithm."""
    print("\n=== Profiling ARMAX ===")
    
    y, u = generate_test_data(1000)
    
    if lp:
        try:
            from sippy.identification.algorithms.armax import ARMAXAlgorithm
            from sippy.utils.compiled_utils import create_regression_matrix_arx_compiled
            
            lp.add_function(ARMAXAlgorithm.identify)
            lp.add_function(create_regression_matrix_arx_compiled)
            lp_wrapper = lp(system_identification)
            
            lp_wrapper(y=y, u=u, id_method='ARMAX', tsample=1.0, AR_orders=(3, 3), ARMA_orders=(2,))
            lp.print_stats()
        except Exception as e:
            print(f"ARMAX profiling failed: {e}")
    else:
        # Simple timing fallback
        try:
            start = time.time()
            system_identification(y=y, u=u, id_method='ARMAX', tsample=1.0, AR_orders=(3, 3), ARMA_orders=(2,))
            elapsed = time.time() - start
            print(f"ARMAX took {elapsed:.3f}s")
        except Exception as e:
            print(f"ARMAX failed: {e}")


if __name__ == '__main__':
    print(f"Numba Available: {NUMBA_AVAILABLE}")
    
    profile_n4sid()
    profile_arma()  
    profile_armax()
    
    print("\n=== Profiling Complete ===")
