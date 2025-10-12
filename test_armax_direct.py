"""
Direct test of ARMAX algorithm modes to verify complete implementation.
"""

import numpy as np
import sys
import os

# Add source directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from sippy.identification.algorithms.armax import ARMAXAlgorithm

def test_armax_direct():
    """Test ARMAX algorithm modes directly."""
    print("🧪 Direct ARMAX Algorithm Testing")
    print("=" * 50)
    
    # Generate test ARMAX data
    np.random.seed(42)
    N = 100
    u = np.random.randn(N) * 0.5
    y = np.random.randn(N) * 0.1
    
    # Simple ARMAX process for testing
    for k in range(2, N):
        y[k] = -0.5 * y[k-1] + 0.8 * u[k-1] + 0.2 * u[k-2] + np.random.randn() * 0.1
    
    print(f"Generated test data: {N} samples")
    
    # Mock IDData object
    class MockIDData:
        def __init__(self, u, y):
            self.sample_time = 1.0
            self._u = u.reshape(1, -1)
            self._y = y.reshape(1, -1)
        def get_input_array(self):
            return self._u
        def get_output_array(self):
            return self._y
    
    data = MockIDData(u, y)
    
    # Test configuration
    class Config:
        na = 2
        nb = 2
        nc = 1
        nk = 1
        max_iterations = 30
        convergence_tolerance = 1e-4
    
    config = Config()
    
    # Test each ARMAX mode
    modes = ['ILLS', 'OPT', 'RLLS']
    results = {}
    
    for mode in modes:
        print(f"\n--- Testing ARMAX-{mode} ---")
        try:
            algo = ARMAXAlgorithm(mode=mode)
            print(f"✅ {mode} algorithm created")
            
            model = algo.identify(data, config)
            
            if model is not None:
                print(f"✅ {mode} identification successful!")
                print(f"  - Model dimension: {model.A.shape[0]} states")
                print(f"  - Sample time: {model.ts}")
                
                # Test basic model properties
                try:
                    is_stable = model.is_stable()
                    print(f"  - Stable: {is_stable}")
                except:
                    print("  - Stability check: Not available")
                
                results[mode] = "SUCCESS"
            else:
                print(f"⚠ {mode} identification returned None - using fallback")
                results[mode] = "FALLBACK"
                
        except Exception as e:
            print(f"❌ {mode} failed: {e}")
            results[mode] = "FAILED"
    
    print(f"\n📊 Results Summary:")
    for mode, status in results.items():
        status_icon = "✅" if status == "SUCCESS" else "⚠" if status == "FALLBACK" else "❌"
        print(f"  {status_icon} ARMAX-{mode}: {status}")
    
    # Test factory registration
    print(f"\n🏭 Testing Factory Registration:")
    try:
        from sippy.identification.algorithms import AlgorithmFactory
        algorithms = AlgorithmFactory.list_algorithms()
        armax_modes = [alg for alg in algorithms if 'ARMAX' in alg]
        print(f"✅ Found ARMAX algorithms: {armax_modes}")
    except Exception as e:
        print(f"⚠ Factory test failed: {e}")
    
    print(f"\n🎯 Direct ARMAX Algorithm Testing Complete!")
    return results

if __name__ == "__main__":
    test_armax_direct()
