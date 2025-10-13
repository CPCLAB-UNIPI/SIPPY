"""
Debug script to reproduce ARMA algorithm failure.
"""
import numpy as np
import sys
import traceback

# Add src to path
sys.path.insert(0, '/Users/josephj/Workspace/SIPPY/src')

from sippy.identification.algorithms.arma import ARMAAlgorithm

def test_arma_basic():
    """Test basic ARMA identification with simple parameters."""
    print("=" * 80)
    print("Testing ARMA Algorithm with basic parameters")
    print("=" * 80)

    # Create simple test data
    np.random.seed(42)
    N = 100

    # Generate AR(1) process: y(k) = 0.7*y(k-1) + e(k)
    y = np.zeros(N)
    noise = np.random.randn(N) * 0.1
    for k in range(1, N):
        y[k] = 0.7 * y[k-1] + noise[k]

    # Reshape to 2D (1 output x N samples)
    y = y.reshape(1, -1)

    print(f"\nData shape: y = {y.shape}")
    print(f"Data range: [{y.min():.3f}, {y.max():.3f}]")

    # Test case 1: na=1, nc=1
    print("\n" + "-" * 80)
    print("Test Case 1: ARMA(1,1) - na=1, nc=1")
    print("-" * 80)

    try:
        arma = ARMAAlgorithm()
        model = arma.identify(y=y, u=None, na=1, nc=1, tsample=1.0)
        print("✓ SUCCESS: ARMA(1,1) identification completed")
        print(f"  Model dimensions: A={model.A.shape}, B={model.B.shape}, C={model.C.shape}, D={model.D.shape}")
        print(f"  Yid shape: {model.Yid.shape if hasattr(model, 'Yid') else 'N/A'}")
    except Exception as e:
        print("✗ FAILURE: ARMA(1,1) identification failed")
        print(f"  Exception: {type(e).__name__}: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
        return False

    # Test case 2: na=2, nc=2
    print("\n" + "-" * 80)
    print("Test Case 2: ARMA(2,2) - na=2, nc=2")
    print("-" * 80)

    try:
        arma = ARMAAlgorithm()
        model = arma.identify(y=y, u=None, na=2, nc=2, tsample=1.0)
        print("✓ SUCCESS: ARMA(2,2) identification completed")
        print(f"  Model dimensions: A={model.A.shape}, B={model.B.shape}, C={model.C.shape}, D={model.D.shape}")
        print(f"  Yid shape: {model.Yid.shape if hasattr(model, 'Yid') else 'N/A'}")
    except Exception as e:
        print("✗ FAILURE: ARMA(2,2) identification failed")
        print(f"  Exception: {type(e).__name__}: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
        return False

    # Test case 3: na=1, nc=0 (edge case)
    print("\n" + "-" * 80)
    print("Test Case 3: AR(1) only - na=1, nc=0 (validation should fail)")
    print("-" * 80)

    try:
        arma = ARMAAlgorithm()
        model = arma.identify(y=y, u=None, na=1, nc=0, tsample=1.0)
        print("✗ UNEXPECTED: Should have raised validation error for nc=0")
    except ValueError as e:
        print(f"✓ SUCCESS: Correctly raised ValueError: {e}")
    except Exception as e:
        print(f"✗ FAILURE: Wrong exception type: {type(e).__name__}: {e}")
        traceback.print_exc()

    print("\n" + "=" * 80)
    print("All tests completed")
    print("=" * 80)
    return True

if __name__ == "__main__":
    test_arma_basic()
