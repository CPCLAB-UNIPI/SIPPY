"""
Debug script to test ARMA through SystemIdentification class (mimics validation test).
"""
import numpy as np
import sys
import traceback

# Add src to path
sys.path.insert(0, '/Users/josephj/Workspace/SIPPY/src')

from sippy.identification import SystemIdentification, SystemIdentificationConfig

def test_arma_via_systemidentification():
    """Test ARMA using SystemIdentification class like the validation test does."""
    print("=" * 80)
    print("Testing ARMA via SystemIdentification (same path as validation test)")
    print("=" * 80)

    # Create test data matching validation test fixture
    np.random.seed(789)
    npts = 300
    u = np.random.randn(1, npts)
    y = np.zeros((1, npts))

    # Generate ARX system
    for i in range(1, npts):
        y[0, i] = 0.7 * y[0, i - 1] + 0.5 * u[0, i - 1] + 0.05 * np.random.randn()

    print(f"\nData shape: y={y.shape}, u={u.shape}")

    # Test case: ARMA with SystemIdentification class (same as validation test)
    print("\n" + "-" * 80)
    print("Test: ARMA via SystemIdentification.identify(y=..., u=None)")
    print("-" * 80)

    try:
        # This is exactly how the validation test does it (line 1046-1052)
        config = SystemIdentificationConfig(method="ARMA")
        config.na = 1
        config.nc = 1
        identifier = SystemIdentification(config)

        print(f"Config: method={config.method}, na={config.na}, nc={config.nc}")
        print(f"Calling identifier.identify(y=data['y'], u=None)...")

        model_harold = identifier.identify(y=y, u=None)

        print("✓ SUCCESS: ARMA identification via SystemIdentification completed")
        print(f"  Model dimensions: A={model_harold.A.shape}, B={model_harold.B.shape}")
        print(f"  C={model_harold.C.shape}, D={model_harold.D.shape}")
        print(f"  Yid shape: {model_harold.Yid.shape if hasattr(model_harold, 'Yid') else 'N/A'}")
        print(f"  H_tf: {model_harold.H_tf if hasattr(model_harold, 'H_tf') else 'N/A'}")

    except Exception as e:
        print("✗ FAILURE: ARMA identification via SystemIdentification failed")
        print(f"  Exception: {type(e).__name__}: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
        return False

    print("\n" + "=" * 80)
    print("Test completed")
    print("=" * 80)
    return True

if __name__ == "__main__":
    test_arma_via_systemidentification()
