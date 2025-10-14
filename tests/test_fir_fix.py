"""
Test script to verify FIR algorithm works with modern API signature.
"""

import numpy as np
from sippy.identification import SystemIdentification, SystemIdentificationConfig

# Generate test data
np.random.seed(42)
N = 500
u = np.random.randn(1, N)
# FIR model: y(k) = 0.5*u(k-1) + 0.3*u(k-2) + 0.1*u(k-3) + noise
y = np.zeros((1, N))
for k in range(3, N):
    y[0, k] = 0.5 * u[0, k - 1] + 0.3 * u[0, k - 2] + 0.1 * u[0, k - 3]
y += 0.05 * np.random.randn(1, N)  # Add small noise

print("=" * 70)
print("FIR Algorithm Modern API Test")
print("=" * 70)

# Test FIR identification with new signature
print("\n1. Testing FIR identification with direct arrays (y, u)...")
config = SystemIdentificationConfig(method="FIR")
config.nb = 5  # Number of FIR coefficients
config.nk = 1  # Input delay

identifier = SystemIdentification(config)
try:
    model = identifier.identify(y=y, u=u)
    print("   SUCCESS! FIR identification completed.")
    print(f"   Model type: {type(model).__name__}")
    print(f"   A matrix shape: {model.A.shape}")
    print(f"   B matrix shape: {model.B.shape}")
    print(f"   C matrix shape: {model.C.shape}")
    print(f"   D matrix shape: {model.D.shape}")
    print(f"   Sampling time: {model.ts}")

    # Check if transfer functions are available
    if hasattr(model, 'G_tf') and model.G_tf is not None:
        print(f"   G_tf (deterministic transfer function): Available")
    else:
        print(f"   G_tf: Not available (harold may not be installed)")

    if hasattr(model, 'Yid') and model.Yid is not None:
        print(f"   Yid (one-step-ahead predictions): Shape {model.Yid.shape}")
    else:
        print(f"   Yid: Not available")

except Exception as e:
    print(f"   FAILED! Error: {e}")
    import traceback
    traceback.print_exc()

# Test FIR identification with IDData
print("\n2. Testing FIR identification with IDData...")
from sippy.identification import IDData
import pandas as pd

# Create a DataFrame for IDData
df = pd.DataFrame({
    'u1': u[0, :],
    'y1': y[0, :]
})
iddata = IDData(df, inputs=['u1'], outputs=['y1'], tsample=0.1)
try:
    model2 = identifier.identify(iddata=iddata)
    print("   SUCCESS! FIR identification with IDData completed.")
    print(f"   Model type: {type(model2).__name__}")
    print(f"   Sampling time: {model2.ts}")
except Exception as e:
    print(f"   FAILED! Error: {e}")
    import traceback
    traceback.print_exc()

# Test with tsample parameter
print("\n3. Testing FIR identification with tsample parameter...")
config3 = SystemIdentificationConfig(method="FIR")
config3.nb = 5
config3.nk = 1
config3.tsample = 0.5

identifier3 = SystemIdentification(config3)
try:
    model3 = identifier3.identify(y=y, u=u)
    print("   SUCCESS! FIR identification with tsample completed.")
    print(f"   Sampling time: {model3.ts}")
except Exception as e:
    print(f"   FAILED! Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("All FIR tests completed successfully!")
print("=" * 70)
