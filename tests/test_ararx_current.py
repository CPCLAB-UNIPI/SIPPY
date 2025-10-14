"""Test current ARARX behavior and identify issues."""

import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '/Users/josephj/Workspace/SIPPY/src')

# Generate test data
np.random.seed(42)
u = np.random.randn(500) * 0.5
y = np.zeros(500)

# Simple SISO system: y[k] = 0.5*y[k-1] + 0.8*u[k-1] + e[k]
for k in range(1, 500):
    y[k] = -0.5 * y[k-1] + 0.8 * u[k-1] + 0.1 * np.random.randn()

# Import after setting path
from sippy.identification.algorithms.ararx import ARARXAlgorithm
from sippy.identification.iddata import IDData

# Create DataFrame for IDData
df = pd.DataFrame({'u': u, 'y': y})
data = IDData(df, inputs=['u'], outputs=['y'], tsample=1.0)

# Test ARARX with basic parameters
print("=" * 80)
print("Testing ARARX with na=1, nb=1, nd=1")
print("=" * 80)

try:
    algo = ARARXAlgorithm()
    model = algo.identify(
        iddata=data, na=1, nb=1, nd=1, theta=1
    )

    print("\n✅ ARARX identification succeeded")
    print(f"Model A shape: {model.A.shape}")
    print(f"Model B shape: {model.B.shape}")
    print(f"Model C shape: {model.C.shape}")
    print(f"Model D shape: {model.D.shape}")
    print(f"\nA matrix:\n{model.A}")
    print(f"\nB matrix:\n{model.B}")
    print(f"\nC matrix:\n{model.C}")
    print(f"\nD matrix:\n{model.D}")

    # Check if transfer functions are available
    if hasattr(model, 'G_tf') and model.G_tf is not None:
        print("\n✅ G_tf (deterministic TF) created successfully")
    else:
        print("\n❌ G_tf not available")

    if hasattr(model, 'H_tf') and model.H_tf is not None:
        print("✅ H_tf (noise TF) created successfully")
    else:
        print("❌ H_tf not available")

    if hasattr(model, 'Yid') and model.Yid is not None:
        print(f"✅ Yid (predictions) available, shape: {model.Yid.shape}")
        fit = 100 * (1 - np.linalg.norm(y - model.Yid.flatten()) / np.linalg.norm(y - np.mean(y)))
        print(f"   Fit percentage: {fit:.2f}%")
    else:
        print("❌ Yid not available")

except Exception as e:
    print(f"\n❌ ARARX identification failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("Testing with higher order: na=2, nb=2, nd=2")
print("=" * 80)

try:
    algo2 = ARARXAlgorithm()
    model2 = algo2.identify(
        iddata=data, na=2, nb=2, nd=2, theta=1
    )

    print("\n✅ Higher order ARARX identification succeeded")
    print(f"Model A shape: {model2.A.shape}")
    print(f"Model B shape: {model2.B.shape}")
    print(f"Model C shape: {model2.C.shape}")
    print(f"Model D shape: {model2.D.shape}")

except Exception as e:
    print(f"\n❌ Higher order ARARX identification failed: {e}")
    import traceback
    traceback.print_exc()
