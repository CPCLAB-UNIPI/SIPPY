"""Compare ARARX between harold and master branches."""

import numpy as np
import pandas as pd
import sys

# Add paths for both branches
sys.path.insert(0, '/Users/josephj/Workspace/SIPPY/src')
sys.path.insert(0, '/Users/josephj/Workspace/SIPPY-master')

# Generate test data
np.random.seed(42)
u = np.random.randn(500) * 0.5
y = np.zeros(500)

# Simple SISO system: y[k] = -0.5*y[k-1] + 0.8*u[k-1] + e[k]
for k in range(1, 500):
    y[k] = -0.5 * y[k-1] + 0.8 * u[k-1] + 0.1 * np.random.randn()

print("="*80)
print("Testing ARARX: Harold Branch vs Master Branch")
print("="*80)

# Test with harold branch
from sippy.identification.algorithms.ararx import ARARXAlgorithm
from sippy.identification.iddata import IDData

df = pd.DataFrame({'u': u, 'y': y})
data = IDData(df, inputs=['u'], outputs=['y'], tsample=1.0)

algo_harold = ARARXAlgorithm()
model_harold = algo_harold.identify(iddata=data, na=1, nb=1, nd=1, theta=1)

print("\nHarold Branch Results:")
print(f"  A matrix shape: {model_harold.A.shape}")
print(f"  A matrix:\n{model_harold.A}")
print(f"  B matrix:\n{model_harold.B}")
print(f"  C matrix:\n{model_harold.C}")
print(f"  D matrix:\n{model_harold.D}")

if hasattr(model_harold, 'Yid') and model_harold.Yid is not None:
    fit_harold = 100 * (1 - np.linalg.norm(y - model_harold.Yid.flatten()) / np.linalg.norm(y - np.mean(y)))
    print(f"  Fit percentage: {fit_harold:.2f}%")

# Test with master branch
from sippy_unipi.io_optMIMO import GEN_MIMO_id

# Prepare data for master (needs 2D arrays)
u_master = np.atleast_2d(u)
y_master = np.atleast_2d(y)

# Call master branch ARARX
try:
    DEN, NUM, DENH, NUMH, G, H, Vn, y_id = GEN_MIMO_id(
        "ARARX",
        y_master,
        u_master,
        na=np.array([1]),
        nb=np.array([[1]]),  # Must be 2D: (ny x nu)
        nc=np.array([0]),
        nd=np.array([1]),
        nf=np.array([0]),
        theta=np.array([[1]]),  # Must be 2D: (ny x nu)
        tsample=1.0,
        max_iterations=200,
        st_m=0.95,
        st_c=False,
    )

    print("\nMaster Branch Results:")
    print(f"  DEN (denominator): {DEN}")
    print(f"  NUM (numerator): {NUM}")
    print(f"  NUMH: {NUMH}")
    print(f"  DENH: {DENH}")

    # Convert to state-space for comparison
    import control.matlab as cnt
    # G is already a transfer function from GEN_MIMO_id
    ss_master = cnt.ss(G)

    print(f"\n  Master A matrix shape: {ss_master.A.shape}")
    print(f"  Master A matrix:\n{ss_master.A}")
    print(f"  Master B matrix:\n{ss_master.B}")
    print(f"  Master C matrix:\n{ss_master.C}")
    print(f"  Master D matrix:\n{ss_master.D}")

    fit_master = 100 * (1 - np.linalg.norm(y - y_id.flatten()) / np.linalg.norm(y - np.mean(y)))
    print(f"  Fit percentage: {fit_master:.2f}%")

    # Compare state-space matrices
    print("\n" + "="*80)
    print("Comparison:")
    print("="*80)

    # Handle shape mismatches
    if model_harold.A.shape == ss_master.A.shape:
        A_error = np.linalg.norm(model_harold.A - ss_master.A) / (np.linalg.norm(ss_master.A) + 1e-10)
        print(f"  A matrix relative error: {A_error:.4f} ({A_error*100:.2f}%)")
    else:
        print(f"  A matrix shape mismatch: {model_harold.A.shape} vs {ss_master.A.shape}")

    if model_harold.B.shape == ss_master.B.shape:
        B_error = np.linalg.norm(model_harold.B - ss_master.B) / (np.linalg.norm(ss_master.B) + 1e-10)
        print(f"  B matrix relative error: {B_error:.4f} ({B_error*100:.2f}%)")
    else:
        print(f"  B matrix shape mismatch: {model_harold.B.shape} vs {ss_master.B.shape}")

    # Compare fit quality
    print(f"\n  Fit difference: {abs(fit_harold - fit_master):.2f}%")
    print(f"  Harold fit: {fit_harold:.2f}%")
    print(f"  Master fit: {fit_master:.2f}%")

except Exception as e:
    print(f"\n❌ Master branch failed: {e}")
    import traceback
    traceback.print_exc()
