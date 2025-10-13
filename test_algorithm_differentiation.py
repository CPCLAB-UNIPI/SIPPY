"""
Test script to verify that different algorithms produce different results.
"""

import numpy as np
import sys
sys.path.insert(0, '/Users/josephj/Workspace/SIPPY/src')

from sippy.identification import SystemIdentification, SystemIdentificationConfig

# Generate test data - SISO system
np.random.seed(42)
N = 500
u = np.random.randn(1, N)
# Simple second-order system response
y = np.zeros((1, N))
for k in range(2, N):
    y[0, k] = 1.5 * y[0, k-1] - 0.7 * y[0, k-2] + 0.5 * u[0, k-1] + 0.3 * u[0, k-2] + np.random.randn() * 0.1

print("=" * 80)
print("TESTING SUBSPACE METHODS (N4SID, MOESP, CVA)")
print("=" * 80)

# Test N4SID, MOESP, CVA
subspace_methods = ['N4SID', 'MOESP', 'CVA']
subspace_results = {}

for method in subspace_methods:
    try:
        config = SystemIdentificationConfig(method=method)
        config.ss_f = 20
        config.ss_fixed_order = 2

        identifier = SystemIdentification(config)
        model = identifier.identify(y=y, u=u)

        print(f"\n{method}:")
        print(f"  A matrix:\n{model.A}")
        print(f"  B matrix:\n{model.B}")
        print(f"  C matrix:\n{model.C}")
        print(f"  D matrix:\n{model.D}")
        print(f"  Vn: {model.Vn}")

        subspace_results[method] = {
            'A': model.A.copy(),
            'B': model.B.copy(),
            'C': model.C.copy(),
            'D': model.D.copy(),
            'Vn': model.Vn
        }
    except Exception as e:
        print(f"\n{method}: FAILED - {e}")

# Compare subspace results
print("\n" + "=" * 80)
print("SUBSPACE METHODS COMPARISON")
print("=" * 80)

if len(subspace_results) >= 2:
    methods = list(subspace_results.keys())
    for i in range(len(methods)):
        for j in range(i+1, len(methods)):
            m1, m2 = methods[i], methods[j]
            A_diff = np.linalg.norm(subspace_results[m1]['A'] - subspace_results[m2]['A'])
            B_diff = np.linalg.norm(subspace_results[m1]['B'] - subspace_results[m2]['B'])
            C_diff = np.linalg.norm(subspace_results[m1]['C'] - subspace_results[m2]['C'])

            print(f"\n{m1} vs {m2}:")
            print(f"  ||A_diff|| = {A_diff:.6e}")
            print(f"  ||B_diff|| = {B_diff:.6e}")
            print(f"  ||C_diff|| = {C_diff:.6e}")

            if A_diff < 1e-10 and B_diff < 1e-10 and C_diff < 1e-10:
                print(f"  ⚠️  IDENTICAL RESULTS (difference < 1e-10)")
            else:
                print(f"  ✓  Results differ as expected")

print("\n" + "=" * 80)
print("TESTING ARMAX MODES (ILLS, RLLS, OPT)")
print("=" * 80)

# Test ARMAX modes - use smaller dataset for OPT (it's slow)
N_armax = 200
u_armax = u[:, :N_armax]
y_armax = y[:, :N_armax]

armax_modes = ['ILLS', 'RLLS', 'OPT']
armax_results = {}

for mode in armax_modes:
    try:
        config = SystemIdentificationConfig(method='ARMAX')
        config.na = 2
        config.nb = 2
        config.nc = 1
        config.nk = 1
        config.max_iterations = 50  # Limit iterations for speed
        config.armx_mode = mode

        identifier = SystemIdentification(config)
        model = identifier.identify(y=y_armax, u=u_armax)

        print(f"\n{mode}:")
        print(f"  A matrix shape: {model.A.shape}, norm: {np.linalg.norm(model.A):.6f}")
        print(f"  B matrix shape: {model.B.shape}, norm: {np.linalg.norm(model.B):.6f}")
        print(f"  C matrix shape: {model.C.shape}, norm: {np.linalg.norm(model.C):.6f}")
        print(f"  Model has Yid: {hasattr(model, 'Yid')}")

        armax_results[mode] = {
            'A': model.A.copy(),
            'B': model.B.copy(),
            'C': model.C.copy(),
            'D': model.D.copy() if model.D is not None else np.zeros_like(model.C)
        }
    except Exception as e:
        print(f"\n{mode}: FAILED - {e}")
        import traceback
        traceback.print_exc()

# Compare ARMAX results
print("\n" + "=" * 80)
print("ARMAX MODES COMPARISON")
print("=" * 80)

if len(armax_results) >= 2:
    modes = list(armax_results.keys())
    for i in range(len(modes)):
        for j in range(i+1, len(modes)):
            m1, m2 = modes[i], modes[j]
            A_diff = np.linalg.norm(armax_results[m1]['A'] - armax_results[m2]['A'])
            B_diff = np.linalg.norm(armax_results[m1]['B'] - armax_results[m2]['B'])
            C_diff = np.linalg.norm(armax_results[m1]['C'] - armax_results[m2]['C'])

            print(f"\n{m1} vs {m2}:")
            print(f"  ||A_diff|| = {A_diff:.6e}")
            print(f"  ||B_diff|| = {B_diff:.6e}")
            print(f"  ||C_diff|| = {C_diff:.6e}")

            if A_diff < 1e-10 and B_diff < 1e-10 and C_diff < 1e-10:
                print(f"  ⚠️  IDENTICAL RESULTS (difference < 1e-10)")
            else:
                print(f"  ✓  Results differ as expected")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)
print("\nIf algorithms show IDENTICAL RESULTS, this confirms TASK 7 issue.")
print("If algorithms show different results, TASK 7 issue is already fixed.")
