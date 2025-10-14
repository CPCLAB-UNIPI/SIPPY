"""
Simple ARMA Diagnostic: Compare harold vs master on basic test case
"""

import numpy as np
import sys
sys.path.insert(0, '/Users/josephj/Workspace/SIPPY-master')

from src.sippy.identification import SystemIdentification, SystemIdentificationConfig
from sippy_unipi import system_identification as master_sysid

# Generate simple AR(1) test data with known parameters
np.random.seed(42)
N = 500

# True AR(1) process: y[k] = -0.7 * y[k-1] + e[k]
# In transfer function form: (1 + 0.7*q^-1) y[k] = e[k]
# So a1 = 0.7

true_a1 = 0.7
y = np.zeros((1, N))
noise = np.random.randn(N) * 0.1

for k in range(1, N):
    y[0, k] = -true_a1 * y[0, k-1] + noise[k]

print("=" * 80)
print("ARMA Simple Diagnostic: AR(1) Process")
print("=" * 80)
print(f"\nTrue process: y[k] = -0.7 * y[k-1] + e[k]")
print(f"True AR coefficient (a1): {true_a1}")
print(f"Data: N={N}, noise std={0.1}")

# Harold branch - ARMA
print("\n" + "-" * 80)
print("Harold Branch (ARMA):")
print("-" * 80)

config_h = SystemIdentificationConfig(method="ARMA")
config_h.na = 1
config_h.nc = 1  # Small MA term to allow algorithm flexibility
config_h.max_iterations = 100

identifier_h = SystemIdentification(config_h)
model_h = identifier_h.identify(y=y)

print(f"\nEstimated AR coefficients: {model_h.AR_coeffs}")
print(f"Estimated MA coefficients: {model_h.MA_coeffs}")

ar_error = np.abs(model_h.AR_coeffs[0, 0] - true_a1) / true_a1 * 100
print(f"\nAR coefficient error: {ar_error:.2f}%")

# Master branch - ARMA
print("\n" + "-" * 80)
print("Master Branch (ARMA via optimization):")
print("-" * 80)

try:
    # Master needs dummy input for ARMA (will be ignored)
    u_dummy = np.zeros((1, N))

    model_m = master_sysid(
        y, u_dummy, "ARMA",
        ARMA_orders=[1, 1, 0],  # na=1, nc=1, theta=0
        tsample=1.0
    )

    # Extract coefficients from master
    # Master stores them in THETA attribute
    if hasattr(model_m, 'THETA'):
        theta_m = model_m.THETA.flatten()
        print(f"\nMaster THETA: {theta_m}")

        # Parse: THETA contains [a1, c1] for ARMA(1,1)
        if len(theta_m) >= 1:
            a1_master = theta_m[0]
            c1_master = theta_m[1] if len(theta_m) > 1 else 0

            print(f"Master AR coefficient (a1): {a1_master}")
            print(f"Master MA coefficient (c1): {c1_master}")

            ar_error_m = np.abs(a1_master - true_a1) / true_a1 * 100
            print(f"\nMaster AR error: {ar_error_m:.2f}%")

            # Compare harold vs master
            print("\n" + "=" * 80)
            print("COMPARISON:")
            print("=" * 80)
            print(f"True AR:    {true_a1:.6f}")
            print(f"Harold AR:  {model_h.AR_coeffs[0, 0]:.6f}")
            print(f"Master AR:  {a1_master:.6f}")
            print(f"\nHarold error: {ar_error:.2f}%")
            print(f"Master error: {ar_error_m:.2f}%")

            # Check if both match
            harold_vs_master = np.abs(model_h.AR_coeffs[0, 0] - a1_master) / np.abs(a1_master) * 100
            print(f"\nHarold vs Master: {harold_vs_master:.2f}% difference")

            if harold_vs_master < 5:
                print("✅ Harold matches master within 5%")
            elif harold_vs_master < 15:
                print("⚠️  Harold differs from master by 5-15% (acceptable)")
            else:
                print("❌ Harold significantly differs from master (>15%)")

except Exception as e:
    print(f"Master identification failed: {e}")
    import traceback
    traceback.print_exc()

print("=" * 80)
