"""
Direct comparison: Harold ARMA vs Master ARMA on same data
"""

import numpy as np
import sys

# Harold branch
from src.sippy.identification import SystemIdentification, SystemIdentificationConfig

# Master branch
sys.path.insert(0, '/Users/josephj/Workspace/SIPPY-master')
from sippy_unipi import system_identification as master_sysid

# Generate simple AR(1) test data
np.random.seed(42)
N = 200
y = np.zeros((1, N))
noise = np.random.randn(N) * 0.1

true_a1 = 0.7
for k in range(1, N):
    y[0, k] = -true_a1 * y[0, k-1] + noise[k]

print("="*80)
print("Direct Harold vs Master Comparison - AR(1)")
print("="*80)
print(f"\nTrue AR coefficient: {true_a1:.6f}")
print(f"Data: N={N}, mean={y.mean():.6f}, std={y.std():.6f}")

# Harold branch
print("\n" + "-"*80)
print("Harold Branch (ARMA NLP):")
print("-"*80)

config_h = SystemIdentificationConfig(method="ARMA")
config_h.na = 1
config_h.nc = 1
config_h.max_iterations = 200

identifier_h = SystemIdentification(config_h)
model_h = identifier_h.identify(y=y)

ar_h = model_h.AR_coeffs[0, 0]
ma_h = model_h.MA_coeffs[0, 0]

print(f"AR coeff: {ar_h:.6f} (error: {abs(ar_h - true_a1)/true_a1*100:.2f}%)")
print(f"MA coeff: {ma_h:.6f}")
print(f"Vn: {model_h.Vn:.6f}")

yid_nrmse_h = np.sqrt(np.mean((y.flatten() - model_h.Yid.flatten())**2)) / np.sqrt(np.mean(y.flatten()**2)) * 100
print(f"Yid NRMSE: {yid_nrmse_h:.2f}%")

# Master branch
print("\n" + "-"*80)
print("Master Branch (ARMA):")
print("-"*80)

try:
    # Master requires dummy input
    u_dummy = np.zeros((1, N))

    print(f"Passing to master: y.shape={y.shape}, u.shape={u_dummy.shape}")
    print(f"y stats: min={y.min():.6f}, max={y.max():.6f}, mean={y.mean():.6f}, std={y.std():.6f}")

    model_m = master_sysid(
        y, u_dummy, "ARMA",
        ARMA_orders=[1, 1, 0],  # na=1, nc=1, theta=0
        tsample=1.0
    )

    # Extract coefficients
    if hasattr(model_m, 'THETA'):
        theta_m = model_m.THETA.flatten()
        ar_m = theta_m[0]
        ma_m = theta_m[1] if len(theta_m) > 1 else 0

        print(f"AR coeff: {ar_m:.6f} (error: {abs(ar_m - true_a1)/true_a1*100:.2f}%)")
        print(f"MA coeff: {ma_m:.6f}")

        if hasattr(model_m, 'Vn'):
            print(f"Vn: {model_m.Vn:.6f}")

        if hasattr(model_m, 'Y_id'):
            yid_m = model_m.Y_id.flatten()
            yid_nrmse_m = np.sqrt(np.mean((y.flatten() - yid_m)**2)) / np.sqrt(np.mean(y.flatten()**2)) * 100
            print(f"Yid NRMSE: {yid_nrmse_m:.2f}%")

        # Comparison
        print("\n" + "="*80)
        print("COMPARISON:")
        print("="*80)
        print(f"AR difference: {abs(ar_h - ar_m)/abs(ar_m)*100:.2f}%")
        print(f"MA difference: {abs(ma_h - ma_m)/abs(ma_m)*100:.2f}% (if ma_m != 0)")

        if hasattr(model_m, 'Y_id'):
            yid_diff = np.sqrt(np.mean((model_h.Yid.flatten() - yid_m)**2))
            print(f"Yid RMS difference: {yid_diff:.6f}")

except Exception as e:
    print(f"Master identification failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
