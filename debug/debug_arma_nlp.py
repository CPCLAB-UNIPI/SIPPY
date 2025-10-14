"""
Debug ARMA NLP: Quick diagnostic script

Simple diagnostic to verify ARMA NLP implementation is working correctly.
Uses a simple AR(1) process for quick validation.
"""

import numpy as np
import sys
sys.path.insert(0, '/Users/josephj/Workspace/SIPPY-master')

from src.sippy.identification import SystemIdentification, SystemIdentificationConfig
from sippy_unipi import system_identification as master_sysid


def generate_ar1_data(n_samples=300, ar_coeff=-0.7, noise_std=0.1, seed=42):
    """Generate simple AR(1) data for testing."""
    np.random.seed(seed)
    y = np.zeros(n_samples)
    e = np.random.randn(n_samples) * noise_std

    for k in range(1, n_samples):
        y[k] = -ar_coeff * y[k-1] + e[k]  # ar_coeff is in TF form (negative)

    return y.reshape(1, -1)


print("=" * 70)
print("ARMA NLP Debug: Simple AR(1) System")
print("=" * 70)
print("\nTrue system: y[k] = 0.7*y[k-1] + e[k]")
print("True AR coefficient: a1 = -0.7 (TF form)")
print("True MA coefficient: c1 = 0.0 (no MA, but nc=1 for ARMA)")

# Generate data
y = generate_ar1_data(n_samples=300, ar_coeff=-0.7, noise_std=0.1, seed=42)
u = np.zeros((1, y.shape[1]))  # Dummy input for master

print(f"\nData: N={y.shape[1]}, Output RMS={np.std(y):.3f}")

na, nc = 1, 1

# Harold branch
print("\n" + "-" * 70)
print("Harold Branch (NLP or ILLS Method):")
print("-" * 70)

config_h = SystemIdentificationConfig(method="ARMA")
config_h.na, config_h.nc = na, nc
config_h.max_iterations = 100

identifier_h = SystemIdentification(config_h)
model_h = identifier_h.identify(y=y, u=None)

print(f"\nAR coefficients: {model_h.AR_coeffs[0, :]}")
print(f"MA coefficients: {model_h.MA_coeffs[0, :]}")
print(f"Vn: {model_h.Vn:.6f}")

# Check transfer function
if hasattr(model_h, 'H_tf') and model_h.H_tf is not None:
    print(f"\nTransfer Function H(q) = C(q)/A(q):")
    print(f"  Numerator (C):   {model_h.H_tf.num.flatten()}")
    print(f"  Denominator (A): {model_h.H_tf.den.flatten()}")

    # Compute poles
    poles_h = np.roots(model_h.H_tf.den.flatten())
    print(f"  Poles: {poles_h}")
    print(f"  Max pole magnitude: {np.max(np.abs(poles_h)):.4f}")
    stable_h = np.all(np.abs(poles_h) < 1.0)
    print(f"  Stable: {stable_h}")
else:
    print("\n⚠️  No transfer function available")

# Master branch
print("\n" + "-" * 70)
print("Master Branch (ARMAX with u=0 as proxy):")
print("-" * 70)

model_m = master_sysid(
    y, u, "ARMAX",
    na_ord=[na],
    nb_ord=[1],  # Minimal input order
    nc_ord=[nc],
    tsample=1.0
)

print(f"\nVn: {model_m.Vn:.6f}")

# Extract master TF
try:
    import control.matlab as cnt
    H_master = model_m.H

    # Get numerator and denominator
    num_master = np.array(H_master.num[0][0])
    den_master = np.array(H_master.den[0][0])

    print(f"\nTransfer Function H(q) = C(q)/A(q):")
    print(f"  Numerator (C):   {num_master}")
    print(f"  Denominator (A): {den_master}")

    # Compute poles
    poles_master = np.roots(den_master)
    print(f"  Poles: {poles_master}")
    print(f"  Max pole magnitude: {np.max(np.abs(poles_master)):.4f}")
    stable_master = np.all(np.abs(poles_master) < 1.0)
    print(f"  Stable: {stable_master}")

    # Compare TF directly (normalize by first denominator coefficient)
    print("\n" + "=" * 70)
    print("TRANSFER FUNCTION COMPARISON (Normalized):")
    print("=" * 70)

    num_harold_norm = model_h.H_tf.num.flatten() / model_h.H_tf.den.flatten()[0]
    den_harold_norm = model_h.H_tf.den.flatten() / model_h.H_tf.den.flatten()[0]

    num_master_norm = num_master / den_master[0]
    den_master_norm = den_master / den_master[0]

    print(f"\nHarold:")
    print(f"  Numerator:   {num_harold_norm}")
    print(f"  Denominator: {den_harold_norm}")

    print(f"\nMaster:")
    print(f"  Numerator:   {num_master_norm}")
    print(f"  Denominator: {den_master_norm}")

    # Pad to same length for comparison
    max_len_num = max(len(num_harold_norm), len(num_master_norm))
    max_len_den = max(len(den_harold_norm), len(den_master_norm))

    num_h_pad = np.pad(num_harold_norm, (0, max_len_num - len(num_harold_norm)))
    num_m_pad = np.pad(num_master_norm, (0, max_len_num - len(num_master_norm)))

    den_h_pad = np.pad(den_harold_norm, (0, max_len_den - len(den_harold_norm)))
    den_m_pad = np.pad(den_master_norm, (0, max_len_den - len(den_master_norm)))

    num_error = np.max(np.abs(num_h_pad - num_m_pad))
    den_error = np.max(np.abs(den_h_pad - den_m_pad))

    print(f"\nErrors:")
    print(f"  Numerator max abs error:   {num_error:.6e}")
    print(f"  Denominator max abs error: {den_error:.6e}")

    # Coefficient comparison
    print("\n" + "=" * 70)
    print("COEFFICIENT COMPARISON:")
    print("=" * 70)

    # True values
    ar_true = np.array([-0.7])
    ma_true = np.array([0.0])

    ar_harold = model_h.AR_coeffs[0, :]
    ma_harold = model_h.MA_coeffs[0, :]

    print(f"\nTrue AR:     {ar_true}")
    print(f"Harold AR:   {ar_harold}")
    print(f"AR error:    {np.abs(ar_harold - ar_true) / np.abs(ar_true) * 100} %")

    print(f"\nTrue MA:     {ma_true}")
    print(f"Harold MA:   {ma_harold}")
    print(f"MA abs:      {np.abs(ma_harold)} (should be near 0)")

    # Compare Yid
    if hasattr(model_h, 'Yid') and hasattr(model_m, 'Yid'):
        print("\n" + "=" * 70)
        print("ONE-STEP PREDICTION COMPARISON (Yid):")
        print("=" * 70)

        yid_h = model_h.Yid.flatten()
        yid_m = model_m.Yid.flatten()

        min_len = min(len(yid_h), len(yid_m), y.shape[1])
        yid_h = yid_h[:min_len]
        yid_m = yid_m[:min_len]
        y_true = y.flatten()[:min_len]

        # Metrics
        yid_mse = np.mean((yid_h - yid_m) ** 2)
        yid_mae = np.mean(np.abs(yid_h - yid_m))
        y_rms = np.sqrt(np.mean(y_true ** 2))
        yid_nrmse = np.sqrt(yid_mse) / y_rms if y_rms > 1e-10 else float('inf')

        if np.std(yid_h) > 1e-10 and np.std(yid_m) > 1e-10:
            yid_corr = np.corrcoef(yid_h, yid_m)[0, 1]
        else:
            yid_corr = 0.0

        print(f"\nYid Comparison (Harold vs Master):")
        print(f"  MSE:         {yid_mse:.6e}")
        print(f"  MAE:         {yid_mae:.6e}")
        print(f"  NRMSE:       {yid_nrmse*100:.2f}%")
        print(f"  Correlation: {yid_corr:.6f}")

    # Final verdict
    print("\n" + "=" * 70)
    print("VERDICT:")
    print("=" * 70)

    ar_coeff_ok = np.abs(ar_harold - ar_true) / np.abs(ar_true) * 100 < 10
    ma_coeff_ok = np.abs(ma_harold[0]) < 0.1
    tf_ok = num_error < 0.01 and den_error < 0.01

    if hasattr(model_h, 'Yid') and hasattr(model_m, 'Yid'):
        yid_ok = yid_nrmse < 0.10 and yid_corr > 0.95
    else:
        yid_ok = False

    if ar_coeff_ok and ma_coeff_ok and tf_ok:
        if yid_ok:
            print("\n✅ EXCELLENT: All checks passed!")
            print("   - AR coefficient error < 10%")
            print("   - MA coefficient near zero (< 0.1)")
            print("   - Transfer functions match (< 1% error)")
            print("   - One-step predictions match (NRMSE < 10%, Corr > 0.95)")
            print("\n   ARMA NLP implementation is working correctly!")
        else:
            print("\n✅ GOOD: Transfer functions and coefficients match!")
            print("   - AR coefficient error < 10%")
            print("   - MA coefficient near zero (< 0.1)")
            print("   - Transfer functions match (< 1% error)")
            if not hasattr(model_h, 'Yid'):
                print("   ⚠️  Yid not available for comparison")
            else:
                print("   ⚠️  Yid comparison needs improvement")
    elif ar_coeff_ok and ma_coeff_ok:
        print("\n⚠️  PARTIAL: Coefficients look good but transfer functions differ")
        print(f"   - AR coefficient: ✅ (error < 10%)")
        print(f"   - MA coefficient: ✅ (near zero)")
        print(f"   - Transfer function: ❌ (num error: {num_error:.2e}, den error: {den_error:.2e})")
    else:
        print("\n❌ ISSUE: Implementation needs debugging")
        print(f"   - AR coefficient: {'✅' if ar_coeff_ok else '❌'} (error: {np.abs(ar_harold - ar_true) / np.abs(ar_true) * 100}%)")
        print(f"   - MA coefficient: {'✅' if ma_coeff_ok else '❌'} (abs: {np.abs(ma_harold[0])})")
        print(f"   - Transfer function: {'✅' if tf_ok else '❌'} (errors: {num_error:.2e}, {den_error:.2e})")

except Exception as e:
    print(f"\n❌ Error in comparison: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
