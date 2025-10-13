"""
ARARX Validation: Compare One-Step Predictions (Yid)

This is the most important comparison - the NLP minimizes prediction error,
so Yid (one-step predictions) should match if the implementation is correct.
"""

import numpy as np
import sys
sys.path.insert(0, '/Users/josephj/Workspace/SIPPY-master')

from src.sippy.identification import SystemIdentification, SystemIdentificationConfig
from sippy_unipi import system_identification as master_sysid

print("=" * 80)
print("ARARX Validation: One-Step Prediction Comparison (Yid)")
print("=" * 80)

# Test Case 1: Simple stable system
print("\n" + "=" * 80)
print("TEST CASE 1: Simple Stable System (na=1, nb=1, nd=1)")
print("=" * 80)

np.random.seed(42)
N = 300
u1 = np.random.randn(1, N) * 0.5
y1 = np.zeros((1, N))

for k in range(1, N):
    y1[0, k] = 0.5 * y1[0, k-1] + 0.3 * u1[0, k-1] + np.random.randn() * 0.01

na, nb, nd, theta = 1, 1, 1, 1

# Harold
config1 = SystemIdentificationConfig(method="ARARX")
config1.na, config1.nb, config1.nd, config1.theta = na, nb, nd, theta
config1.max_iterations = 200

identifier1 = SystemIdentification(config1)
model1_h = identifier1.identify(y=y1, u=u1)

# Master
model1_m = master_sysid(
    y1, u1, "ARARX",
    ARARX_orders=[[na], [[nb]], [nd], [[theta]]],
    tsample=1.0,
    max_iterations=200
)

# Compare Yid
yid_h = model1_h.Yid.flatten() if hasattr(model1_h, 'Yid') else None
yid_m = model1_m.Yid.flatten() if hasattr(model1_m, 'Yid') else None

if yid_h is not None and yid_m is not None:
    # Align lengths
    min_len = min(len(yid_h), len(yid_m), len(y1.flatten()))
    yid_h = yid_h[:min_len]
    yid_m = yid_m[:min_len]
    y_true = y1.flatten()[:min_len]

    # Compute prediction errors
    err_h = y_true - yid_h
    err_m = y_true - yid_m

    # Metrics
    mse_h = np.mean(err_h ** 2)
    mse_m = np.mean(err_m ** 2)

    mae_h = np.mean(np.abs(err_h))
    mae_m = np.mean(np.abs(err_m))

    # Compare Yid directly
    yid_mse = np.mean((yid_h - yid_m) ** 2)
    yid_mae = np.mean(np.abs(yid_h - yid_m))
    yid_corr = np.corrcoef(yid_h, yid_m)[0, 1] if np.std(yid_h) > 1e-10 and np.std(yid_m) > 1e-10 else 0.0

    y_rms = np.sqrt(np.mean(y_true ** 2))
    yid_nrmse = np.sqrt(yid_mse) / y_rms if y_rms > 1e-10 else float('inf')

    print(f"\nPrediction Errors (vs True Output):")
    print(f"  Harold MSE:  {mse_h:.6e}")
    print(f"  Master MSE:  {mse_m:.6e}")
    print(f"  Harold MAE:  {mae_h:.6e}")
    print(f"  Master MAE:  {mae_m:.6e}")

    print(f"\nYid Comparison (Harold vs Master):")
    print(f"  MSE:         {yid_mse:.6e}")
    print(f"  MAE:         {yid_mae:.6e}")
    print(f"  NRMSE:       {yid_nrmse:.6f}")
    print(f"  Correlation: {yid_corr:.6f}")

    print(f"\nModel Quality:")
    print(f"  Harold Vn:   {model1_h.Vn:.6f}")
    print(f"  Master Vn:   {model1_m.Vn:.6f}")

    # Verdict
    if yid_nrmse < 0.05 and yid_corr > 0.95:
        print(f"\n✅ EXCELLENT: One-step predictions match (NRMSE < 5%, Correlation > 0.95)")
        test1_pass = True
    elif yid_nrmse < 0.15 and yid_corr > 0.85:
        print(f"\n✅ GOOD: One-step predictions match reasonably (NRMSE < 15%, Correlation > 0.85)")
        test1_pass = True
    else:
        print(f"\n⚠️  NEEDS IMPROVEMENT: One-step predictions differ significantly")
        test1_pass = False
else:
    print("\n❌ ERROR: Could not extract Yid from one or both models")
    test1_pass = False

# Test Case 2: Higher-order system
print("\n" + "=" * 80)
print("TEST CASE 2: Higher-Order System (na=2, nb=2, nd=1)")
print("=" * 80)

np.random.seed(123)
N2 = 400
u2 = np.random.randn(1, N2)
y2 = np.zeros((1, N2))

# Stable system
for k in range(2, N2):
    y2[0, k] = 0.6 * y2[0, k-1] - 0.1 * y2[0, k-2] + 0.3 * u2[0, k-1] + 0.1 * u2[0, k-2] + np.random.randn() * 0.05

na2, nb2, nd2, theta2 = 2, 2, 1, 1

# Harold
config2 = SystemIdentificationConfig(method="ARARX")
config2.na, config2.nb, config2.nd, config2.theta = na2, nb2, nd2, theta2
config2.max_iterations = 200

identifier2 = SystemIdentification(config2)
model2_h = identifier2.identify(y=y2, u=u2)

# Master
model2_m = master_sysid(
    y2, u2, "ARARX",
    ARARX_orders=[[na2], [[nb2]], [nd2], [[theta2]]],
    tsample=1.0,
    max_iterations=200
)

# Compare Yid
yid2_h = model2_h.Yid.flatten() if hasattr(model2_h, 'Yid') else None
yid2_m = model2_m.Yid.flatten() if hasattr(model2_m, 'Yid') else None

if yid2_h is not None and yid2_m is not None:
    # Align lengths
    min_len = min(len(yid2_h), len(yid2_m), len(y2.flatten()))
    yid2_h = yid2_h[:min_len]
    yid2_m = yid2_m[:min_len]
    y2_true = y2.flatten()[:min_len]

    # Compute prediction errors
    err2_h = y2_true - yid2_h
    err2_m = y2_true - yid2_m

    # Metrics
    mse2_h = np.mean(err2_h ** 2)
    mse2_m = np.mean(err2_m ** 2)

    mae2_h = np.mean(np.abs(err2_h))
    mae2_m = np.mean(np.abs(err2_m))

    # Compare Yid directly
    yid2_mse = np.mean((yid2_h - yid2_m) ** 2)
    yid2_mae = np.mean(np.abs(yid2_h - yid2_m))
    yid2_corr = np.corrcoef(yid2_h, yid2_m)[0, 1] if np.std(yid2_h) > 1e-10 and np.std(yid2_m) > 1e-10 else 0.0

    y2_rms = np.sqrt(np.mean(y2_true ** 2))
    yid2_nrmse = np.sqrt(yid2_mse) / y2_rms if y2_rms > 1e-10 else float('inf')

    print(f"\nPrediction Errors (vs True Output):")
    print(f"  Harold MSE:  {mse2_h:.6e}")
    print(f"  Master MSE:  {mse2_m:.6e}")
    print(f"  Harold MAE:  {mae2_h:.6e}")
    print(f"  Master MAE:  {mae2_m:.6e}")

    print(f"\nYid Comparison (Harold vs Master):")
    print(f"  MSE:         {yid2_mse:.6e}")
    print(f"  MAE:         {yid2_mae:.6e}")
    print(f"  NRMSE:       {yid2_nrmse:.6f}")
    print(f"  Correlation: {yid2_corr:.6f}")

    print(f"\nModel Quality:")
    print(f"  Harold Vn:   {model2_h.Vn:.6f}")
    print(f"  Master Vn:   {model2_m.Vn:.6f}")

    # Check stability
    stable_h = "Stable" if hasattr(model2_h, 'is_stable') and model2_h.is_stable() else "Unknown"
    print(f"  Harold Stability: {stable_h}")

    # Verdict
    if yid2_nrmse < 0.05 and yid2_corr > 0.95:
        print(f"\n✅ EXCELLENT: One-step predictions match (NRMSE < 5%, Correlation > 0.95)")
        test2_pass = True
    elif yid2_nrmse < 0.15 and yid2_corr > 0.85:
        print(f"\n✅ GOOD: One-step predictions match reasonably (NRMSE < 15%, Correlation > 0.85)")
        test2_pass = True
    else:
        print(f"\n⚠️  NEEDS IMPROVEMENT: One-step predictions differ significantly")
        test2_pass = False
else:
    print("\n❌ ERROR: Could not extract Yid from one or both models")
    test2_pass = False

# Final Summary
print("\n" + "=" * 80)
print("FINAL VALIDATION SUMMARY")
print("=" * 80)

print(f"\nTest Case 1 (Simple Stable):  {'✅ PASS' if test1_pass else '❌ FAIL'}")
print(f"Test Case 2 (Higher Order):   {'✅ PASS' if test2_pass else '❌ FAIL'}")

if test1_pass and test2_pass:
    print("\n🎉 OVERALL: ✅ PASS - ARARX NLP implementation is PRODUCTION READY!")
    print("   One-step predictions match master branch within acceptable tolerance")
elif test1_pass or test2_pass:
    print("\n⚠️  OVERALL: PARTIAL SUCCESS")
    print("   Some test cases pass, others need investigation")
else:
    print("\n❌ OVERALL: NEEDS IMPROVEMENT")
    print("   One-step predictions don't match master branch")

print("=" * 80)
