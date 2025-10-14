"""
Detailed NLP Diagnostic - Check what's happening inside ARMA NLP
"""

import numpy as np
from src.sippy.identification import SystemIdentification, SystemIdentificationConfig

# Generate simple AR(1) test data
np.random.seed(42)
N = 100  # Smaller dataset for debugging
y = np.zeros((1, N))
noise = np.random.randn(N) * 0.1

true_a1 = 0.7
for k in range(1, N):
    y[0, k] = -true_a1 * y[0, k-1] + noise[k]

print("="*80)
print("ARMA NLP Detailed Diagnostic")
print("="*80)
print(f"\nData shape: {y.shape}")
print(f"Data range: [{y.min():.6f}, {y.max():.6f}]")
print(f"Data mean: {y.mean():.6f}")
print(f"Data std: {y.std():.6f}")
print(f"\nTrue AR coefficient: {true_a1:.6f}")

# Identify
config = SystemIdentificationConfig(method="ARMA")
config.na = 1
config.nc = 1
config.max_iterations = 200

identifier = SystemIdentification(config)
model = identifier.identify(y=y)

print(f"\n--- Results ---")
print(f"Estimated AR coefficients: {model.AR_coeffs}")
print(f"Estimated MA coefficients: {model.MA_coeffs}")
print(f"Noise variance (Vn): {model.Vn:.6f}")

# Check Yid
print(f"\n--- Yid Analysis ---")
print(f"Yid shape: {model.Yid.shape}")
print(f"Yid range: [{model.Yid.min():.6f}, {model.Yid.max():.6f}]")
print(f"Yid mean: {model.Yid.mean():.6f}")
print(f"Yid std: {model.Yid.std():.6f}")

print(f"\nOriginal y range: [{y.min():.6f}, {y.max():.6f}]")
print(f"Original y mean: {y.mean():.6f}")
print(f"Original y std: {y.std():.6f}")

# Compute errors
residuals = y.flatten() - model.Yid.flatten()
print(f"\n--- Residuals ---")
print(f"Residual range: [{residuals.min():.6f}, {residuals.max():.6f}]")
print(f"Residual mean: {residuals.mean():.6f}")
print(f"Residual std: {residuals.std():.6f}")

# Compute NRMSE
mse = np.mean((y.flatten() - model.Yid.flatten())**2)
y_rms = np.sqrt(np.mean(y.flatten()**2))
nrmse = np.sqrt(mse) / y_rms * 100

print(f"\n--- Metrics ---")
print(f"MSE: {mse:.6f}")
print(f"RMSE: {np.sqrt(mse):.6f}")
print(f"y RMS: {y_rms:.6f}")
print(f"NRMSE: {nrmse:.2f}%")

# AR coefficient error
ar_error = abs(model.AR_coeffs[0, 0] - true_a1) / abs(true_a1) * 100
print(f"\nAR coefficient error: {ar_error:.2f}%")

# Sample some values
print(f"\n--- Sample Values (first 10) ---")
print("k | y[k] | Yid[k] | error")
print("-" * 50)
for k in range(10):
    print(f"{k:2d} | {y[0, k]:7.4f} | {model.Yid[0, k]:7.4f} | {y[0, k] - model.Yid[0, k]:7.4f}")

# Check if Yid is just constant or has structure
yid_variation = np.std(model.Yid.flatten())
y_variation = np.std(y.flatten())
print(f"\n--- Variation Check ---")
print(f"y variation (std): {y_variation:.6f}")
print(f"Yid variation (std): {yid_variation:.6f}")
print(f"Variation ratio (Yid/y): {yid_variation/y_variation:.6f}")

if nrmse > 50:
    print(f"\n❌ HIGH NRMSE DETECTED!")
    print(f"   This suggests Yid is not properly tracking y")
    print(f"   Possible issues:")
    print(f"   1. Rescaling problem")
    print(f"   2. NLP constraints not satisfied")
    print(f"   3. Optimization failed to converge")
else:
    print(f"\n✅ NRMSE looks reasonable")
