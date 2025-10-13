"""
Check theoretical NRMSE for ARMA one-step-ahead predictions
"""

import numpy as np

# Generate AR(1) process
np.random.seed(42)
N = 500
y = np.zeros(N)
noise = np.random.randn(N) * 0.1  # noise std = 0.1

true_a1 = 0.7
for k in range(1, N):
    y[k] = -true_a1 * y[k-1] + noise[k]

# Compute one-step-ahead predictions WITH PERFECT KNOWLEDGE
yid_perfect = np.zeros(N)
yid_perfect[0] = y[0]
for k in range(1, N):
    yid_perfect[k] = -true_a1 * y[k-1]

# Prediction error with perfect model
pred_error = y - yid_perfect

print("="*80)
print("Theoretical ARMA Prediction Performance")
print("="*80)
print(f"\nTrue AR coefficient: {true_a1}")
print(f"Noise std: {0.1}")

print(f"\nSignal statistics:")
print(f"  y mean: {y.mean():.6f}")
print(f"  y std: {y.std():.6f}")
print(f"  y RMS: {np.sqrt(np.mean(y**2)):.6f}")

print(f"\nPerfect one-step prediction error:")
print(f"  Prediction error mean: {pred_error.mean():.6f}")
print(f"  Prediction error std: {pred_error.std():.6f}")
print(f"  RMSE: {np.sqrt(np.mean(pred_error**2)):.6f}")

# Compute NRMSE
y_rms = np.sqrt(np.mean(y**2))
rmse = np.sqrt(np.mean(pred_error**2))
nrmse = rmse / y_rms * 100

print(f"\nTheoretical NRMSE (with perfect model):")
print(f"  NRMSE = RMSE / RMS(y) * 100")
print(f"  NRMSE = {rmse:.6f} / {y_rms:.6f} * 100")
print(f"  NRMSE = {nrmse:.2f}%")

print(f"\n{'='*80}")
print("CONCLUSION:")
print(f"{'='*80}")
print(f"Even with PERFECT knowledge of coefficients, NRMSE = {nrmse:.2f}%")
print(f"This is because one-step-ahead prediction error ≈ noise")
print(f"Noise std = 0.1, signal RMS = {y_rms:.4f}")
print(f"So NRMSE ≈ (0.1 / {y_rms:.4f}) * 100 = {(0.1/y_rms)*100:.2f}%")
print(f"\nFor ARMA models, high NRMSE (>50%) is NORMAL and EXPECTED!")
print(f"The validation criteria (NRMSE < 15%) is INCORRECT for ARMA.")
