"""
ARARX Validation with Stable System
Test on a simple stable system where both branches should converge
"""

import numpy as np
import sys
sys.path.insert(0, '/Users/josephj/Workspace/SIPPY-master')

from src.sippy.identification import SystemIdentification, SystemIdentificationConfig
from sippy_unipi import system_identification as master_sysid

# Generate test data from a STABLE system
np.random.seed(42)
N = 300
u = np.random.randn(1, N) * 0.5

# True stable system: y[k] = 0.5*y[k-1] + 0.3*u[k-1] + noise
y = np.zeros((1, N))
for k in range(1, N):
    y[0, k] = 0.5 * y[0, k-1] + 0.3 * u[0, k-1] + np.random.randn() * 0.01

print("=" * 80)
print("ARARX Validation: Simple Stable System")
print("=" * 80)
print(f"\nTrue system: y[k] = 0.5*y[k-1] + 0.3*u[k-1] + noise")
print(f"Data: N={N}, SISO, guaranteed stable")

# Simple model orders
na, nb, nd, theta = 1, 1, 1, 1

# Harold branch
print("\n" + "-" * 80)
print("Harold Branch:")
print("-" * 80)

config = SystemIdentificationConfig(method="ARARX")
config.na = na
config.nb = nb
config.nd = nd
config.theta = theta
config.max_iterations = 200

identifier = SystemIdentification(config)
model_harold = identifier.identify(y=y, u=u)

print(f"Vn: {model_harold.Vn:.6f}")

if hasattr(model_harold, 'G_tf') and model_harold.G_tf is not None:
    print(f"\nG_tf numerator:   {model_harold.G_tf.num}")
    print(f"G_tf denominator: {model_harold.G_tf.den}")

    poles_h = np.roots(model_harold.G_tf.den.flatten())
    print(f"Poles: {poles_h}")
    print(f"Max pole magnitude: {np.max(np.abs(poles_h)):.6f}")
    print(f"Stable: {np.all(np.abs(poles_h) < 1.0)}")

# Master branch
print("\n" + "-" * 80)
print("Master Branch:")
print("-" * 80)

model_master = master_sysid(
    y, u, "ARARX",
    ARARX_orders=[[na], [[nb]], [nd], [[theta]]],
    tsample=1.0,
    max_iterations=200
)

print(f"Vn: {model_master.Vn:.6f}")

try:
    import control.matlab as cnt
    G_master = model_master.G

    num_master = np.array(G_master.num[0][0])
    den_master = np.array(G_master.den[0][0])

    print(f"\nG numerator:   {num_master}")
    print(f"G denominator: {den_master}")

    poles_m = np.roots(den_master)
    print(f"Poles: {poles_m}")
    print(f"Max pole magnitude: {np.max(np.abs(poles_m)):.6f}")
    print(f"Stable: {np.all(np.abs(poles_m) < 1.0)}")

except Exception as e:
    print(f"Error: {e}")

# Time domain comparison (step response)
print("\n" + "=" * 80)
print("TIME DOMAIN COMPARISON (Step Response)")
print("=" * 80)

n_steps = 50

# Harold step response
A_h, B_h, C_h, D_h = model_harold.A, model_harold.B, model_harold.C, model_harold.D
nx = A_h.shape[0]
x_h = np.zeros((nx, 1))
u_step = np.ones((1, 1))
y_step_h = np.zeros(n_steps)

for k in range(n_steps):
    y_step_h[k] = (C_h @ x_h + D_h @ u_step)[0, 0]
    x_h = A_h @ x_h + B_h @ u_step

# Master step response
try:
    import control.matlab as cnt
    t = np.arange(n_steps)
    _, y_step_m = cnt.step(model_master.G, t)
    y_step_m = y_step_m.flatten()

    # Compute metrics
    min_len = min(len(y_step_h), len(y_step_m))
    y_step_h = y_step_h[:min_len]
    y_step_m = y_step_m[:min_len]

    mse = np.mean((y_step_h - y_step_m) ** 2)
    mae = np.mean(np.abs(y_step_h - y_step_m))
    max_error = np.max(np.abs(y_step_h - y_step_m))

    y_m_rms = np.sqrt(np.mean(y_step_m ** 2))
    if y_m_rms > 1e-10:
        nrmse = np.sqrt(mse) / y_m_rms
    else:
        nrmse = float('inf')

    correlation = np.corrcoef(y_step_h, y_step_m)[0, 1] if np.std(y_step_h) > 1e-10 and np.std(y_step_m) > 1e-10 else 0.0

    print(f"\nStep Response Metrics:")
    print(f"  MSE:          {mse:.6e}")
    print(f"  MAE:          {mae:.6e}")
    print(f"  Max Error:    {max_error:.6e}")
    print(f"  NRMSE:        {nrmse:.6f}")
    print(f"  Correlation:  {correlation:.6f}")

    if nrmse < 0.01 and correlation > 0.99:
        print(f"\n✅ EXCELLENT: Step responses match (NRMSE < 1%, Correlation > 0.99)")
    elif nrmse < 0.1 and correlation > 0.95:
        print(f"\n✅ GOOD: Step responses match reasonably (NRMSE < 10%, Correlation > 0.95)")
    else:
        print(f"\n⚠️  NEEDS IMPROVEMENT: Step responses differ (NRMSE = {nrmse:.2%})")

except Exception as e:
    print(f"Error in step response: {e}")

# Transfer function direct comparison
print("\n" + "=" * 80)
print("TRANSFER FUNCTION COEFFICIENT COMPARISON")
print("=" * 80)

if hasattr(model_harold, 'G_tf') and model_harold.G_tf is not None:
    num_h = model_harold.G_tf.num.flatten()
    den_h = model_harold.G_tf.den.flatten()

    num_h_norm = num_h / den_h[0]
    den_h_norm = den_h / den_h[0]

    num_m_norm = num_master / den_master[0]
    den_m_norm = den_master / den_master[0]

    print(f"\nHarold:")
    print(f"  Numerator:   {num_h_norm}")
    print(f"  Denominator: {den_h_norm}")

    print(f"\nMaster:")
    print(f"  Numerator:   {num_m_norm}")
    print(f"  Denominator: {den_m_norm}")

    # Pad to same length
    max_len_num = max(len(num_h_norm), len(num_m_norm))
    max_len_den = max(len(den_h_norm), len(den_m_norm))

    num_h_pad = np.pad(num_h_norm, (0, max_len_num - len(num_h_norm)))
    num_m_pad = np.pad(num_m_norm, (0, max_len_num - len(num_m_norm)))

    den_h_pad = np.pad(den_h_norm, (0, max_len_den - len(den_h_norm)))
    den_m_pad = np.pad(den_m_norm, (0, max_len_den - len(den_m_norm)))

    num_error = np.max(np.abs(num_h_pad - num_m_pad))
    den_error = np.max(np.abs(den_h_pad - den_m_pad))

    print(f"\nErrors:")
    print(f"  Numerator max abs error:   {num_error:.6e}")
    print(f"  Denominator max abs error: {den_error:.6e}")

    if num_error < 0.01 and den_error < 0.01:
        print("\n✅ EXCELLENT: Transfer functions match within 1% tolerance!")
    elif num_error < 0.1 and den_error < 0.1:
        print("\n✅ GOOD: Transfer functions match within 10% tolerance")
    else:
        print(f"\n⚠️  ISSUE: Transfer functions differ (num_error={num_error:.2%}, den_error={den_error:.2%})")

print("=" * 80)
