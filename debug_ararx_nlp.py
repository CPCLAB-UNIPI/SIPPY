"""
Debug ARARX NLP: Simple diagnostic to understand the mismatch
"""

import numpy as np
import sys
sys.path.insert(0, '/Users/josephj/Workspace/SIPPY-master')

from src.sippy.identification import SystemIdentification, SystemIdentificationConfig
from sippy_unipi import system_identification as master_sysid

# Generate simple test data with known system
np.random.seed(42)
N = 200
u = np.random.randn(1, N) * 0.5

# True system: y[k] = 0.5*y[k-1] + 0.3*u[k-1] + noise
y = np.zeros((1, N))
for k in range(1, N):
    y[0, k] = 0.5 * y[0, k-1] + 0.3 * u[0, k-1] + np.random.randn() * 0.01

print("=" * 70)
print("ARARX NLP Debug: Simple SISO System")
print("=" * 70)
print(f"\nTrue system: y[k] = 0.5*y[k-1] + 0.3*u[k-1] + noise")
print(f"Data: N={N}, Input RMS={np.std(u):.3f}, Output RMS={np.std(y):.3f}")

# Model orders (simple: na=1, nb=1, nd=1, theta=1)
na, nb, nd, theta = 1, 1, 1, 1

# Harold branch
print("\n" + "-" * 70)
print("Harold Branch (NLP + Rescaling):")
print("-" * 70)

config = SystemIdentificationConfig(method="ARARX")
config.na, config.nb, config.nd, config.theta = na, nb, nd, theta
config.max_iterations = 200

identifier = SystemIdentification(config)
model_harold = identifier.identify(y=y, u=u)

print(f"Vn: {model_harold.Vn:.6f}")
print(f"A matrix shape: {model_harold.A.shape}")
print(f"B matrix shape: {model_harold.B.shape}")
print(f"A matrix:\n{model_harold.A}")
print(f"B matrix:\n{model_harold.B}")

if hasattr(model_harold, 'G_tf') and model_harold.G_tf is not None:
    print(f"\nTransfer Function G(z):")
    print(f"  Numerator:   {model_harold.G_tf.num}")
    print(f"  Denominator: {model_harold.G_tf.den}")
    
    # Compute poles
    poles_harold = np.roots(model_harold.G_tf.den.flatten())
    print(f"  Poles: {poles_harold}")
    print(f"  Max pole magnitude: {np.max(np.abs(poles_harold)):.4f}")
    stable_harold = np.all(np.abs(poles_harold) < 1.0)
    print(f"  Stable: {stable_harold}")

# Master branch
print("\n" + "-" * 70)
print("Master Branch (Reference):")
print("-" * 70)

model_master = master_sysid(
    y, u, "ARARX",
    ARARX_orders=[[na], [[nb]], [nd], [[theta]]],
    tsample=1.0,
    max_iterations=200
)

print(f"Vn: {model_master.Vn:.6f}")

# Extract master TF
try:
    import control.matlab as cnt
    G_master = model_master.G
    
    # Get numerator and denominator
    num_master = np.array(G_master.num[0][0])
    den_master = np.array(G_master.den[0][0])
    
    print(f"\nTransfer Function G(z):")
    print(f"  Numerator:   {num_master}")
    print(f"  Denominator: {den_master}")
    
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
    
    num_harold_norm = model_harold.G_tf.num.flatten() / model_harold.G_tf.den.flatten()[0]
    den_harold_norm = model_harold.G_tf.den.flatten() / model_harold.G_tf.den.flatten()[0]
    
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
    
    if num_error < 0.01 and den_error < 0.01:
        print("\n✅ EXCELLENT: Transfer functions match within 1% tolerance!")
    elif num_error < 0.1 and den_error < 0.1:
        print("\n✅ GOOD: Transfer functions match within 10% tolerance")
    else:
        print("\n⚠️  ISSUE: Transfer functions don't match")
        print("   This suggests the NLP formulation needs debugging")
    
except Exception as e:
    print(f"Error in comparison: {e}")
    import traceback
    traceback.print_exc()

print("=" * 70)
