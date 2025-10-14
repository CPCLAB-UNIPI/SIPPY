"""
Detailed ARARX Debug: Understand the validation discrepancy
"""

import numpy as np
import sys
sys.path.insert(0, '/Users/josephj/Workspace/SIPPY-master')

from src.sippy.identification import SystemIdentification, SystemIdentificationConfig
from sippy_unipi import system_identification as master_sysid

# Generate test data (same as validation script)
np.random.seed(42)
N = 500
u = np.random.randn(1, N)
y = np.zeros((1, N))

# True system: SISO with some dynamics
for k in range(2, N):
    y[0, k] = 0.6 * y[0, k-1] - 0.1 * y[0, k-2] + 0.3 * u[0, k-1] + 0.1 * u[0, k-2] + np.random.randn() * 0.01

print("=" * 80)
print("ARARX Detailed Debug")
print("=" * 80)

# Model orders
na, nb, nd, theta = 2, 2, 1, 1

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
print(f"A:\n{model_harold.A}")
print(f"B:\n{model_harold.B}")
print(f"C:\n{model_harold.C}")
print(f"D:\n{model_harold.D}")

if hasattr(model_harold, 'G_tf') and model_harold.G_tf is not None:
    print(f"\nG_tf numerator:   {model_harold.G_tf.num}")
    print(f"G_tf denominator: {model_harold.G_tf.den}")

    # Check poles
    poles_h = np.roots(model_harold.G_tf.den.flatten())
    print(f"G_tf poles: {poles_h}")
    print(f"Max pole magnitude: {np.max(np.abs(poles_h)):.6f}")
    print(f"Stable: {np.all(np.abs(poles_h) < 1.0)}")

if hasattr(model_harold, 'H_tf') and model_harold.H_tf is not None:
    print(f"\nH_tf numerator:   {model_harold.H_tf.num}")
    print(f"H_tf denominator: {model_harold.H_tf.den}")

    # Check poles
    poles_h_noise = np.roots(model_harold.H_tf.den.flatten())
    print(f"H_tf poles: {poles_h_noise}")
    print(f"Max pole magnitude: {np.max(np.abs(poles_h_noise)):.6f}")

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

# Extract master TF
try:
    import control.matlab as cnt
    G_master = model_master.G

    # Get numerator and denominator
    num_master = np.array(G_master.num[0][0])
    den_master = np.array(G_master.den[0][0])

    print(f"\nG numerator:   {num_master}")
    print(f"G denominator: {den_master}")

    # Check poles
    poles_m = np.roots(den_master)
    print(f"G poles: {poles_m}")
    print(f"Max pole magnitude: {np.max(np.abs(poles_m)):.6f}")
    print(f"Stable: {np.all(np.abs(poles_m) < 1.0)}")

    # Check H transfer function
    H_master = model_master.H
    num_h_master = np.array(H_master.num[0][0])
    den_h_master = np.array(H_master.den[0][0])

    print(f"\nH numerator:   {num_h_master}")
    print(f"H denominator: {den_h_master}")

    # Check poles
    poles_h_m = np.roots(den_h_master)
    print(f"H poles: {poles_h_m}")
    print(f"Max pole magnitude: {np.max(np.abs(poles_h_m)):.6f}")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

# Compare coefficients directly
print("\n" + "=" * 80)
print("COEFFICIENT COMPARISON:")
print("=" * 80)

if hasattr(model_master, 'THETA'):
    print(f"\nMaster THETA: {model_master.THETA}")
    print(f"  Shape: {model_master.THETA.shape}")

    # Try to parse THETA
    # THETA structure: [a_coeffs, b_coeffs, d_coeffs]
    theta_master = model_master.THETA.flatten()
    a_master = theta_master[:na]
    b_master = theta_master[na:na+nb]
    d_master = theta_master[na+nb:na+nb+nd]

    print(f"\nMaster coefficients:")
    print(f"  A: {a_master}")
    print(f"  B: {b_master}")
    print(f"  D: {d_master}")

# Harold coefficients (extracted from transfer functions)
if hasattr(model_harold, 'G_tf') and model_harold.G_tf is not None:
    # Extract from harold's transfer functions
    num_h = model_harold.G_tf.num.flatten()
    den_h = model_harold.G_tf.den.flatten()

    # A coefficients are in denominator (excluding leading 1.0)
    a_harold = -den_h[1:na+1]  # Negate because harold uses 1 + a1*z^-1 + ...

    # B coefficients are in numerator (excluding trailing zeros from delay)
    b_harold = num_h[:nb]

    # D coefficients from H_tf denominator
    if hasattr(model_harold, 'H_tf') and model_harold.H_tf is not None:
        den_h_tf = model_harold.H_tf.den.flatten()
        # H = 1 / (A*D), so den_h_tf = A*D polynomial
        # This is tricky to extract D from A*D...
        # For now, just show the full H denominator
        print(f"\nHarold coefficients (from transfer functions):")
        print(f"  A (from G denominator): {a_harold}")
        print(f"  B (from G numerator): {b_harold}")
        print(f"  H denominator (A*D): {den_h_tf}")

print("=" * 80)
