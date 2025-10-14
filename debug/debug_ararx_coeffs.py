"""
Debug ARARX Coefficients: Check what the NLP is estimating
"""

import numpy as np
import sys
sys.path.insert(0, '/Users/josephj/Workspace/SIPPY-master')

from src.sippy.identification.algorithms.ararx import ARARXAlgorithm
from sippy_unipi import system_identification as master_sysid

# Generate simple test data
np.random.seed(42)
N = 200
u = np.random.randn(1, N) * 0.5
y = np.zeros((1, N))
for k in range(1, N):
    y[0, k] = 0.5 * y[0, k-1] + 0.3 * u[0, k-1] + np.random.randn() * 0.01

na, nb, nd, theta = 1, 1, 1, 1

print("=" * 70)
print("ARARX Coefficient Debug")
print("=" * 70)
print(f"Model orders: na={na}, nb={nb}, nd={nd}, theta={theta}")

# Harold branch - direct access to coefficients
print("\n" + "-" * 70)
print("Harold Branch:")
print("-" * 70)

algo = ARARXAlgorithm()
model = algo.identify(y=y, u=u, na=na, nb=nb, nd=nd, theta=theta, max_iterations=200)

# Try to extract polynomial coefficients
print("\nAttempting to extract polynomial coefficients...")

# Check if we stored them anywhere
if hasattr(model, 'G_tf') and model.G_tf is not None:
    print(f"\nG_tf numerator:   {model.G_tf.num}")
    print(f"G_tf denominator: {model.G_tf.den}")
    
# Now let's manually call _identify_nlp and capture coefficients
print("\n" + "-" * 70)
print("Re-running with coefficient extraction:")
print("-" * 70)

# Patch the method to print coefficients
original_identify_nlp = algo._identify_nlp

def patched_identify_nlp(y, u, na, nb, nd, theta, sample_time, **kwargs):
    """Patched version that prints coefficients."""
    import numpy as np
    from casadi import DM, SX, mtimes, nlpsol, norm_inf, vertcat
    
    # Get data dimensions
    ny, N = y.shape
    nu, _ = u.shape
    
    # Data rescaling
    y_std, y_scaled = algo._rescale(y.flatten())
    u_std, u_scaled = algo._rescale(u.flatten())
    y_flat = y_scaled
    u_flat = u_scaled
    
    print(f"\nData scaling:")
    print(f"  y_std = {y_std:.6f}")
    print(f"  u_std = {u_std:.6f}")
    
    # Extract NLP parameters
    max_iterations = kwargs.get("max_iterations", 200)
    stability_cons = kwargs.get("stability_constraint", False)
    stab_marg = kwargs.get("stability_margin", 1.0)
    n_tr = max(na, nb + theta, nd)
    
    # Build and solve NLP
    solver, w_lb, w_ub, g_lb, g_ub, w_0 = algo._build_ararx_nlp(
        y_flat, u_flat, na, nb, nd, theta, N, n_tr,
        max_iterations, stab_marg, stability_cons
    )
    
    # Solve
    sol = solver(lbx=w_lb, ubx=w_ub, x0=w_0, lbg=g_lb, ubg=g_ub)
    
    # Extract solution
    x_opt = sol["x"]
    n_coeff = na + nb + nd
    
    # Extract coefficients (SCALED)
    THETA = np.array(x_opt[:n_coeff]).flatten()
    A_coeffs_scaled = THETA[:na] if na > 0 else np.array([])
    B_coeffs_scaled = THETA[na:na+nb]
    D_coeffs_scaled = THETA[na+nb:na+nb+nd]
    
    print(f"\nScaled coefficients (from NLP):")
    print(f"  A_coeffs (scaled): {A_coeffs_scaled}")
    print(f"  B_coeffs (scaled): {B_coeffs_scaled}")
    print(f"  D_coeffs (scaled): {D_coeffs_scaled}")
    
    # Rescale B coefficients
    B_coeffs = B_coeffs_scaled * (y_std / u_std)
    
    print(f"\nRescaled coefficients:")
    print(f"  A_coeffs: {A_coeffs_scaled}  (no rescaling needed)")
    print(f"  B_coeffs: {B_coeffs}")
    print(f"  D_coeffs: {D_coeffs_scaled}  (no rescaling needed)")
    
    # Continue with original method
    return original_identify_nlp(y, u, na, nb, nd, theta, sample_time, **kwargs)

algo._identify_nlp = patched_identify_nlp
model2 = algo.identify(y=y, u=u, na=na, nb=nb, nd=nd, theta=theta, max_iterations=200)

# Master branch
print("\n" + "=" * 70)
print("Master Branch:")
print("=" * 70)

model_master = master_sysid(
    y, u, "ARARX",
    ARARX_orders=[[na], [[nb]], [nd], [[theta]]],
    tsample=1.0,
    max_iterations=200
)

# Try to extract master coefficients
print(f"\nMaster Vn: {model_master.Vn:.6f}")

# The master stores THETA attribute with coefficients
if hasattr(model_master, 'THETA'):
    print(f"\nMaster THETA: {model_master.THETA}")

print("=" * 70)
