"""
GEN (Generalized Model) Example

Demonstrates the use of the GEN identification algorithm, which is the most general
input-output model structure with 5 polynomial orders:

A(q) * y(t) = [B(q)/F(q)] * u(t-nk) + [C(q)/D(q)] * e(t)

GEN generalizes all other input-output methods:
- ARX = GEN(na, nb, 0, 0, 0, nk)
- ARMAX = GEN(na, nb, nc, 0, 0, nk)
- ARARX = GEN(na, nb, 0, 0, nf, nk)
- ARARMAX = GEN(na, nb, nc, nd, 0, nk)
- OE = GEN(0, nb, 0, 0, nf, nk)
- BJ = GEN(0, nb, nc, nd, nf, nk)

@author: Claude (claude.ai/code)
"""

import matplotlib.pyplot as plt
import numpy as np

from sippy.identification import SystemIdentification
from sippy.utils.signal_utils import GBN_seq, white_noise_var

# Define sampling time and time vector
sampling_time = 1.0  # [s]
end_time = 400  # [s]
npts = int(end_time / sampling_time) + 1
Time = np.linspace(0, end_time, npts)

# Generate input signal (Generalized Binary Noise)
switch_probability = 0.08  # [0..1]
Usim, _, _ = GBN_seq(npts, switch_probability, Range=[-1, 1])

# Generate white noise for stochastic part
white_noise_variance = [0.01]
e_t = white_noise_var(Usim.size, white_noise_variance)[0]

print("="*80)
print("GEN (Generalized Model) Identification Example")
print("="*80)
print("\nGEN is the most general input-output model structure:")
print("A(q) * y(t) = [B(q)/F(q)] * u(t-nk) + [C(q)/D(q)] * e(t)")
print("\nwhere:")
print("  A(q) - Output autoregressive polynomial (order na)")
print("  B(q) - Input numerator polynomial (order nb)")
print("  F(q) - Input denominator polynomial (order nf)")
print("  C(q) - Noise numerator polynomial (order nc)")
print("  D(q) - Noise denominator polynomial (order nd)")
print("  nk   - Input delay")
print("\n" + "="*80)

# ============================================================================
# Example 1: GEN as ARX (na, nb, 0, 0, 0, nk)
# ============================================================================
print("\nExample 1: GEN configured as ARX (nc=nd=nf=0)")
print("-" * 80)

# True system parameters for ARX
true_a = [0.5]  # AR coefficient
true_b = [0.6, 0.3]  # Input coefficients
nk_arx = 1

# Generate ARX output
y_arx = np.zeros(npts)
for k in range(2, npts):
    y_arx[k] = (
        -true_a[0] * y_arx[k-1]
        + true_b[0] * Usim[k-nk_arx]
        + true_b[1] * Usim[k-nk_arx-1]
        + e_t[k]
    )

# Identify using GEN with nc=nd=nf=0 (ARX structure)
y_arx_2d = y_arx.reshape(1, -1)
u_arx_2d = Usim.reshape(1, -1)

sysid = SystemIdentification()
model_arx = sysid.identify(
    y=y_arx_2d,
    u=u_arx_2d,
    method="GEN",
    na=1,
    nb=2,
    nc=0,
    nd=0,
    nf=0,
    nk=nk_arx,
    tsample=sampling_time
)

print(f"True A:      {true_a}")
print(f"Estimated A: {model_arx.A_coeffs[0, :] if model_arx.A_coeffs.shape[1] > 0 else []}")
print(f"True B:      {true_b}")
print(f"Estimated B: {model_arx.B_coeffs[0, :]}")
print(f"Noise variance (Vn): {model_arx.Vn:.6f}")

# ============================================================================
# Example 2: GEN as ARMAX (na, nb, nc, 0, 0, nk)
# ============================================================================
print("\nExample 2: GEN configured as ARMAX (nd=nf=0)")
print("-" * 80)

# True system parameters for ARMAX
true_c = [0.4]  # Noise AR coefficient

# Generate ARMAX output
y_armax = np.zeros(npts)
epsilon = np.zeros(npts)
for k in range(2, npts):
    y_armax[k] = (
        -true_a[0] * y_armax[k-1]
        + true_b[0] * Usim[k-nk_arx]
        + true_b[1] * Usim[k-nk_arx-1]
        + e_t[k]
        + true_c[0] * e_t[k-1]
    )

# Identify using GEN with nd=nf=0 (ARMAX structure)
y_armax_2d = y_armax.reshape(1, -1)

model_armax = sysid.identify(
    y=y_armax_2d,
    u=u_arx_2d,
    method="GEN",
    na=1,
    nb=2,
    nc=1,
    nd=0,
    nf=0,
    nk=nk_arx,
    tsample=sampling_time
)

print(f"True A:      {true_a}")
print(f"Estimated A: {model_armax.A_coeffs[0, :] if model_armax.A_coeffs.shape[1] > 0 else []}")
print(f"True B:      {true_b}")
print(f"Estimated B: {model_armax.B_coeffs[0, :]}")
print(f"True C:      {true_c}")
print(f"Estimated C: {model_armax.C_coeffs[0, :] if model_armax.C_coeffs.shape[1] > 0 else []}")
print(f"Noise variance (Vn): {model_armax.Vn:.6f}")

# ============================================================================
# Example 3: GEN as OE (0, nb, 0, 0, nf, nk)
# ============================================================================
print("\nExample 3: GEN configured as OE (na=nc=nd=0)")
print("-" * 80)

# True system parameters for OE
true_f = [0.5]  # Input denominator coefficient

# Generate OE output
y_oe = np.zeros(npts)
w = np.zeros(npts)  # Intermediate variable for input path
for k in range(2, npts):
    w[k] = (
        -true_f[0] * w[k-1]
        + true_b[0] * Usim[k-nk_arx]
        + true_b[1] * Usim[k-nk_arx-1]
    )
    y_oe[k] = w[k] + e_t[k]

# Identify using GEN with na=nc=nd=0 (OE structure)
y_oe_2d = y_oe.reshape(1, -1)

model_oe = sysid.identify(
    y=y_oe_2d,
    u=u_arx_2d,
    method="GEN",
    na=0,
    nb=2,
    nc=0,
    nd=0,
    nf=1,
    nk=nk_arx,
    tsample=sampling_time
)

print(f"True B:      {true_b}")
print(f"Estimated B: {model_oe.B_coeffs[0, :]}")
print(f"True F:      {true_f}")
print(f"Estimated F: {model_oe.F_coeffs[0, :] if model_oe.F_coeffs.shape[1] > 0 else []}")
print(f"Noise variance (Vn): {model_oe.Vn:.6f}")

# ============================================================================
# Example 4: Full GEN structure (na, nb, nc, nd, nf, nk)
# ============================================================================
print("\nExample 4: Full GEN structure with all polynomials")
print("-" * 80)

# True system parameters for full GEN
true_d = [0.3]  # Noise denominator coefficient

# Generate full GEN output
y_gen = np.zeros(npts)
w_gen = np.zeros(npts)
v_gen = np.zeros(npts)
for k in range(3, npts):
    # Input path: B/F * u
    w_gen[k] = (
        -true_f[0] * w_gen[k-1]
        + true_b[0] * Usim[k-nk_arx]
        + true_b[1] * Usim[k-nk_arx-1]
    )

    # Noise path: C/D * e
    v_gen[k] = (
        -true_d[0] * v_gen[k-1]
        + e_t[k]
        + true_c[0] * e_t[k-1]
    )

    # Output: A * y = w + v
    y_gen[k] = (
        -true_a[0] * y_gen[k-1]
        + w_gen[k]
        + v_gen[k]
    )

# Identify using full GEN
y_gen_2d = y_gen.reshape(1, -1)

model_gen = sysid.identify(
    y=y_gen_2d,
    u=u_arx_2d,
    method="GEN",
    na=1,
    nb=2,
    nc=1,
    nd=1,
    nf=1,
    nk=nk_arx,
    tsample=sampling_time,
    max_iterations=300
)

print(f"True A:      {true_a}")
print(f"Estimated A: {model_gen.A_coeffs[0, :] if model_gen.A_coeffs.shape[1] > 0 else []}")
print(f"True B:      {true_b}")
print(f"Estimated B: {model_gen.B_coeffs[0, :]}")
print(f"True C:      {true_c}")
print(f"Estimated C: {model_gen.C_coeffs[0, :] if model_gen.C_coeffs.shape[1] > 0 else []}")
print(f"True D:      {true_d}")
print(f"Estimated D: {model_gen.D_coeffs[0, :] if model_gen.D_coeffs.shape[1] > 0 else []}")
print(f"True F:      {true_f}")
print(f"Estimated F: {model_gen.F_coeffs[0, :] if model_gen.F_coeffs.shape[1] > 0 else []}")
print(f"Noise variance (Vn): {model_gen.Vn:.6f}")

# ============================================================================
# Visualization
# ============================================================================
print("\nGenerating plots...")

fig, axes = plt.subplots(4, 1, figsize=(12, 10))
fig.suptitle("GEN Identification Examples", fontsize=16)

# Plot 1: Input signal
axes[0].plot(Time, Usim, 'b-', linewidth=0.8)
axes[0].set_ylabel('Input u(t)')
axes[0].set_title('Input Signal (GBN)')
axes[0].grid(True)

# Plot 2: GEN as ARX
axes[1].plot(Time, y_arx, 'b-', label='True ARX', linewidth=1.0, alpha=0.7)
axes[1].plot(Time, model_arx.Yid[0, :], 'r--', label='GEN (ARX)', linewidth=1.0)
axes[1].set_ylabel('Output y(t)')
axes[1].set_title('Example 1: GEN as ARX')
axes[1].legend()
axes[1].grid(True)

# Plot 3: GEN as ARMAX
axes[2].plot(Time, y_armax, 'b-', label='True ARMAX', linewidth=1.0, alpha=0.7)
axes[2].plot(Time, model_armax.Yid[0, :], 'g--', label='GEN (ARMAX)', linewidth=1.0)
axes[2].set_ylabel('Output y(t)')
axes[2].set_title('Example 2: GEN as ARMAX')
axes[2].legend()
axes[2].grid(True)

# Plot 4: Full GEN
axes[3].plot(Time, y_gen, 'b-', label='True GEN', linewidth=1.0, alpha=0.7)
axes[3].plot(Time, model_gen.Yid[0, :], 'm--', label='GEN (Full)', linewidth=1.0)
axes[3].set_ylabel('Output y(t)')
axes[3].set_xlabel('Time [s]')
axes[3].set_title('Example 4: Full GEN Structure')
axes[3].legend()
axes[3].grid(True)

plt.tight_layout()
plt.show()

print("\n" + "="*80)
print("GEN Example Complete")
print("="*80)
print("\nKey Observations:")
print("1. GEN can reproduce ARX behavior when nc=nd=nf=0")
print("2. GEN can reproduce ARMAX behavior when nd=nf=0")
print("3. GEN can reproduce OE behavior when na=nc=nd=0")
print("4. Full GEN structure handles complex dynamics with all polynomials")
print("\nGEN is the most flexible but also the most challenging identification algorithm.")
print("Consider using specialized methods (ARX, ARMAX, OE, BJ) when possible for better")
print("numerical stability and parameter identifiability.")
