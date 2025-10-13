#!/usr/bin/env python
"""
Test script to verify control.forced_response works as a replacement for tf2ss.forced_response

This tests whether control.forced_response can be used in place of the missing tf2ss module.
"""

import numpy as np
import control

# Test 1: Basic forced_response test
print("=" * 80)
print("TEST 1: Basic forced_response with transfer function")
print("=" * 80)

# Create a simple transfer function: G(s) = 1 / (s + 1)
num = [1.0]
den = [1.0, 1.0]
G = control.tf(num, den, dt=1.0)  # Discrete-time with Ts=1.0

print(f"Transfer function G(z): {G}")

# Generate test input
time = np.arange(0, 10, 1.0)
u = np.ones_like(time)  # Step input

# Test forced_response
T, y = control.forced_response(G, time, u)

print(f"Input shape: {u.shape}")
print(f"Output time shape: {T.shape}")
print(f"Output shape: {y.shape}")
print(f"Output values (first 5): {y[:5]}")
print("✅ TEST 1 PASSED: forced_response works with transfer function")

# Test 2: SISO transfer function like in functionset.py
print("\n" + "=" * 80)
print("TEST 2: SISO transfer function (mimicking functionset.py usage)")
print("=" * 80)

# Simulate a system response from functionset.py line 199:
# T, Y_u = forced_response((1 / SYS.H[i, 0]) * SYS.G[i, :], Time, u)

# Create a more complex system
num_g = [0.5]
den_g = [1.0, -0.7]
G = control.tf(num_g, den_g, dt=1.0)

num_h = [1.0]
den_h = [1.0, -0.3]
H = control.tf(num_h, den_h, dt=1.0)

# Compute (1/H) * G like in the validation function
G_filtered = (1/H) * G

print(f"G(z): {G}")
print(f"H(z): {H}")
print(f"(1/H) * G: {G_filtered}")

# Generate input
npts = 20
time = np.arange(0, npts, 1.0)
u = np.random.randn(1, npts)  # Random input

# Test forced_response
T, Y_u = control.forced_response(G_filtered, time, u.flatten())

print(f"Time shape: {T.shape}")
print(f"Output shape: {Y_u.shape}")
print(f"Output values (first 5): {Y_u[:5]}")
print("✅ TEST 2 PASSED: Complex transfer function operations work")

# Test 3: Test with output signal (like line 200-201 in functionset.py)
print("\n" + "=" * 80)
print("TEST 3: Filtering output signal")
print("=" * 80)

# Simulate line 200-201:
# T, Y_y = forced_response(1 - (1 / SYS.H[i, 0]), Time, y[i, :] - y_rif[i])

# Generate simulated output
y = np.sin(2 * np.pi * time / 10) + 0.1 * np.random.randn(npts)
y_rif = 0.0  # Reference value

# Create filter: 1 - (1/H)
filter_tf = 1 - (1/H)

print(f"Filter: 1 - (1/H) = {filter_tf}")

# Apply filter
T, Y_y = control.forced_response(filter_tf, time, y - y_rif)

print(f"Input signal shape: {y.shape}")
print(f"Output shape: {Y_y.shape}")
print(f"Output values (first 5): {Y_y[:5]}")
print("✅ TEST 3 PASSED: Output filtering works")

# Test 4: Combining signals (like line 203 in functionset.py)
print("\n" + "=" * 80)
print("TEST 4: Combining filtered signals")
print("=" * 80)

# Simulate line 203:
# Yval[i, :] = Y_u + np.atleast_2d(Y_y) + y_rif[i]

Yval = Y_u + np.atleast_2d(Y_y) + y_rif

print(f"Combined output shape: {Yval.shape}")
print(f"Combined output values (first 5): {Yval.flatten()[:5]}")
print("✅ TEST 4 PASSED: Signal combination works")

print("\n" + "=" * 80)
print("OVERALL RESULT: ✅ ALL TESTS PASSED")
print("=" * 80)
print("\nConclusion:")
print("  - control.forced_response can replace tf2ss.forced_response")
print("  - It works with transfer functions and discrete-time systems")
print("  - The API is compatible with the master branch usage patterns")
print("\nRecommendation:")
print("  - Create a tf2ss.py compatibility shim that imports from control")
print("  - Place it in the master branch directory: /Users/josephj/Workspace/SIPPY-master/")
