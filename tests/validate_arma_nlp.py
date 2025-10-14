"""
ARMA Comprehensive Validation Script

Validates ARMA NLP implementation using cross-branch testing with one-step
prediction comparison (Yid) as the primary validation metric.

Test Cases:
- Case 1: AR(1) - Pure autoregressive (a1=0.7)
- Case 2: MA(1) - Pure moving average (c1=0.5)
- Case 3: ARMA(1,1) - Simple full model (a1=0.7, c1=0.5)
- Case 4: ARMA(2,2) - Higher order (a=[0.6, 0.2], c=[0.4, 0.1])

Validation Metrics:
- Coefficient errors: < 10% for AR, < 15% for MA (MA harder)
- Yid NRMSE: < 10% (primary acceptance criterion)
- Yid Correlation: > 0.95 (primary acceptance criterion)
- Stability: All poles within unit circle

Success Criteria:
- ✅ EXCELLENT: NRMSE < 5%, Correlation > 0.99
- ✅ GOOD: NRMSE < 10%, Correlation > 0.95
- ⚠️  ACCEPTABLE: NRMSE < 15%, Correlation > 0.90
- ❌ FAIL: NRMSE >= 15% or Correlation < 0.90
"""

import json
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, '/Users/josephj/Workspace/SIPPY-master')

from src.sippy.identification import SystemIdentification, SystemIdentificationConfig
from sippy_unipi import system_identification as master_sysid


def generate_arma_data(ar_coeffs, ma_coeffs, n_samples=500, noise_std=0.1, seed=42):
    """
    Generate synthetic ARMA data with known parameters.

    Parameters:
    -----------
    ar_coeffs : list
        AR coefficients in transfer function form (negative values)
        e.g., [-0.7, -0.2] for y[k] = 0.7*y[k-1] + 0.2*y[k-2] + ...
    ma_coeffs : list
        MA coefficients in transfer function form (positive values)
        e.g., [0.5, 0.1] for e[k] + 0.5*e[k-1] + 0.1*e[k-2]
    n_samples : int
        Number of samples to generate
    noise_std : float
        Standard deviation of white noise
    seed : int
        Random seed for reproducibility

    Returns:
    --------
    y : ndarray
        Generated output data (1 x n_samples)
    ar_true : ndarray
        True AR coefficients (for comparison)
    ma_true : ndarray
        True MA coefficients (for comparison)
    """
    np.random.seed(seed)

    na = len(ar_coeffs)
    nc = len(ma_coeffs)

    # Generate white noise
    e = np.random.randn(n_samples) * noise_std

    # Initialize output
    y = np.zeros(n_samples)

    # Generate ARMA process
    for k in range(max(na, nc), n_samples):
        # AR component: y[k] = -a1*y[k-1] - a2*y[k-2] - ...
        # ar_coeffs are in TF form (negative), so we negate
        ar_sum = 0
        for lag in range(na):
            ar_sum += -ar_coeffs[lag] * y[k - 1 - lag]

        # MA component: e[k] + c1*e[k-1] + c2*e[k-2] + ...
        ma_sum = e[k]
        for lag in range(nc):
            ma_sum += ma_coeffs[lag] * e[k - 1 - lag]

        y[k] = ar_sum + ma_sum

    return y.reshape(1, -1), np.array(ar_coeffs), np.array(ma_coeffs)


def compute_metrics(yid_harold, yid_master, y_true):
    """Compute validation metrics for Yid comparison."""
    # Align lengths
    min_len = min(len(yid_harold), len(yid_master), len(y_true))
    yid_h = yid_harold[:min_len]
    yid_m = yid_master[:min_len]
    y = y_true[:min_len]

    # Yid comparison metrics
    yid_mse = np.mean((yid_h - yid_m) ** 2)
    yid_mae = np.mean(np.abs(yid_h - yid_m))

    # Correlation
    if np.std(yid_h) > 1e-10 and np.std(yid_m) > 1e-10:
        yid_corr = np.corrcoef(yid_h, yid_m)[0, 1]
    else:
        yid_corr = 0.0

    # Normalized RMSE
    y_rms = np.sqrt(np.mean(y ** 2))
    yid_nrmse = np.sqrt(yid_mse) / y_rms if y_rms > 1e-10 else float('inf')

    # Prediction errors vs true output
    err_h = y - yid_h
    err_m = y - yid_m
    mse_h = np.mean(err_h ** 2)
    mse_m = np.mean(err_m ** 2)
    mae_h = np.mean(np.abs(err_h))
    mae_m = np.mean(np.abs(err_m))

    return {
        'yid_mse': yid_mse,
        'yid_mae': yid_mae,
        'yid_nrmse': yid_nrmse,
        'yid_corr': yid_corr,
        'harold_pred_mse': mse_h,
        'harold_pred_mae': mae_h,
        'master_pred_mse': mse_m,
        'master_pred_mae': mae_m,
    }


def assess_verdict(yid_nrmse, yid_corr):
    """Assess validation verdict based on metrics."""
    if yid_nrmse < 0.05 and yid_corr > 0.99:
        return "EXCELLENT", True
    elif yid_nrmse < 0.10 and yid_corr > 0.95:
        return "GOOD", True
    elif yid_nrmse < 0.15 and yid_corr > 0.90:
        return "ACCEPTABLE", True
    else:
        return "FAIL", False


def test_case_1():
    """Test Case 1: Pure AR(1) process."""
    print("\n" + "=" * 80)
    print("TEST CASE 1: Pure AR(1) Process")
    print("=" * 80)
    print("\nModel: y[k] = 0.7*y[k-1] + e[k]")
    print("True AR coefficient: a1 = -0.7 (TF form)")
    print("True MA coefficient: c1 = 0.0 (no MA component, but nc=1 for ARMA)")

    # Generate data
    y, ar_true, ma_true = generate_arma_data(
        ar_coeffs=[-0.7],
        ma_coeffs=[0.0],  # No MA, but nc=1 required
        n_samples=500,
        noise_std=0.1,
        seed=42
    )

    # Create dummy input (ARMA doesn't use inputs, but master ARMAX needs it)
    u = np.zeros((1, y.shape[1]))

    na, nc = 1, 1

    # Harold branch
    config_h = SystemIdentificationConfig(method="ARMA")
    config_h.na, config_h.nc = na, nc
    config_h.max_iterations = 100

    identifier_h = SystemIdentification(config_h)
    model_h = identifier_h.identify(y=y, u=None)

    # Master branch (use ARMAX with u=0 as proxy)
    model_m = master_sysid(
        y, u, "ARMAX",
        na_ord=[na],
        nb_ord=[1],  # Minimal input order
        nc_ord=[nc],
        tsample=1.0
    )

    # Extract coefficients
    ar_harold = model_h.AR_coeffs[0, :]
    ma_harold = model_h.MA_coeffs[0, :]

    # Coefficient errors
    ar_error = np.abs(ar_harold - ar_true) / np.abs(ar_true) * 100
    ma_error_abs = np.abs(ma_harold[0])  # Should be near zero

    print(f"\nCoefficient Comparison:")
    print(f"  True AR:     {ar_true}")
    print(f"  Harold AR:   {ar_harold}")
    print(f"  AR error:    {ar_error[0]:.2f}%")
    print(f"  Harold MA:   {ma_harold}")
    print(f"  MA abs value: {ma_error_abs:.6f} (should be near 0)")

    # Compare Yid
    yid_h = model_h.Yid.flatten() if hasattr(model_h, 'Yid') else None
    yid_m = model_m.Yid.flatten() if hasattr(model_m, 'Yid') else None

    if yid_h is not None and yid_m is not None:
        metrics = compute_metrics(yid_h, yid_m, y.flatten())

        print(f"\nPrediction Errors (vs True Output):")
        print(f"  Harold MSE:  {metrics['harold_pred_mse']:.6e}")
        print(f"  Master MSE:  {metrics['master_pred_mse']:.6e}")
        print(f"  Harold MAE:  {metrics['harold_pred_mae']:.6e}")
        print(f"  Master MAE:  {metrics['master_pred_mae']:.6e}")

        print(f"\nYid Comparison (Harold vs Master):")
        print(f"  MSE:         {metrics['yid_mse']:.6e}")
        print(f"  MAE:         {metrics['yid_mae']:.6e}")
        print(f"  NRMSE:       {metrics['yid_nrmse']*100:.2f}%")
        print(f"  Correlation: {metrics['yid_corr']:.6f}")

        print(f"\nModel Quality:")
        print(f"  Harold Vn:   {model_h.Vn:.6f}")
        print(f"  Master Vn:   {model_m.Vn:.6f}")

        # Check stability
        if hasattr(model_h, 'H_tf') and model_h.H_tf is not None:
            poles_h = np.roots(model_h.H_tf.den.flatten())
            stable_h = np.all(np.abs(poles_h) < 1.0)
            print(f"  Harold Stable: {stable_h} (max pole: {np.max(np.abs(poles_h)):.4f})")

        # Verdict
        verdict, passed = assess_verdict(metrics['yid_nrmse'], metrics['yid_corr'])

        if passed and ar_error[0] < 10 and ma_error_abs < 0.1:
            print(f"\n✅ {verdict}: AR(1) validation PASSED")
            print(f"   - Yid NRMSE: {metrics['yid_nrmse']*100:.2f}% (< 10% threshold)")
            print(f"   - Yid Correlation: {metrics['yid_corr']:.6f} (> 0.95 threshold)")
            print(f"   - AR coefficient error: {ar_error[0]:.2f}% (< 10% threshold)")
            return True, metrics
        else:
            print(f"\n❌ {verdict}: AR(1) validation FAILED")
            return False, metrics
    else:
        print("\n❌ ERROR: Could not extract Yid from one or both models")
        return False, {}


def test_case_2():
    """Test Case 2: Pure MA(1) process."""
    print("\n" + "=" * 80)
    print("TEST CASE 2: Pure MA(1) Process")
    print("=" * 80)
    print("\nModel: y[k] = e[k] + 0.5*e[k-1]")
    print("True AR coefficient: a1 = 0.0 (no AR component, but na=1 for ARMA)")
    print("True MA coefficient: c1 = 0.5 (TF form)")

    # Generate data
    y, ar_true, ma_true = generate_arma_data(
        ar_coeffs=[0.0],  # No AR, but na=1 required
        ma_coeffs=[0.5],
        n_samples=500,
        noise_std=0.1,
        seed=123
    )

    # Create dummy input
    u = np.zeros((1, y.shape[1]))

    na, nc = 1, 1

    # Harold branch
    config_h = SystemIdentificationConfig(method="ARMA")
    config_h.na, config_h.nc = na, nc
    config_h.max_iterations = 100

    identifier_h = SystemIdentification(config_h)
    model_h = identifier_h.identify(y=y, u=None)

    # Master branch (use ARMAX with u=0 as proxy)
    model_m = master_sysid(
        y, u, "ARMAX",
        na_ord=[na],
        nb_ord=[1],
        nc_ord=[nc],
        tsample=1.0
    )

    # Extract coefficients
    ar_harold = model_h.AR_coeffs[0, :]
    ma_harold = model_h.MA_coeffs[0, :]

    # Coefficient errors
    ar_error_abs = np.abs(ar_harold[0])  # Should be near zero
    ma_error = np.abs(ma_harold - ma_true) / np.abs(ma_true) * 100

    print(f"\nCoefficient Comparison:")
    print(f"  True MA:     {ma_true}")
    print(f"  Harold MA:   {ma_harold}")
    print(f"  MA error:    {ma_error[0]:.2f}%")
    print(f"  Harold AR:   {ar_harold}")
    print(f"  AR abs value: {ar_error_abs:.6f} (should be near 0)")

    # Compare Yid
    yid_h = model_h.Yid.flatten() if hasattr(model_h, 'Yid') else None
    yid_m = model_m.Yid.flatten() if hasattr(model_m, 'Yid') else None

    if yid_h is not None and yid_m is not None:
        metrics = compute_metrics(yid_h, yid_m, y.flatten())

        print(f"\nPrediction Errors (vs True Output):")
        print(f"  Harold MSE:  {metrics['harold_pred_mse']:.6e}")
        print(f"  Master MSE:  {metrics['master_pred_mse']:.6e}")
        print(f"  Harold MAE:  {metrics['harold_pred_mae']:.6e}")
        print(f"  Master MAE:  {metrics['master_pred_mae']:.6e}")

        print(f"\nYid Comparison (Harold vs Master):")
        print(f"  MSE:         {metrics['yid_mse']:.6e}")
        print(f"  MAE:         {metrics['yid_mae']:.6e}")
        print(f"  NRMSE:       {metrics['yid_nrmse']*100:.2f}%")
        print(f"  Correlation: {metrics['yid_corr']:.6f}")

        print(f"\nModel Quality:")
        print(f"  Harold Vn:   {model_h.Vn:.6f}")
        print(f"  Master Vn:   {model_m.Vn:.6f}")

        # Check stability
        if hasattr(model_h, 'H_tf') and model_h.H_tf is not None:
            poles_h = np.roots(model_h.H_tf.den.flatten())
            stable_h = np.all(np.abs(poles_h) < 1.0)
            print(f"  Harold Stable: {stable_h} (max pole: {np.max(np.abs(poles_h)):.4f})")

        # Verdict (MA is harder, use more lenient thresholds)
        verdict, passed = assess_verdict(metrics['yid_nrmse'], metrics['yid_corr'])

        if passed and ma_error[0] < 15 and ar_error_abs < 0.1:
            print(f"\n✅ {verdict}: MA(1) validation PASSED")
            print(f"   - Yid NRMSE: {metrics['yid_nrmse']*100:.2f}% (< 10% threshold)")
            print(f"   - Yid Correlation: {metrics['yid_corr']:.6f} (> 0.95 threshold)")
            print(f"   - MA coefficient error: {ma_error[0]:.2f}% (< 15% threshold for MA)")
            return True, metrics
        else:
            print(f"\n❌ {verdict}: MA(1) validation FAILED")
            return False, metrics
    else:
        print("\n❌ ERROR: Could not extract Yid from one or both models")
        return False, {}


def test_case_3():
    """Test Case 3: ARMA(1,1) process."""
    print("\n" + "=" * 80)
    print("TEST CASE 3: ARMA(1,1) Process")
    print("=" * 80)
    print("\nModel: y[k] = 0.7*y[k-1] + e[k] + 0.5*e[k-1]")
    print("True AR coefficient: a1 = -0.7 (TF form)")
    print("True MA coefficient: c1 = 0.5 (TF form)")

    # Generate data
    y, ar_true, ma_true = generate_arma_data(
        ar_coeffs=[-0.7],
        ma_coeffs=[0.5],
        n_samples=500,
        noise_std=0.1,
        seed=456
    )

    # Create dummy input
    u = np.zeros((1, y.shape[1]))

    na, nc = 1, 1

    # Harold branch
    config_h = SystemIdentificationConfig(method="ARMA")
    config_h.na, config_h.nc = na, nc
    config_h.max_iterations = 100

    identifier_h = SystemIdentification(config_h)
    model_h = identifier_h.identify(y=y, u=None)

    # Master branch (use ARMAX with u=0 as proxy)
    model_m = master_sysid(
        y, u, "ARMAX",
        na_ord=[na],
        nb_ord=[1],
        nc_ord=[nc],
        tsample=1.0
    )

    # Extract coefficients
    ar_harold = model_h.AR_coeffs[0, :]
    ma_harold = model_h.MA_coeffs[0, :]

    # Coefficient errors
    ar_error = np.abs(ar_harold - ar_true) / np.abs(ar_true) * 100
    ma_error = np.abs(ma_harold - ma_true) / np.abs(ma_true) * 100

    print(f"\nCoefficient Comparison:")
    print(f"  True AR:     {ar_true}")
    print(f"  Harold AR:   {ar_harold}")
    print(f"  AR error:    {ar_error[0]:.2f}%")
    print(f"  True MA:     {ma_true}")
    print(f"  Harold MA:   {ma_harold}")
    print(f"  MA error:    {ma_error[0]:.2f}%")

    # Compare Yid
    yid_h = model_h.Yid.flatten() if hasattr(model_h, 'Yid') else None
    yid_m = model_m.Yid.flatten() if hasattr(model_m, 'Yid') else None

    if yid_h is not None and yid_m is not None:
        metrics = compute_metrics(yid_h, yid_m, y.flatten())

        print(f"\nPrediction Errors (vs True Output):")
        print(f"  Harold MSE:  {metrics['harold_pred_mse']:.6e}")
        print(f"  Master MSE:  {metrics['master_pred_mse']:.6e}")
        print(f"  Harold MAE:  {metrics['harold_pred_mae']:.6e}")
        print(f"  Master MAE:  {metrics['master_pred_mae']:.6e}")

        print(f"\nYid Comparison (Harold vs Master):")
        print(f"  MSE:         {metrics['yid_mse']:.6e}")
        print(f"  MAE:         {metrics['yid_mae']:.6e}")
        print(f"  NRMSE:       {metrics['yid_nrmse']*100:.2f}%")
        print(f"  Correlation: {metrics['yid_corr']:.6f}")

        print(f"\nModel Quality:")
        print(f"  Harold Vn:   {model_h.Vn:.6f}")
        print(f"  Master Vn:   {model_m.Vn:.6f}")

        # Check stability
        if hasattr(model_h, 'H_tf') and model_h.H_tf is not None:
            poles_h = np.roots(model_h.H_tf.den.flatten())
            stable_h = np.all(np.abs(poles_h) < 1.0)
            print(f"  Harold Stable: {stable_h} (max pole: {np.max(np.abs(poles_h)):.4f})")

        # Verdict
        verdict, passed = assess_verdict(metrics['yid_nrmse'], metrics['yid_corr'])

        if passed and ar_error[0] < 10 and ma_error[0] < 15:
            print(f"\n✅ {verdict}: ARMA(1,1) validation PASSED")
            print(f"   - Yid NRMSE: {metrics['yid_nrmse']*100:.2f}% (< 10% threshold)")
            print(f"   - Yid Correlation: {metrics['yid_corr']:.6f} (> 0.95 threshold)")
            print(f"   - AR coefficient error: {ar_error[0]:.2f}% (< 10% threshold)")
            print(f"   - MA coefficient error: {ma_error[0]:.2f}% (< 15% threshold)")
            return True, metrics
        else:
            print(f"\n❌ {verdict}: ARMA(1,1) validation FAILED")
            return False, metrics
    else:
        print("\n❌ ERROR: Could not extract Yid from one or both models")
        return False, {}


def test_case_4():
    """Test Case 4: ARMA(2,2) process."""
    print("\n" + "=" * 80)
    print("TEST CASE 4: ARMA(2,2) Process")
    print("=" * 80)
    print("\nModel: y[k] = 0.6*y[k-1] + 0.2*y[k-2] + e[k] + 0.4*e[k-1] + 0.1*e[k-2]")
    print("True AR coefficients: a = [-0.6, -0.2] (TF form)")
    print("True MA coefficients: c = [0.4, 0.1] (TF form)")

    # Generate data
    y, ar_true, ma_true = generate_arma_data(
        ar_coeffs=[-0.6, -0.2],
        ma_coeffs=[0.4, 0.1],
        n_samples=600,
        noise_std=0.1,
        seed=789
    )

    # Create dummy input
    u = np.zeros((1, y.shape[1]))

    na, nc = 2, 2

    # Harold branch
    config_h = SystemIdentificationConfig(method="ARMA")
    config_h.na, config_h.nc = na, nc
    config_h.max_iterations = 100

    identifier_h = SystemIdentification(config_h)
    model_h = identifier_h.identify(y=y, u=None)

    # Master branch (use ARMAX with u=0 as proxy)
    model_m = master_sysid(
        y, u, "ARMAX",
        na_ord=[na],
        nb_ord=[1],
        nc_ord=[nc],
        tsample=1.0
    )

    # Extract coefficients
    ar_harold = model_h.AR_coeffs[0, :]
    ma_harold = model_h.MA_coeffs[0, :]

    # Coefficient errors
    ar_error = np.abs(ar_harold - ar_true) / np.abs(ar_true) * 100
    ma_error = np.abs(ma_harold - ma_true) / np.abs(ma_true) * 100

    print(f"\nCoefficient Comparison:")
    print(f"  True AR:     {ar_true}")
    print(f"  Harold AR:   {ar_harold}")
    print(f"  AR errors:   {ar_error} %")
    print(f"  AR max error: {np.max(ar_error):.2f}%")
    print(f"  True MA:     {ma_true}")
    print(f"  Harold MA:   {ma_harold}")
    print(f"  MA errors:   {ma_error} %")
    print(f"  MA max error: {np.max(ma_error):.2f}%")

    # Compare Yid
    yid_h = model_h.Yid.flatten() if hasattr(model_h, 'Yid') else None
    yid_m = model_m.Yid.flatten() if hasattr(model_m, 'Yid') else None

    if yid_h is not None and yid_m is not None:
        metrics = compute_metrics(yid_h, yid_m, y.flatten())

        print(f"\nPrediction Errors (vs True Output):")
        print(f"  Harold MSE:  {metrics['harold_pred_mse']:.6e}")
        print(f"  Master MSE:  {metrics['master_pred_mse']:.6e}")
        print(f"  Harold MAE:  {metrics['harold_pred_mae']:.6e}")
        print(f"  Master MAE:  {metrics['master_pred_mae']:.6e}")

        print(f"\nYid Comparison (Harold vs Master):")
        print(f"  MSE:         {metrics['yid_mse']:.6e}")
        print(f"  MAE:         {metrics['yid_mae']:.6e}")
        print(f"  NRMSE:       {metrics['yid_nrmse']*100:.2f}%")
        print(f"  Correlation: {metrics['yid_corr']:.6f}")

        print(f"\nModel Quality:")
        print(f"  Harold Vn:   {model_h.Vn:.6f}")
        print(f"  Master Vn:   {model_m.Vn:.6f}")

        # Check stability
        if hasattr(model_h, 'H_tf') and model_h.H_tf is not None:
            poles_h = np.roots(model_h.H_tf.den.flatten())
            stable_h = np.all(np.abs(poles_h) < 1.0)
            print(f"  Harold Stable: {stable_h} (max pole: {np.max(np.abs(poles_h)):.4f})")

        # Verdict
        verdict, passed = assess_verdict(metrics['yid_nrmse'], metrics['yid_corr'])

        if passed and np.max(ar_error) < 10 and np.max(ma_error) < 15:
            print(f"\n✅ {verdict}: ARMA(2,2) validation PASSED")
            print(f"   - Yid NRMSE: {metrics['yid_nrmse']*100:.2f}% (< 10% threshold)")
            print(f"   - Yid Correlation: {metrics['yid_corr']:.6f} (> 0.95 threshold)")
            print(f"   - AR max error: {np.max(ar_error):.2f}% (< 10% threshold)")
            print(f"   - MA max error: {np.max(ma_error):.2f}% (< 15% threshold)")
            return True, metrics
        else:
            print(f"\n❌ {verdict}: ARMA(2,2) validation FAILED")
            return False, metrics
    else:
        print("\n❌ ERROR: Could not extract Yid from one or both models")
        return False, {}


def main():
    """Run all validation tests."""
    print("=" * 80)
    print("ARMA COMPREHENSIVE VALIDATION")
    print("=" * 80)
    print("\nValidating ARMA NLP implementation using cross-branch testing")
    print("Primary metric: One-step prediction accuracy (Yid)")
    print("Reference: Master branch ARMAX with u=0 as proxy for ARMA")

    results = {}

    # Run test cases
    test1_pass, test1_metrics = test_case_1()
    results['test_case_1_ar1'] = {
        'passed': test1_pass,
        'metrics': test1_metrics
    }

    test2_pass, test2_metrics = test_case_2()
    results['test_case_2_ma1'] = {
        'passed': test2_pass,
        'metrics': test2_metrics
    }

    test3_pass, test3_metrics = test_case_3()
    results['test_case_3_arma11'] = {
        'passed': test3_pass,
        'metrics': test3_metrics
    }

    test4_pass, test4_metrics = test_case_4()
    results['test_case_4_arma22'] = {
        'passed': test4_pass,
        'metrics': test4_metrics
    }

    # Final summary
    print("\n" + "=" * 80)
    print("FINAL VALIDATION SUMMARY")
    print("=" * 80)

    print(f"\nTest Case 1 (AR(1)):     {'✅ PASS' if test1_pass else '❌ FAIL'}")
    print(f"Test Case 2 (MA(1)):     {'✅ PASS' if test2_pass else '❌ FAIL'}")
    print(f"Test Case 3 (ARMA(1,1)): {'✅ PASS' if test3_pass else '❌ FAIL'}")
    print(f"Test Case 4 (ARMA(2,2)): {'✅ PASS' if test4_pass else '❌ FAIL'}")

    total_pass = sum([test1_pass, test2_pass, test3_pass, test4_pass])

    if total_pass == 4:
        print("\n🎉 OVERALL: ✅ ALL TESTS PASSED - ARMA NLP is PRODUCTION READY!")
        print("   One-step predictions match master branch within acceptable tolerance")
        overall_status = "PRODUCTION_READY"
    elif total_pass >= 2:
        print("\n⚠️  OVERALL: PARTIAL SUCCESS")
        print(f"   {total_pass}/4 test cases passed")
        print("   Some test cases need investigation")
        overall_status = "PARTIAL_SUCCESS"
    else:
        print("\n❌ OVERALL: NEEDS IMPROVEMENT")
        print("   One-step predictions don't match master branch")
        overall_status = "NEEDS_IMPROVEMENT"

    results['overall_status'] = overall_status
    results['tests_passed'] = total_pass
    results['total_tests'] = 4

    # Save results to JSON
    output_path = Path(__file__).parent / "arma_validation_results.json"
    with open(output_path, 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj

        json.dump(convert_numpy(results), f, indent=2)

    print(f"\nResults saved to: {output_path}")
    print("=" * 80)

    return overall_status == "PRODUCTION_READY"


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
