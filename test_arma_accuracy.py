"""
Test ARMA algorithm accuracy before and after improvements.
"""
import numpy as np
import sys

sys.path.insert(0, '/Users/josephj/Workspace/SIPPY/src')

from sippy.identification.algorithms.arma import ARMAAlgorithm


def generate_arma_data(ar_coeffs, ma_coeffs, n_samples=1000, noise_std=0.1, seed=42):
    """
    Generate synthetic ARMA data.

    ARMA model convention: A(q) y(k) = C(q) e(k)
    where A(q) = 1 + a1*q^-1 + a2*q^-2 + ...
          C(q) = 1 + c1*q^-1 + c2*q^-2 + ...

    This means: y[k] = -a1*y[k-1] - a2*y[k-2] - ... + e[k] + c1*e[k-1] + c2*e[k-2] + ...

    Parameters:
    -----------
    ar_coeffs : list
        AR coefficients [a1, a2, ...] in transfer function form (will be negated for generation)
    ma_coeffs : list
        MA coefficients [c1, c2, ...] in transfer function form
    n_samples : int
        Number of samples to generate
    noise_std : float
        Standard deviation of white noise
    seed : int
        Random seed for reproducibility

    Returns:
    --------
    y : ndarray
        Generated output time series (1 x n_samples)
    true_ar : ndarray
        True AR coefficients (in transfer function convention)
    true_ma : ndarray
        True MA coefficients (in transfer function convention)
    """
    np.random.seed(seed)

    na = len(ar_coeffs)
    nc = len(ma_coeffs)
    max_lag = max(na, nc)

    # Generate white noise
    e = np.random.normal(0, noise_std, n_samples + max_lag)

    # Initialize output
    y = np.zeros(n_samples + max_lag)

    # Generate ARMA process
    # A(q) y(k) = C(q) e(k) means y(k) + a1*y(k-1) + ... = e(k) + c1*e(k-1) + ...
    # So: y(k) = -a1*y(k-1) - a2*y(k-2) - ... + e(k) + c1*e(k-1) + ...
    for k in range(max_lag, n_samples + max_lag):
        # AR part: negate coefficients for generation
        ar_sum = sum(-ar_coeffs[i] * y[k-1-i] for i in range(na))

        # MA part: use coefficients as-is
        ma_sum = e[k] + sum(ma_coeffs[i] * e[k-1-i] for i in range(nc))

        y[k] = ar_sum + ma_sum

    # Remove initial transient
    y = y[max_lag:]

    return y.reshape(1, -1), np.array(ar_coeffs), np.array(ma_coeffs)


def test_arma_accuracy_simple():
    """Test ARMA(1,1) with simple known coefficients."""
    print("=" * 80)
    print("TEST 1: ARMA(1,1) - Simple case")
    print("=" * 80)

    # True model in transfer function form: (1 + a1*q^-1) y[k] = (1 + c1*q^-1) e[k]
    # This gives: y[k] = -a1*y[k-1] + e[k] + c1*e[k-1]
    # We want: y[k] = 0.7*y[k-1] + e[k] + 0.3*e[k-1]
    # So: a1 = -0.7, c1 = 0.3
    true_ar = [-0.7]  # Transfer function convention
    true_ma = [0.3]

    print(f"\nTrue AR coefficients (TF form): {true_ar}")
    print(f"True MA coefficients (TF form): {true_ma}")
    print(f"Generated model: y[k] = {-true_ar[0]:.1f}*y[k-1] + e[k] + {true_ma[0]:.1f}*e[k-1]")

    # Generate data
    y, _, _ = generate_arma_data(true_ar, true_ma, n_samples=1000, noise_std=0.1)

    print(f"Generated data shape: {y.shape}")
    print(f"Data statistics: mean={y.mean():.4f}, std={y.std():.4f}")

    # Identify using ARMA algorithm
    arma = ARMAAlgorithm()
    model = arma.identify(y=y, u=None, na=1, nc=1, tsample=1.0)

    print("\nIdentification completed")
    print(f"Model dimensions: A={model.A.shape}, B={model.B.shape}, C={model.C.shape}")

    # Extract estimated coefficients directly from model attributes
    estimated_ar = model.AR_coeffs[0, :]  # Extract AR coefficients
    estimated_ma = model.MA_coeffs[0, :]  # Extract MA coefficients

    print(f"\nEstimated AR coefficients: {estimated_ar}")
    print(f"Estimated MA coefficients: {estimated_ma}")

    # Calculate relative errors
    ar_error = np.abs(estimated_ar - true_ar) / np.abs(true_ar) * 100
    ma_error = np.abs(estimated_ma - true_ma) / np.abs(true_ma) * 100

    print(f"\nRelative errors:")
    print(f"  AR error: {ar_error[0]:.2f}%")
    print(f"  MA error: {ma_error[0]:.2f}%")

    # Check if Yid is reasonable
    if hasattr(model, 'Yid') and model.Yid is not None:
        prediction_error = np.sqrt(np.mean((model.Yid - y)**2))
        normalized_error = prediction_error / np.std(y) * 100
        print(f"\nPrediction RMSE: {prediction_error:.4f}")
        print(f"Normalized RMSE: {normalized_error:.2f}% of signal std")

    # Overall assessment
    max_error = max(ar_error[0], ma_error[0])
    print(f"\n{'='*80}")
    if max_error < 10:
        print(f"✓ PASS: Maximum error {max_error:.2f}% < 10% threshold")
    else:
        print(f"✗ FAIL: Maximum error {max_error:.2f}% >= 10% threshold")
    print(f"{'='*80}")

    return max_error < 10


def test_arma_accuracy_complex():
    """Test ARMA(2,1) with more complex coefficients."""
    print("\n" + "=" * 80)
    print("TEST 2: ARMA(2,1) - Complex case")
    print("=" * 80)

    # True model: y[k] = 0.6*y[k-1] + 0.2*y[k-2] + e[k] + 0.4*e[k-1]
    # In TF form: (1 + a1*q^-1 + a2*q^-2) y[k] = (1 + c1*q^-1) e[k]
    # So: a1 = -0.6, a2 = -0.2, c1 = 0.4
    true_ar = [-0.6, -0.2]
    true_ma = [0.4]

    print(f"\nTrue AR coefficients (TF form): {true_ar}")
    print(f"True MA coefficients (TF form): {true_ma}")
    print(f"Generated model: y[k] = {-true_ar[0]:.1f}*y[k-1] + {-true_ar[1]:.1f}*y[k-2] + e[k] + {true_ma[0]:.1f}*e[k-1]")

    # Generate data
    y, _, _ = generate_arma_data(true_ar, true_ma, n_samples=1000, noise_std=0.1)

    print(f"Generated data shape: {y.shape}")
    print(f"Data statistics: mean={y.mean():.4f}, std={y.std():.4f}")

    # Identify using ARMA algorithm
    arma = ARMAAlgorithm()
    model = arma.identify(y=y, u=None, na=2, nc=1, tsample=1.0)

    print("\nIdentification completed")
    print(f"Model dimensions: A={model.A.shape}, B={model.B.shape}, C={model.C.shape}")

    # Extract estimated coefficients directly from model attributes
    estimated_ar = model.AR_coeffs[0, :]  # Extract AR coefficients
    estimated_ma = model.MA_coeffs[0, :]  # Extract MA coefficients

    print(f"\nEstimated AR coefficients: {estimated_ar}")
    print(f"Estimated MA coefficients: {estimated_ma}")

    # Calculate relative errors
    ar_errors = np.abs(estimated_ar - true_ar) / np.abs(true_ar) * 100
    ma_error = np.abs(estimated_ma - true_ma) / np.abs(true_ma) * 100

    print(f"\nRelative errors:")
    print(f"  AR[0] error: {ar_errors[0]:.2f}%")
    print(f"  AR[1] error: {ar_errors[1]:.2f}%")
    print(f"  MA[0] error: {ma_error[0]:.2f}%")

    # Check if Yid is reasonable
    if hasattr(model, 'Yid') and model.Yid is not None:
        prediction_error = np.sqrt(np.mean((model.Yid - y)**2))
        normalized_error = prediction_error / np.std(y) * 100
        print(f"\nPrediction RMSE: {prediction_error:.4f}")
        print(f"Normalized RMSE: {normalized_error:.2f}% of signal std")

    # Overall assessment
    max_error = max(ar_errors.max(), ma_error[0])
    print(f"\n{'='*80}")
    if max_error < 10:
        print(f"✓ PASS: Maximum error {max_error:.2f}% < 10% threshold")
    else:
        print(f"✗ FAIL: Maximum error {max_error:.2f}% >= 10% threshold")
    print(f"{'='*80}")

    return max_error < 10


def test_arma_accuracy_high_snr():
    """Test ARMA(1,1) with high SNR (low noise)."""
    print("\n" + "=" * 80)
    print("TEST 3: ARMA(1,1) - High SNR (low noise)")
    print("=" * 80)

    # True model: y[k] = 0.8*y[k-1] + e[k] + 0.5*e[k-1]
    # In TF form: a1 = -0.8, c1 = 0.5
    true_ar = [-0.8]
    true_ma = [0.5]

    print(f"\nTrue AR coefficients (TF form): {true_ar}")
    print(f"True MA coefficients (TF form): {true_ma}")
    print(f"Generated model: y[k] = {-true_ar[0]:.1f}*y[k-1] + e[k] + {true_ma[0]:.1f}*e[k-1]")

    # Generate data with very low noise
    y, _, _ = generate_arma_data(true_ar, true_ma, n_samples=2000, noise_std=0.01)

    print(f"Generated data shape: {y.shape}")
    print(f"Data statistics: mean={y.mean():.4f}, std={y.std():.4f}")

    # Identify using ARMA algorithm
    arma = ARMAAlgorithm()
    model = arma.identify(y=y, u=None, na=1, nc=1, tsample=1.0)

    print("\nIdentification completed")

    # Extract estimated coefficients directly from model attributes
    estimated_ar = model.AR_coeffs[0, :]
    estimated_ma = model.MA_coeffs[0, :]

    print(f"\nEstimated AR coefficients: {estimated_ar}")
    print(f"Estimated MA coefficients: {estimated_ma}")

    # Calculate relative errors
    ar_error = np.abs(estimated_ar - true_ar) / np.abs(true_ar) * 100
    ma_error = np.abs(estimated_ma - true_ma) / np.abs(true_ma) * 100

    print(f"\nRelative errors:")
    print(f"  AR error: {ar_error[0]:.2f}%")
    print(f"  MA error: {ma_error[0]:.2f}%")

    # Check if Yid is reasonable
    if hasattr(model, 'Yid') and model.Yid is not None:
        prediction_error = np.sqrt(np.mean((model.Yid - y)**2))
        normalized_error = prediction_error / np.std(y) * 100
        print(f"\nPrediction RMSE: {prediction_error:.4f}")
        print(f"Normalized RMSE: {normalized_error:.2f}% of signal std")

    # Overall assessment
    max_error = max(ar_error[0], ma_error[0])
    print(f"\n{'='*80}")
    if max_error < 10:
        print(f"✓ PASS: Maximum error {max_error:.2f}% < 10% threshold")
    else:
        print(f"✗ FAIL: Maximum error {max_error:.2f}% >= 10% threshold")
    print(f"{'='*80}")

    return max_error < 10


if __name__ == "__main__":
    print("\n" + "█" * 80)
    print("ARMA ALGORITHM ACCURACY TEST SUITE")
    print("█" * 80)

    results = []

    # Run all tests
    results.append(("ARMA(1,1) Simple", test_arma_accuracy_simple()))
    results.append(("ARMA(2,1) Complex", test_arma_accuracy_complex()))
    results.append(("ARMA(1,1) High SNR", test_arma_accuracy_high_snr()))

    # Final summary
    print("\n" + "█" * 80)
    print("FINAL SUMMARY")
    print("█" * 80)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")

    total_passed = sum(results[i][1] for i in range(len(results)))
    print(f"\nTotal: {total_passed}/{len(results)} tests passed")
    print("█" * 80)
