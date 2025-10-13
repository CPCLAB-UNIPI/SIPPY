"""
Debug ARMAX Poor Fit Quality Issue - Harold Branch Only

This script creates a simple ARMA(1,1) test case to isolate the ARMAX poor fit issue.
"""

import numpy as np

# Harold branch imports
from sippy.identification import SystemIdentification, SystemIdentificationConfig


def generate_armax_test_data(na=1, nb=1, nc=1, nk=1, npts=500, seed=42):
    """
    Generate test data from a known ARMAX system.

    System: A(q) y(k) = B(q) u(k-nk) + C(q) e(k) + e(k)

    For na=1, nb=1, nc=1, nk=1:
    y[k] = 0.7*y[k-1] + 0.5*u[k-1] + 0.3*e[k-1] + e[k]
    """
    np.random.seed(seed)

    # True parameters
    A_true = np.array([0.7])  # AR coefficient
    B_true = np.array([0.5])  # X coefficient
    C_true = np.array([0.3])  # MA coefficient

    # Generate input
    u = np.random.randn(npts)

    # Generate white noise
    e = np.random.randn(npts) * 0.1

    # Generate output using the ARMAX model
    y = np.zeros(npts)
    for k in range(max(na, nb + nk, nc), npts):
        # AR part
        ar_part = A_true[0] * y[k-1]

        # X part (with delay nk)
        x_part = B_true[0] * u[k-nk]

        # MA part
        ma_part = C_true[0] * e[k-1]

        # Output
        y[k] = ar_part + x_part + ma_part + e[k]

    return y, u, A_true, B_true, C_true, e


def compute_fit_percentage(y_true, y_pred):
    """
    Compute fit percentage.

    Fit% = 100 * (1 - ||y_true - y_pred|| / ||y_true - mean(y_true)||)
    """
    numerator = np.linalg.norm(y_true - y_pred)
    denominator = np.linalg.norm(y_true - np.mean(y_true))

    if denominator < 1e-12:
        return 100.0 if numerator < 1e-12 else 0.0

    fit_percent = 100.0 * (1.0 - numerator / denominator)
    return fit_percent


def test_armax_harold_branch():
    """Test ARMAX on harold branch."""
    print("=" * 80)
    print("Testing ARMAX on Harold Branch")
    print("=" * 80)

    # Generate test data
    y, u, A_true, B_true, C_true, e = generate_armax_test_data()

    print(f"\nTrue parameters:")
    print(f"  A (AR): {A_true}")
    print(f"  B (X):  {B_true}")
    print(f"  C (MA): {C_true}")
    print(f"\nData shape: y={y.shape}, u={u.shape}")

    # Identify using harold branch
    config = SystemIdentificationConfig(
        method="ARMAX",
        na=1,
        nb=1,
        nc=1,
        nk=1,
        max_iterations=200
    )

    identifier = SystemIdentification(config)

    # Reshape for SIPPY (expects (n_outputs, n_samples))
    y_sippy = y.reshape(1, -1)
    u_sippy = u.reshape(1, -1)

    try:
        model_harold = identifier.identify(y=y_sippy, u=u_sippy)

        print(f"\nIdentification successful!")
        print(f"  Model type: {type(model_harold)}")
        print(f"  Model order: {model_harold.n}")
        print(f"  A matrix shape: {model_harold.A.shape}")
        print(f"  B matrix shape: {model_harold.B.shape}")
        print(f"  C matrix shape: {model_harold.C.shape}")

        # Extract estimated parameters from state-space matrices
        # For ARMAX in companion form:
        # A matrix last row contains -A_coeffs
        # B matrix first na rows contain B_coeffs
        # C matrix last nc columns contain C_coeffs

        print(f"\nState-space matrices:")
        print(f"  A:\n{model_harold.A}")
        print(f"  B:\n{model_harold.B}")
        print(f"  C:\n{model_harold.C}")
        print(f"  D:\n{model_harold.D}")

        # Try to simulate
        try:
            y_sim = model_harold.simulate(u_sippy)
            print(f"\nSimulation successful!")
            print(f"  y_sim shape: {y_sim.shape}")

            # Compute fit percentage
            fit_pct = compute_fit_percentage(y_sippy.flatten(), y_sim.flatten())
            print(f"  Fit percentage: {fit_pct:.2f}%")

            if fit_pct < 0:
                print(f"  ❌ POOR FIT - This is the issue!")

                # Debug: Check if model is stable
                eigenvalues = np.linalg.eigvals(model_harold.A)
                print(f"\n  A matrix eigenvalues: {eigenvalues}")
                print(f"  Max abs eigenvalue: {np.max(np.abs(eigenvalues)):.4f}")

                if np.max(np.abs(eigenvalues)) > 1.0:
                    print(f"  ⚠️  Model is UNSTABLE!")
            else:
                print(f"  ✅ GOOD FIT!")

        except Exception as e:
            print(f"\n❌ Simulation failed: {e}")

        # Check if model has Yid (one-step-ahead predictions)
        if hasattr(model_harold, 'Yid') and model_harold.Yid is not None:
            print(f"\nModel has one-step-ahead predictions (Yid)")
            print(f"  Yid shape: {model_harold.Yid.shape}")

            # Compute fit with Yid
            fit_pct_yid = compute_fit_percentage(y_sippy.flatten(), model_harold.Yid.flatten())
            print(f"  Fit percentage (Yid): {fit_pct_yid:.2f}%")

        # Check if model has G_tf and H_tf
        if hasattr(model_harold, 'G_tf') and model_harold.G_tf is not None:
            print(f"\nModel has transfer function G_tf:")
            print(f"  {model_harold.G_tf}")

        if hasattr(model_harold, 'H_tf') and model_harold.H_tf is not None:
            print(f"\nModel has transfer function H_tf:")
            print(f"  {model_harold.H_tf}")

        return model_harold

    except Exception as e:
        print(f"\n❌ Identification failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    print("ARMAX Poor Fit Quality Investigation - Harold Branch Only")
    print("=" * 80)
    print("TASK 5 from MIGRATION_ACCURACY_TODO.md")
    print("=" * 80)

    # Test harold branch
    model_harold = test_armax_harold_branch()

    print("\n" + "=" * 80)
    print("Investigation Complete")
    print("=" * 80)
