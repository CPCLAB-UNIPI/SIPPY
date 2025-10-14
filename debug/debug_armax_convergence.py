"""
Debug ARMAX convergence behavior comparing harold and master branches.

This script investigates the 34.6% numerator error by:
1. Running both implementations on the same data
2. Comparing iteration-by-iteration behavior
3. Analyzing initialization strategies
4. Checking convergence criteria
"""

import sys
from pathlib import Path

import numpy as np

# Add master branch to path
MASTER_PATH = Path("/Users/josephj/Workspace/SIPPY-master")
sys.path.insert(0, str(MASTER_PATH))

# Import master branch ARMAX
from sippy_unipi.armax import Armax as MasterARMAX

# Remove master path and import harold branch
sys.path.pop(0)

# Import harold branch
from sippy.identification import SystemIdentification, SystemIdentificationConfig


def generate_arx_test_data():
    """Generate simple SISO ARX data for testing."""
    np.random.seed(789)

    npts = 300
    u = np.random.randn(1, npts)
    y = np.zeros((1, npts))

    # Generate ARX system: y[k] = 0.7*y[k-1] + 0.5*u[k-1] + noise
    for i in range(1, npts):
        y[0, i] = 0.7 * y[0, i - 1] + 0.5 * u[0, i - 1] + 0.05 * np.random.randn()

    return y, u


def debug_master_armax(y, u):
    """Run master branch ARMAX and capture details."""
    print("=" * 80)
    print("MASTER BRANCH ARMAX IDENTIFICATION")
    print("=" * 80)

    # Flatten for master branch
    y_flat = y.flatten()
    u_flat = u.flatten()

    # Create ARMAX model
    armax_model = MasterARMAX(
        na_bounds=[1, 1], nb_bounds=[1, 1], nc_bounds=[1, 1], delay_bounds=[1, 1], dt=1.0, max_iterations=200
    )

    # Run identification
    armax_model.find_best_estimate(y_flat, u_flat)

    # Extract results
    G_master = armax_model.G
    H_master = armax_model.H
    Vn_master = armax_model.Vn
    Yid_master = armax_model.Yid

    print("\nMaster Results:")
    print(f"  G(z) numerator:   {G_master.num[0][0]}")
    print(f"  G(z) denominator: {G_master.den[0][0]}")
    print(f"  H(z) numerator:   {H_master.num[0][0]}")
    print(f"  H(z) denominator: {H_master.den[0][0]}")
    print(f"  Variance (Vn):    {Vn_master}")
    print(f"  Max reached:      {armax_model.max_reached}")

    return {
        "G_num": G_master.num[0][0],
        "G_den": G_master.den[0][0],
        "H_num": H_master.num[0][0],
        "H_den": H_master.den[0][0],
        "Vn": Vn_master,
        "Yid": Yid_master,
        "max_reached": armax_model.max_reached,
    }


def debug_harold_armax(y, u):
    """Run harold branch ARMAX and capture details."""
    print("\n" + "=" * 80)
    print("HAROLD BRANCH ARMAX IDENTIFICATION")
    print("=" * 80)

    # Harold branch identification
    config = SystemIdentificationConfig(method="ARMAX")
    config.na = 1
    config.nb = 1
    config.nc = 1
    config.nk = 1
    config.max_iterations = 200
    identifier = SystemIdentification(config)
    model_harold = identifier.identify(y=y, u=u)

    # Extract transfer functions
    if model_harold.G_tf is not None:
        G_num_harold = model_harold.G_tf.num[0]
        G_den_harold = model_harold.G_tf.den[0]
    else:
        G_num_harold = None
        G_den_harold = None

    if model_harold.H_tf is not None:
        H_num_harold = model_harold.H_tf.num[0]
        H_den_harold = model_harold.H_tf.den[0]
    else:
        H_num_harold = None
        H_den_harold = None

    print("\nHarold Results:")
    print(f"  G(z) numerator:   {G_num_harold}")
    print(f"  G(z) denominator: {G_den_harold}")
    print(f"  H(z) numerator:   {H_num_harold}")
    print(f"  H(z) denominator: {H_den_harold}")
    print(f"  Variance (Vn):    {model_harold.Vn}")

    # Check if Yid is available
    Yid_harold = getattr(model_harold, "Yid", None)

    return {
        "G_num": G_num_harold,
        "G_den": G_den_harold,
        "H_num": H_num_harold,
        "H_den": H_den_harold,
        "Vn": model_harold.Vn,
        "Yid": Yid_harold,
        "model": model_harold,
    }


def compare_results(master_results, harold_results):
    """Compare master and harold results."""
    print("\n" + "=" * 80)
    print("COMPARISON ANALYSIS")
    print("=" * 80)

    # Normalize transfer functions
    master_num = np.trim_zeros(master_results["G_num"], "fb")
    master_den = np.trim_zeros(master_results["G_den"], "fb")
    harold_num = np.trim_zeros(harold_results["G_num"], "fb")
    harold_den = np.trim_zeros(harold_results["G_den"], "fb")

    # Normalize by leading denominator coefficient
    master_num_norm = master_num / master_den[0]
    master_den_norm = master_den / master_den[0]
    harold_num_norm = harold_num / harold_den[0]
    harold_den_norm = harold_den / harold_den[0]

    # Compute errors
    num_error = np.max(np.abs(harold_num_norm - master_num_norm))
    den_error = np.max(np.abs(harold_den_norm - master_den_norm))

    print("\n1. TRANSFER FUNCTION COEFFICIENTS:")
    print(f"\n   Numerator (B coefficients):")
    print(f"     Master:  {master_num_norm}")
    print(f"     Harold:  {harold_num_norm}")
    print(f"     Error:   {num_error:.6e} ({100 * num_error / np.max(np.abs(master_num_norm)):.2f}%)")

    print(f"\n   Denominator (A coefficients):")
    print(f"     Master:  {master_den_norm}")
    print(f"     Harold:  {harold_den_norm}")
    print(f"     Error:   {den_error:.6e} ({100 * den_error / np.max(np.abs(master_den_norm)):.2f}%)")

    print("\n2. NOISE MODEL (H transfer function):")
    master_h_num = np.trim_zeros(master_results["H_num"], "fb")
    master_h_den = np.trim_zeros(master_results["H_den"], "fb")
    harold_h_num = np.trim_zeros(harold_results["H_num"], "fb")
    harold_h_den = np.trim_zeros(harold_results["H_den"], "fb")

    print(f"   Master H numerator:   {master_h_num}")
    print(f"   Harold H numerator:   {harold_h_num}")
    print(f"   Master H denominator: {master_h_den}")
    print(f"   Harold H denominator: {harold_h_den}")

    print("\n3. VARIANCE (Vn):")
    print(f"   Master: {master_results['Vn']}")
    print(f"   Harold: {harold_results['Vn']}")
    print(f"   Ratio:  {harold_results['Vn'] / master_results['Vn']:.6f}")

    print("\n4. CONVERGENCE STATUS:")
    print(f"   Master max_reached: {master_results.get('max_reached', 'N/A')}")

    print("\n" + "=" * 80)
    print("ERROR SUMMARY")
    print("=" * 80)
    print(f"Numerator error:    {num_error:.6e} ({100 * num_error / np.max(np.abs(master_num_norm)):.2f}%)")
    print(f"Denominator error:  {den_error:.6e} ({100 * den_error / np.max(np.abs(master_den_norm)):.2f}%)")
    print(f"Variance ratio:     {harold_results['Vn'] / master_results['Vn']:.6f}")

    # Determine if acceptable
    if num_error < 1e-7 and den_error < 1e-7:
        print("\n✅ PASS: Errors within machine precision")
    elif num_error < 0.01 and den_error < 0.01:
        print("\n⚠️ CONDITIONAL: Errors within 1% (acceptable for iterative methods)")
    else:
        print("\n❌ FAIL: Errors exceed acceptable tolerance")
        print("\nPossible causes:")
        print("  1. Different initialization strategies")
        print("  2. Different convergence criteria")
        print("  3. Different noise estimation updates")
        print("  4. Local minima in iterative optimization")


def main():
    """Main debug function."""
    print("ARMAX Convergence Investigation")
    print("=" * 80)
    print("Comparing harold branch vs master branch implementations")
    print("=" * 80)

    # Generate test data
    print("\nGenerating test data...")
    y, u = generate_arx_test_data()
    print(f"Data shape: y={y.shape}, u={u.shape}")
    print(f"Data length: {y.shape[1]} samples")

    # Run master branch
    master_results = debug_master_armax(y, u)

    # Run harold branch
    harold_results = debug_harold_armax(y, u)

    # Compare results
    compare_results(master_results, harold_results)

    print("\n" + "=" * 80)
    print("INVESTIGATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
