"""
Validation script to ensure the optimized loop-based parsim_y_tilde_estimation_compiled
function produces numerically equivalent results to the original implementation.
"""

import numpy as np

# Import the compiled function
from src.sippy.utils.compiled_utils import parsim_y_tilde_estimation_compiled


def original_y_tilde_estimation(H_K, Uf, G_K, Yf, i, m, l_):
    """
    Original implementation using matrix slicing and np.dot for comparison.
    This is the reference implementation.
    """
    y_tilde = np.dot(H_K[0:l_, :], Uf[m * i : m * (i + 1), :])

    for j in range(1, i):
        y_tilde = (
            y_tilde
            + np.dot(
                H_K[l_ * j : l_ * (j + 1), :],
                Uf[m * (i - j) : m * (i - j + 1), :],
            )
            + np.dot(
                G_K[l_ * j : l_ * (j + 1), :],
                Yf[l_ * (i - j) : l_ * (i - j + 1), :],
            )
        )

    return y_tilde


def create_test_case(l_=2, m=3, f=20, n_cols=100, seed=42):
    """Create deterministic test matrices."""
    np.random.seed(seed)

    H_K = np.random.randn(l_ * f, m)
    G_K = np.random.randn(l_ * f, l_)
    Uf = np.random.randn(m * f, n_cols)
    Yf = np.random.randn(l_ * f, n_cols)

    return H_K, G_K, Uf, Yf


def validate_implementation():
    """Validate that optimized and original implementations match."""
    print("=" * 70)
    print("VALIDATION: Loop-based vs Matrix-based Implementation")
    print("=" * 70)

    test_configs = [
        {"name": "Small", "l_": 2, "m": 2, "f": 10, "n_cols": 50},
        {"name": "Medium", "l_": 2, "m": 3, "f": 20, "n_cols": 100},
        {"name": "Large", "l_": 3, "m": 3, "f": 50, "n_cols": 200},
        {"name": "SISO", "l_": 1, "m": 1, "f": 15, "n_cols": 80},
        {"name": "MIMO", "l_": 4, "m": 4, "f": 30, "n_cols": 150},
    ]

    all_passed = True

    for config in test_configs:
        print(f"\n{config['name']} system (l={config['l_']}, m={config['m']}, f={config['f']})")
        print("-" * 70)

        H_K, G_K, Uf, Yf = create_test_case(
            config["l_"], config["m"], config["f"], config["n_cols"]
        )

        # Test multiple iterations
        max_errors = []
        rel_errors = []

        for i in range(1, config["f"]):
            # Original implementation
            original = original_y_tilde_estimation(
                H_K, Uf, G_K, Yf, i, config["m"], config["l_"]
            )

            # Optimized implementation
            optimized = parsim_y_tilde_estimation_compiled(
                H_K, Uf, G_K, Yf, i, config["m"], config["l_"], config["f"]
            )

            # Compute errors
            abs_error = np.max(np.abs(original - optimized))
            rel_error = abs_error / (np.max(np.abs(original)) + 1e-15)

            max_errors.append(abs_error)
            rel_errors.append(rel_error)

        max_abs_error = np.max(max_errors)
        max_rel_error = np.max(rel_errors)
        mean_rel_error = np.mean(rel_errors)

        # Check tolerance
        tolerance = 1e-10
        passed = max_rel_error < tolerance

        print(f"  Iterations tested: {config['f'] - 1}")
        print(f"  Max absolute error: {max_abs_error:.2e}")
        print(f"  Max relative error: {max_rel_error:.2e}")
        print(f"  Mean relative error: {mean_rel_error:.2e}")
        print(f"  Tolerance: {tolerance:.2e}")
        print(f"  Status: {'PASS ✓' if passed else 'FAIL ✗'}")

        if not passed:
            all_passed = False
            print(f"  WARNING: Errors exceed tolerance!")

    print("\n" + "=" * 70)
    if all_passed:
        print("VALIDATION SUCCESSFUL: All tests passed ✓")
    else:
        print("VALIDATION FAILED: Some tests failed ✗")
    print("=" * 70)

    return all_passed


def test_edge_cases():
    """Test edge cases."""
    print("\n" + "=" * 70)
    print("EDGE CASE TESTS")
    print("=" * 70)

    all_passed = True

    # Edge case 1: i=1 (only initial term, no accumulation loop)
    print("\nEdge case 1: i=1 (no accumulation loop)")
    print("-" * 70)
    H_K, G_K, Uf, Yf = create_test_case(l_=2, m=2, f=10, n_cols=50)
    i = 1
    m, l_, f = 2, 2, 10

    original = original_y_tilde_estimation(H_K, Uf, G_K, Yf, i, m, l_)
    optimized = parsim_y_tilde_estimation_compiled(H_K, Uf, G_K, Yf, i, m, l_, f)

    error = np.max(np.abs(original - optimized))
    passed = error < 1e-10

    print(f"  Max error: {error:.2e}")
    print(f"  Status: {'PASS ✓' if passed else 'FAIL ✗'}")
    all_passed = all_passed and passed

    # Edge case 2: Small l_ values
    print("\nEdge case 2: l_=1 (single output)")
    print("-" * 70)
    H_K, G_K, Uf, Yf = create_test_case(l_=1, m=3, f=15, n_cols=100)
    i = 10
    m, l_, f = 3, 1, 15

    original = original_y_tilde_estimation(H_K, Uf, G_K, Yf, i, m, l_)
    optimized = parsim_y_tilde_estimation_compiled(H_K, Uf, G_K, Yf, i, m, l_, f)

    error = np.max(np.abs(original - optimized))
    passed = error < 1e-10

    print(f"  Max error: {error:.2e}")
    print(f"  Status: {'PASS ✓' if passed else 'FAIL ✗'}")
    all_passed = all_passed and passed

    # Edge case 3: Large future horizon
    print("\nEdge case 3: f=50 (large future horizon)")
    print("-" * 70)
    H_K, G_K, Uf, Yf = create_test_case(l_=2, m=2, f=50, n_cols=100)
    i = 49
    m, l_, f = 2, 2, 50

    original = original_y_tilde_estimation(H_K, Uf, G_K, Yf, i, m, l_)
    optimized = parsim_y_tilde_estimation_compiled(H_K, Uf, G_K, Yf, i, m, l_, f)

    error = np.max(np.abs(original - optimized))
    passed = error < 1e-10

    print(f"  Max error: {error:.2e}")
    print(f"  Status: {'PASS ✓' if passed else 'FAIL ✗'}")
    all_passed = all_passed and passed

    print("\n" + "=" * 70)
    if all_passed:
        print("EDGE CASE TESTS PASSED ✓")
    else:
        print("EDGE CASE TESTS FAILED ✗")
    print("=" * 70)

    return all_passed


def main():
    """Run all validation tests."""
    validation_passed = validate_implementation()
    edge_cases_passed = test_edge_cases()

    print("\n" + "=" * 70)
    print("FINAL VALIDATION REPORT")
    print("=" * 70)
    print(f"Standard tests: {'PASS ✓' if validation_passed else 'FAIL ✗'}")
    print(f"Edge case tests: {'PASS ✓' if edge_cases_passed else 'FAIL ✗'}")
    print("=" * 70)

    if validation_passed and edge_cases_passed:
        print("\nCONCLUSION: Optimized implementation is numerically equivalent ✓")
        print("The loop-based version maintains accuracy within 1e-10 relative error.")
        return 0
    else:
        print("\nCONCLUSION: Validation failed ✗")
        return 1


if __name__ == "__main__":
    exit(main())
