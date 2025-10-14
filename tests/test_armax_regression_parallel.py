"""
Test suite for ARMAX ILLS regression matrix parallel optimization.

Tests correctness (parallel vs sequential) and performance (speedup measurement).
"""

import time

import numpy as np
import pytest

# Import both compiled and non-compiled versions
try:
    from sippy.utils.compiled_utils import (
        NUMBA_AVAILABLE,
        build_armax_regression_parallel,
    )

    HAS_NUMBA = NUMBA_AVAILABLE and build_armax_regression_parallel is not None
except ImportError:
    HAS_NUMBA = False
    build_armax_regression_parallel = None


def build_armax_regression_sequential(y, u, noise_hat, na, nb, nc, nk, max_order, N_eff):
    """Sequential (non-parallel) version for comparison."""
    sum_order = na + nb + nc
    Phi = np.zeros((N_eff, sum_order))

    for i in range(N_eff):
        # AR part (lagged outputs)
        for j in range(na):
            Phi[i, j] = -y[i + max_order - 1 - j]

        # X part (lagged inputs)
        for j in range(nb):
            Phi[i, na + j] = u[max_order + i - 1 - (nk + j)]

        # MA part (estimated noise terms)
        for j in range(nc):
            Phi[i, na + nb + j] = noise_hat[max_order + i - 1 - j]

    return Phi


# ============================================================================
# CORRECTNESS TESTS
# ============================================================================


@pytest.mark.skipif(not HAS_NUMBA, reason="Numba not available")
@pytest.mark.parametrize(
    "na,nb,nc,nk,N",
    [
        (1, 1, 1, 1, 100),  # Simple case
        (2, 2, 2, 1, 200),  # Medium case
        (5, 5, 5, 2, 500),  # Complex case
        (10, 10, 10, 3, 1000),  # Large orders
    ],
)
def test_parallel_correctness_various_orders(na, nb, nc, nk, N):
    """Test that parallel version produces identical results to sequential."""
    # Generate synthetic data
    np.random.seed(42)
    y = np.random.randn(N)
    u = np.random.randn(N)
    noise_hat = np.random.randn(N) * 0.1

    max_order = max(na, nb + nk, nc)
    N_eff = N - max_order

    # Build regression matrices
    Phi_sequential = build_armax_regression_sequential(
        y, u, noise_hat, na, nb, nc, nk, max_order, N_eff
    )
    Phi_parallel = build_armax_regression_parallel(
        y, u, noise_hat, na, nb, nc, nk, max_order, N_eff
    )

    # Check shapes match
    assert Phi_sequential.shape == Phi_parallel.shape

    # Check values match (tight tolerance)
    np.testing.assert_allclose(
        Phi_parallel, Phi_sequential, rtol=1e-12, atol=1e-12, err_msg=f"Mismatch for na={na}, nb={nb}, nc={nc}, nk={nk}, N={N}"
    )


@pytest.mark.skipif(not HAS_NUMBA, reason="Numba not available")
def test_parallel_correctness_siso():
    """Test SISO case specifically."""
    np.random.seed(123)
    N = 500
    na, nb, nc, nk = 3, 3, 3, 1

    # Generate SISO data
    y = np.sin(np.linspace(0, 10, N)) + np.random.randn(N) * 0.1
    u = np.random.randn(N)
    noise_hat = np.random.randn(N) * 0.05

    max_order = max(na, nb + nk, nc)
    N_eff = N - max_order

    Phi_seq = build_armax_regression_sequential(
        y, u, noise_hat, na, nb, nc, nk, max_order, N_eff
    )
    Phi_par = build_armax_regression_parallel(
        y, u, noise_hat, na, nb, nc, nk, max_order, N_eff
    )

    # Verify numerical equality
    np.testing.assert_allclose(Phi_par, Phi_seq, rtol=1e-12, atol=1e-12)

    # Verify structure (columns are correct size)
    assert Phi_par.shape[1] == na + nb + nc


@pytest.mark.skipif(not HAS_NUMBA, reason="Numba not available")
def test_parallel_edge_cases():
    """Test edge cases like minimal orders."""
    np.random.seed(456)

    # Test with minimal orders (na=1, nb=1, nc=1)
    N = 100
    na, nb, nc, nk = 1, 1, 1, 1
    y = np.random.randn(N)
    u = np.random.randn(N)
    noise_hat = np.random.randn(N) * 0.1

    max_order = max(na, nb + nk, nc)
    N_eff = N - max_order

    Phi_seq = build_armax_regression_sequential(
        y, u, noise_hat, na, nb, nc, nk, max_order, N_eff
    )
    Phi_par = build_armax_regression_parallel(
        y, u, noise_hat, na, nb, nc, nk, max_order, N_eff
    )

    np.testing.assert_allclose(Phi_par, Phi_seq, rtol=1e-12, atol=1e-12)


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================


@pytest.mark.skipif(not HAS_NUMBA, reason="Numba not available")
@pytest.mark.parametrize(
    "na,nb,nc,N",
    [
        (3, 3, 3, 1000),  # Small
        (5, 5, 5, 5000),  # Medium
        (10, 10, 10, 10000),  # Large
    ],
)
def test_parallel_performance(na, nb, nc, N):
    """Benchmark parallel vs sequential performance."""
    np.random.seed(789)
    nk = 1
    y = np.random.randn(N)
    u = np.random.randn(N)
    noise_hat = np.random.randn(N) * 0.1

    max_order = max(na, nb + nk, nc)
    N_eff = N - max_order

    # Warm up JIT
    _ = build_armax_regression_parallel(
        y[:100], u[:100], noise_hat[:100], 2, 2, 2, 1, 5, 95
    )

    # Time sequential version
    num_runs_seq = 10
    start_seq = time.perf_counter()
    for _ in range(num_runs_seq):
        Phi_seq = build_armax_regression_sequential(
            y, u, noise_hat, na, nb, nc, nk, max_order, N_eff
        )
    time_seq = (time.perf_counter() - start_seq) / num_runs_seq

    # Time parallel version
    num_runs_par = 10
    start_par = time.perf_counter()
    for _ in range(num_runs_par):
        Phi_par = build_armax_regression_parallel(
            y, u, noise_hat, na, nb, nc, nk, max_order, N_eff
        )
    time_par = (time.perf_counter() - start_par) / num_runs_par

    speedup = time_seq / time_par

    print(
        f"\nPerformance (na={na}, nb={nb}, nc={nc}, N={N}):"
        f"\n  Sequential: {time_seq*1000:.2f} ms"
        f"\n  Parallel:   {time_par*1000:.2f} ms"
        f"\n  Speedup:    {speedup:.2f}x"
    )

    # Verify correctness still holds
    np.testing.assert_allclose(Phi_par, Phi_seq, rtol=1e-12, atol=1e-12)

    # Speedup should be > 1.5x for larger problems
    # (Conservative threshold to account for overhead)
    if N >= 5000:
        assert speedup > 1.5, f"Expected speedup > 1.5x, got {speedup:.2f}x"


@pytest.mark.skipif(not HAS_NUMBA, reason="Numba not available")
def test_detailed_performance_report():
    """Generate detailed performance report across problem sizes."""
    print("\n" + "=" * 70)
    print("ARMAX ILLS Regression Matrix Parallelization Performance Report")
    print("=" * 70)

    test_cases = [
        (2, 2, 2, 500, "Tiny"),
        (3, 3, 3, 1000, "Small"),
        (5, 5, 5, 5000, "Medium"),
        (7, 7, 7, 10000, "Large"),
        (10, 10, 10, 20000, "Very Large"),
    ]

    results = []

    for na, nb, nc, N, label in test_cases:
        np.random.seed(42)
        nk = 1
        y = np.random.randn(N)
        u = np.random.randn(N)
        noise_hat = np.random.randn(N) * 0.1

        max_order = max(na, nb + nk, nc)
        N_eff = N - max_order

        # Warm up
        _ = build_armax_regression_parallel(
            y[:100], u[:100], noise_hat[:100], 2, 2, 2, 1, 5, 95
        )

        # Time sequential
        num_runs = 5
        start = time.perf_counter()
        for _ in range(num_runs):
            Phi_seq = build_armax_regression_sequential(
                y, u, noise_hat, na, nb, nc, nk, max_order, N_eff
            )
        time_seq = (time.perf_counter() - start) / num_runs

        # Time parallel
        start = time.perf_counter()
        for _ in range(num_runs):
            Phi_par = build_armax_regression_parallel(
                y, u, noise_hat, na, nb, nc, nk, max_order, N_eff
            )
        time_par = (time.perf_counter() - start) / num_runs

        speedup = time_seq / time_par

        # Verify correctness
        np.testing.assert_allclose(Phi_par, Phi_seq, rtol=1e-12, atol=1e-12)

        results.append((label, na, nb, nc, N, N_eff, time_seq, time_par, speedup))

    # Print table
    print(
        f"\n{'Problem':<15} {'Orders':<12} {'N_eff':<8} {'Sequential':<12} {'Parallel':<12} {'Speedup'}"
    )
    print("-" * 70)
    for label, na, nb, nc, N, N_eff, t_seq, t_par, speedup in results:
        orders = f"({na},{nb},{nc})"
        print(
            f"{label:<15} {orders:<12} {N_eff:<8} {t_seq*1000:>9.2f} ms  {t_par*1000:>9.2f} ms  {speedup:>5.2f}x"
        )

    print("=" * 70)
    print(
        "NOTE: Speedup depends on CPU cores, problem size, and Numba compilation."
    )
    print("=" * 70 + "\n")


# ============================================================================
# INTEGRATION TEST
# ============================================================================


@pytest.mark.skipif(not HAS_NUMBA, reason="Numba not available")
def test_integration_with_armax():
    """Test that the optimization integrates correctly with ARMAX algorithm."""
    try:
        from sippy.identification.algorithms.armax import ARMAXAlgorithm

        # Generate synthetic ARMAX system
        np.random.seed(999)
        N = 1000
        u = np.random.randn(N)
        y = np.zeros(N)

        # Simple ARMAX(2,2,2) system
        a1, a2 = 0.5, 0.2
        b1, b2 = 0.8, 0.3
        c1, c2 = 0.6, 0.1
        noise = np.random.randn(N) * 0.1

        for k in range(2, N):
            y[k] = (
                -a1 * y[k - 1]
                - a2 * y[k - 2]
                + b1 * u[k - 1]
                + b2 * u[k - 2]
                + noise[k]
                + c1 * noise[k - 1]
                + c2 * noise[k - 2]
            )

        # Identify using ARMAX ILLS (will use parallel version automatically)
        armax_alg = ARMAXAlgorithm()
        model = armax_alg.identify(y=y, u=u, na=2, nb=2, nc=2, nk=1, mode="ILLS")

        assert model is not None, "ARMAX identification failed"
        assert hasattr(model, "A"), "Model missing state-space matrices"

        print("\nARMAX Integration Test:")
        print(f"  Model identified successfully")
        print(f"  A matrix shape: {model.A.shape}")
        print(f"  Parallel optimization used automatically")

    except ImportError as e:
        pytest.skip(f"ARMAX not available for integration test: {e}")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])
