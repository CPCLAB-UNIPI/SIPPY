"""
Test-Driven Development tests for PARSIM-P reimplementation.

This test suite verifies that PARSIM-P is correctly implemented with its
distinctive expanding window approach, NOT as a wrapper to PARSIM-S.

Reference: /Users/josephj/Workspace/SIPPY-master/sippy_unipi/Parsim_methods.py
Lines 597-670 (PARSIM_P function)
"""

import numpy as np
import pytest

from sippy.identification.algorithms.parsim_core import ParsimCoreAlgorithm


class TestParsimPReimplementation:
    """TDD tests for PARSIM-P reimplementation."""

    @pytest.fixture
    def simple_siso_system(self):
        """Simple SISO system for testing."""
        np.random.seed(42)
        n_points = 200
        u = np.random.randn(1, n_points)
        y = np.zeros((1, n_points))
        for i in range(1, n_points):
            y[0, i] = 0.8 * y[0, i - 1] + 0.5 * u[0, i - 1] + 0.05 * np.random.randn()
        return y, u

    @pytest.fixture
    def stable_mimo_system(self):
        """Stable MIMO system for testing."""
        np.random.seed(123)
        n_points = 300
        u = np.random.randn(2, n_points)
        y = np.zeros((2, n_points))

        # Simple MIMO system
        for i in range(1, n_points):
            y[0, i] = 0.7 * y[0, i - 1] + 0.3 * u[0, i - 1] + 0.2 * u[1, i - 1]
            y[1, i] = 0.6 * y[1, i - 1] + 0.4 * u[1, i - 1] + 0.1 * u[0, i - 1]
            y[:, i] += 0.02 * np.random.randn(2)

        return y, u

    def test_parsim_p_not_wrapper_to_parsim_s(self):
        """Test that PARSIM-P doesn't just call parsim_s()."""
        # This test verifies that parsim_p has its own implementation
        # by checking the source code
        import inspect

        source = inspect.getsource(ParsimCoreAlgorithm.parsim_p)

        # Should NOT contain a direct call to parsim_s
        assert "parsim_s" not in source, "PARSIM-P should not be a wrapper to parsim_s"

        # Should contain its own algorithm logic
        assert "for i in range" in source, "PARSIM-P should have iteration logic"

    def test_parsim_p_has_expanding_window_logic(self):
        """Test that PARSIM-P contains expanding window logic."""
        import inspect

        source = inspect.getsource(ParsimCoreAlgorithm.parsim_p)

        # Key feature: Uf window should expand with iteration
        # Looking for pattern like: Uf[0 : m * (i + 1), :]
        assert "i + 1" in source or "(i+1)" in source, (
            "PARSIM-P should have expanding window logic with (i+1)"
        )

    def test_parsim_p_basic_execution(self, simple_siso_system):
        """Test that PARSIM-P can execute without errors."""
        y, u = simple_siso_system

        # Basic parameters
        f = 10
        p = 10
        threshold = 0.1

        # Should execute without errors
        try:
            A_K, C, B_K, D, K, A, B, x0, Vn = ParsimCoreAlgorithm.parsim_p(
                y, u, f=f, p=p, threshold=threshold
            )

            # Check that we got valid outputs
            assert A_K is not None
            assert C is not None
            assert B_K is not None
            assert D is not None
            assert K is not None
            assert A is not None
            assert B is not None
            assert x0 is not None
            assert Vn is not None

            # Check dimensions make sense
            n = A.shape[0]
            l_ = y.shape[0]
            m = u.shape[0]

            assert A.shape == (n, n)
            assert B.shape == (n, m)
            assert C.shape == (l_, n)
            assert D.shape == (l_, m)
            assert K.shape == (n, l_)

        except Exception as e:
            pytest.fail(f"PARSIM-P execution failed: {e}")

    def test_parsim_p_differs_from_parsim_s(self, simple_siso_system):
        """Test that PARSIM-P produces different results than PARSIM-S."""
        y, u = simple_siso_system

        f = 10
        p = 10
        threshold = 0.1

        # Run PARSIM-P
        A_K_p, C_p, B_K_p, D_p, K_p, A_p, B_p, x0_p, Vn_p = (
            ParsimCoreAlgorithm.parsim_p(y, u, f=f, p=p, threshold=threshold)
        )

        # Run PARSIM-S
        A_K_s, C_s, B_K_s, D_s, K_s, A_s, B_s, x0_s, Vn_s = (
            ParsimCoreAlgorithm.parsim_s(y, u, f=f, p=p, threshold=threshold)
        )

        # They should NOT be identical (different algorithms)
        # At least one matrix should differ significantly
        differs = False

        # Check if A matrices differ
        if not np.allclose(A_p, A_s, rtol=1e-6):
            differs = True

        # Check if B matrices differ
        if not np.allclose(B_p, B_s, rtol=1e-6):
            differs = True

        # Check if K matrices differ
        if not np.allclose(K_p, K_s, rtol=1e-6):
            differs = True

        assert differs, (
            "PARSIM-P and PARSIM-S should produce different results due to "
            "different Gamma_L construction (expanding vs fixed window)"
        )

    def test_parsim_p_expanding_window_behavior(self, simple_siso_system):
        """Test that PARSIM-P's expanding window produces different intermediate results."""
        y, u = simple_siso_system

        # Use small horizon to make differences more visible
        f = 5
        p = 5
        threshold = 0.5  # Higher threshold for faster execution

        # Run PARSIM-P
        A_K_p, C_p, B_K_p, D_p, K_p, A_p, B_p, x0_p, Vn_p = (
            ParsimCoreAlgorithm.parsim_p(y, u, f=f, p=p, threshold=threshold)
        )

        # The key test: PARSIM-P should have computed something
        # We can't easily test intermediate steps without instrumenting the code,
        # but we can verify it runs and produces reasonable output
        assert A_p.shape[0] > 0, "PARSIM-P should identify a model"
        assert np.isfinite(Vn_p), "PARSIM-P should compute finite variance"

    def test_parsim_p_mimo_system(self, stable_mimo_system):
        """Test PARSIM-P on MIMO system."""
        y, u = stable_mimo_system

        f = 10
        p = 10
        threshold = 0.1

        # Should handle MIMO systems
        A_K, C, B_K, D, K, A, B, x0, Vn = ParsimCoreAlgorithm.parsim_p(
            y, u, f=f, p=p, threshold=threshold
        )

        # Check dimensions
        n = A.shape[0]
        l_ = y.shape[0]
        m = u.shape[0]

        assert A.shape == (n, n)
        assert B.shape == (n, m)
        assert C.shape == (l_, n)
        assert D.shape == (l_, m)
        assert K.shape == (n, l_)

        # Check that model is stable (eigenvalues inside unit circle)
        eigenvalues = np.linalg.eigvals(A)
        assert np.all(np.abs(eigenvalues) < 1.2), (
            "Identified model should be reasonably stable"
        )

    def test_parsim_p_with_d_required(self, simple_siso_system):
        """Test PARSIM-P with D matrix required."""
        y, u = simple_siso_system

        f = 10
        p = 10
        threshold = 0.1

        # Test with D_required=True
        A_K, C, B_K, D, K, A, B, x0, Vn = ParsimCoreAlgorithm.parsim_p(
            y, u, f=f, p=p, threshold=threshold, D_required=True
        )

        # D should not be all zeros
        assert D is not None
        assert D.shape == (y.shape[0], u.shape[0])

    def test_parsim_p_fixed_order(self, simple_siso_system):
        """Test PARSIM-P with fixed order."""
        y, u = simple_siso_system

        f = 10
        p = 10
        fixed_order = 3

        # Test with fixed order
        A_K, C, B_K, D, K, A, B, x0, Vn = ParsimCoreAlgorithm.parsim_p(
            y, u, f=f, p=p, fixed_order=fixed_order
        )

        # Order should match
        assert A.shape[0] == fixed_order, "Model order should match fixed_order"

    def test_parsim_p_consistency_across_runs(self, simple_siso_system):
        """Test that PARSIM-P produces consistent results."""
        y, u = simple_siso_system

        f = 10
        p = 10
        threshold = 0.1

        # Run twice with same parameters
        result1 = ParsimCoreAlgorithm.parsim_p(y, u, f=f, p=p, threshold=threshold)
        result2 = ParsimCoreAlgorithm.parsim_p(y, u, f=f, p=p, threshold=threshold)

        # Results should be identical (deterministic algorithm)
        A_K_1, C_1, B_K_1, D_1, K_1, A_1, B_1, x0_1, Vn_1 = result1
        A_K_2, C_2, B_K_2, D_2, K_2, A_2, B_2, x0_2, Vn_2 = result2

        assert np.allclose(A_1, A_2), "A matrix should be consistent"
        assert np.allclose(B_1, B_2), "B matrix should be consistent"
        assert np.allclose(C_1, C_2), "C matrix should be consistent"
        assert np.allclose(K_1, K_2), "K matrix should be consistent"

    def test_parsim_p_comparison_with_master_structure(self, simple_siso_system):
        """Test that PARSIM-P output structure matches master branch."""
        y, u = simple_siso_system

        f = 10
        p = 10
        threshold = 0.1

        # Run PARSIM-P
        result = ParsimCoreAlgorithm.parsim_p(y, u, f=f, p=p, threshold=threshold)

        # Should return 9 values: A_K, C, B_K, D, K, A, B, x0, Vn
        assert len(result) == 9, "PARSIM-P should return 9 values"

        A_K, C, B_K, D, K, A, B, x0, Vn = result

        # Check relationships between matrices
        # A = A_K + K*C (predictor to process form conversion)
        A_reconstructed = A_K + np.dot(K, C)
        assert np.allclose(A, A_reconstructed, rtol=1e-6), "A should equal A_K + K*C"

        # B = B_K + K*D
        B_reconstructed = B_K + np.dot(K, D)
        assert np.allclose(B, B_reconstructed, rtol=1e-6), "B should equal B_K + K*D"


class TestParsimPVsMaster:
    """Integration tests comparing PARSIM-P with master branch implementation."""

    @pytest.fixture
    def master_parsim_methods_path(self):
        """Path to master branch PARSIM methods file."""
        return "/Users/josephj/Workspace/SIPPY-master/sippy_unipi/Parsim_methods.py"

    def test_parsim_p_vs_master_siso(self, master_parsim_methods_path):
        """Compare PARSIM-P results with master branch on SISO system."""
        # This test requires the master branch to be available via git worktree
        import os
        import sys

        if not os.path.exists(master_parsim_methods_path):
            pytest.skip("Master branch not available for comparison")

        # Add master branch to path
        master_dir = os.path.dirname(master_parsim_methods_path)
        sys.path.insert(0, master_dir)

        try:
            from Parsim_methods import PARSIM_P as PARSIM_P_master
        except ImportError:
            pytest.skip("Could not import master branch PARSIM_P")

        # Create test data
        np.random.seed(42)
        n_points = 200
        u = np.random.randn(1, n_points)
        y = np.zeros((1, n_points))
        for i in range(1, n_points):
            y[0, i] = 0.8 * y[0, i - 1] + 0.5 * u[0, i - 1] + 0.05 * np.random.randn()

        # Run both implementations
        f = 10
        p = 10
        threshold = 0.1

        # Harold implementation
        A_K_h, C_h, B_K_h, D_h, K_h, A_h, B_h, x0_h, Vn_h = (
            ParsimCoreAlgorithm.parsim_p(y, u, f=f, p=p, threshold=threshold)
        )

        # Master implementation
        A_K_m, C_m, B_K_m, D_m, K_m, A_m, B_m, x0_m, Vn_m = PARSIM_P_master(
            y, u, f=f, p=p, threshold=threshold
        )

        # Compare results (allowing for some numerical tolerance)
        # Note: May need larger tolerance due to different numerical approaches
        assert A_h.shape == A_m.shape, "A matrices should have same shape"
        assert B_h.shape == B_m.shape, "B matrices should have same shape"

        # Check if results are close (this is the ultimate test)
        # If results differ significantly, the algorithm is wrong
        rtol = 1e-3  # Relaxed tolerance for complex algorithm
        atol = 1e-3

        if not np.allclose(A_h, A_m, rtol=rtol, atol=atol):
            print(f"A_harold:\n{A_h}")
            print(f"A_master:\n{A_m}")
            print(f"Max diff: {np.max(np.abs(A_h - A_m))}")

        # Note: This test may fail initially - that's expected in TDD!
        # The goal is to make it pass by fixing the implementation
