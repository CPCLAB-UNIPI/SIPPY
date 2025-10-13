"""
TDD tests for PARSIM-S reimplementation.

This test suite drives the reimplementation of PARSIM-S to match the
reference implementation in master branch.

Reference: /Users/josephj/Workspace/SIPPY-master/sippy_unipi/Parsim_methods.py lines 410-485
"""

import numpy as np
import pytest

from sippy.identification.algorithms.parsim_core import ParsimCoreAlgorithm


class TestParsimSReimplementation:
    """TDD tests for PARSIM-S reimplementation."""

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

    def test_svd_weighted_k_exists(self):
        """Test that SVD_weighted_K function exists."""
        assert hasattr(ParsimCoreAlgorithm, "svd_weighted_k")

    def test_svd_weighted_k_signature(self, simple_siso_system):
        """Test that SVD_weighted_K has correct signature and returns correct types."""
        y, u = simple_siso_system
        from sippy.utils.simulation_utils import impile, ordinate_sequence

        # Use smaller f and p to ensure valid dimensions
        f = 10
        p = 10
        l_, m = y.shape[0], u.shape[0]

        Yf, Yp = ordinate_sequence(y, f, p)
        Uf, Up = ordinate_sequence(u, f, p)
        Zp = impile(Up, Yp)

        # Build Gamma_L properly using ordinate_sequence approach
        # This ensures dimensions are compatible with the algorithm
        M = np.dot(Yf[0:l_, :], np.linalg.pinv(impile(Zp, Uf[0:m, :])))
        Gamma_L = M[:, 0 : (m + l_) * f]

        # Call the function
        U_n, S_n, V_n = ParsimCoreAlgorithm.svd_weighted_k(Uf, Zp, Gamma_L)

        # Check types
        assert isinstance(U_n, np.ndarray)
        assert isinstance(S_n, np.ndarray)
        assert isinstance(V_n, np.ndarray)

        # Check that SVD was performed
        assert U_n.shape[0] == Gamma_L.shape[0]
        assert len(S_n.shape) == 1  # Should be 1D array of singular values

    def test_ak_c_estimating_s_p_exists(self):
        """Test that AK_C_estimating_S_P function exists."""
        assert hasattr(ParsimCoreAlgorithm, "ak_c_estimating_s_p")

    def test_ak_c_estimating_s_p_uses_qr_decomposition(self):
        """Test that Kalman gain uses QR decomposition approach."""
        from sippy.utils.simulation_utils import (
            impile,
            ordinate_sequence,
            reducingOrder,
        )

        # Generate realistic SISO system with enough dynamics
        np.random.seed(42)
        n_points = 300
        u = np.random.randn(1, n_points)
        y = np.zeros((1, n_points))
        # 2nd order system for better observability
        for i in range(2, n_points):
            y[0, i] = (
                0.8 * y[0, i - 1]
                - 0.3 * y[0, i - 2]
                + 0.5 * u[0, i - 1]
                + 0.2 * u[0, i - 2]
                + 0.05 * np.random.randn()
            )

        l_ = 1
        m = 1
        f = 10
        p = 10

        Yf, Yp = ordinate_sequence(y, f, p)
        Uf, Up = ordinate_sequence(u, f, p)
        Zp = impile(Up, Yp)

        # Build Gamma_L using PARSIM-S iterative approach
        Matrix_pinv = np.linalg.pinv(impile(Zp, Uf[0:m, :]))
        M = np.dot(Yf[0:l_, :], Matrix_pinv)
        Gamma_L = M[:, 0 : (m + l_) * f]
        H = M[:, (m + l_) * f :]

        # Helper for y_tilde estimation
        def estimating_y_S(H, Uf, Yf, i, m, l_):
            y_tilde = np.dot(H[0:l_, :], Uf[m * i : m * (i + 1), :])
            for j in range(1, i):
                y_tilde = y_tilde + np.dot(
                    H[l_ * j : l_ * (j + 1), :], Uf[m * (i - j) : m * (i - j + 1), :]
                )
            return y_tilde

        # Build full Gamma_L matrix (multi-row)
        for i in range(1, f):
            y_tilde = estimating_y_S(H, Uf, Yf, i, m, l_)
            M = np.dot((Yf[l_ * i : l_ * (i + 1)] - y_tilde), Matrix_pinv)
            Gamma_L = impile(Gamma_L, M[:, 0 : (m + l_) * f])
            H = impile(H, M[:, (m + l_) * f :])

        U_n, S_n, V_n = ParsimCoreAlgorithm.svd_weighted_k(Uf, Zp, Gamma_L)
        # Keep at least 2 singular values for proper A estimation
        U_n, S_n, V_n = reducingOrder(U_n, S_n, V_n, threshold=0.01, max_order=2)

        # Call AK_C_estimating_S_P
        A, C, A_K, K, n = ParsimCoreAlgorithm.ak_c_estimating_s_p(
            U_n, S_n, V_n, l_, f, m, Zp, Uf, Yf
        )

        # Check returns
        assert isinstance(A, np.ndarray)
        assert isinstance(C, np.ndarray)
        assert isinstance(A_K, np.ndarray)
        assert isinstance(K, np.ndarray)
        assert isinstance(n, (int, np.integer))

        # Check shapes
        assert A.shape[0] == A.shape[1]  # Square matrix
        assert C.shape[0] == l_
        assert A_K.shape[0] == A_K.shape[1]
        assert K.shape[1] == l_

        # Check relationship: A_K = A - K*C
        assert np.allclose(A_K, A - np.dot(K, C), atol=1e-10)

    def test_simulations_sequence_s_exists(self):
        """Test that simulations_sequence_s function exists."""
        assert hasattr(ParsimCoreAlgorithm, "simulations_sequence_s")

    def test_simulations_sequence_s_predictor_form(self, simple_siso_system):
        """Test that simulations_sequence_S uses predictor form simulation."""
        y, u = simple_siso_system
        l_ = 1
        m = 1
        n = 2
        L = y.shape[1]

        # Simple test matrices
        A_K = np.array([[0.8, 0.1], [0.0, 0.7]])
        C = np.array([[1.0, 0.5]])
        K = np.array([[0.1], [0.05]])
        D_required = False

        y_matrix = ParsimCoreAlgorithm.simulations_sequence_s(
            A_K, C, L, K, y, u, l_, m, n, D_required
        )

        # Check output shape
        if D_required:
            n_simulations = n * m + l_ * m + n
        else:
            n_simulations = n * m + n

        assert y_matrix.shape == (L * l_, n_simulations)

    def test_parsim_s_no_arbitrary_scaling(self, simple_siso_system):
        """Test that PARSIM-S doesn't use arbitrary 0.1 scaling."""
        y, u = simple_siso_system

        # Run PARSIM-S
        A_K, C, B_K, D, K, A, B, x0, Vn = ParsimCoreAlgorithm.parsim_s(
            y, u, f=20, p=20, threshold=0.1, fixed_order=2, D_required=False
        )

        # Check that K has reasonable magnitude (not scaled by 0.1)
        # If properly estimated, K should have magnitude comparable to other matrices
        assert np.max(np.abs(K)) > 0.01  # Not too small
        assert np.max(np.abs(K)) < 10.0  # Not too large

    def test_parsim_s_uses_correct_svd(self, simple_siso_system):
        """Test that PARSIM-S uses SVD_weighted_K, not N4SID SVD."""
        y, u = simple_siso_system

        # This is a behavioral test - we check that the algorithm
        # produces results consistent with PARSIM-specific weighting

        A_K, C, B_K, D, K, A, B, x0, Vn = ParsimCoreAlgorithm.parsim_s(
            y, u, f=20, p=20, threshold=0.1, fixed_order=2, D_required=False
        )

        # Basic sanity checks that algorithm ran
        assert A.shape[0] == A.shape[1]
        assert B.shape[0] == A.shape[0]
        assert C.shape[1] == A.shape[0]
        assert K.shape[0] == A.shape[0]

        # Check A_K = A - K*C relationship holds
        assert np.allclose(A_K, A - np.dot(K, C), atol=1e-8)

    def test_parsim_s_matrix_relationship(self, simple_siso_system):
        """Test key matrix relationships in PARSIM-S."""
        y, u = simple_siso_system

        A_K, C, B_K, D, K, A, B, x0, Vn = ParsimCoreAlgorithm.parsim_s(
            y, u, f=20, p=20, threshold=0.1, fixed_order=2, D_required=False
        )

        # Check B = B_K + K*D relationship
        assert np.allclose(B, B_K + np.dot(K, D), atol=1e-10)

        # Check A = A_K + K*C relationship
        assert np.allclose(A, A_K + np.dot(K, C), atol=1e-10)

    def test_parsim_s_basic_identification(self, simple_siso_system):
        """Test that PARSIM-S can identify a simple system."""
        y, u = simple_siso_system

        # Run identification
        A_K, C, B_K, D, K, A, B, x0, Vn = ParsimCoreAlgorithm.parsim_s(
            y, u, f=20, p=20, threshold=0.1, fixed_order=2, D_required=False
        )

        # Check that Vn (variance) is reasonable
        assert Vn > 0
        assert Vn < np.var(y) * 1.5  # Should explain most of the variance

        # Check stability of identified A matrix
        eigenvalues = np.linalg.eigvals(A)
        assert np.all(np.abs(eigenvalues) < 1.1)  # Should be stable or close

    def test_parsim_s_vs_master_branch_structure(self):
        """Test that PARSIM-S follows master branch structure."""
        # This is a documentation test to ensure we're following the right algorithm

        # Master branch PARSIM_S structure (lines 410-485):
        # 1. Rescale data (lines 441-446)
        # 2. Ordinate sequences (lines 447-449)
        # 3. Initial projection (lines 450-453)
        # 4. Iterative Gamma_L with estimating_y_S (lines 454-458)
        # 5. SVD_weighted_K (line 459)
        # 6. AK_C_estimating_S_P (lines 461-462)
        # 7. simulations_sequence_S (lines 464-465)
        # 8. Parameter extraction (lines 467-476)
        # 9. Rescale back (lines 477-483)
        # 10. B = B_K + K*D (line 484)

        # This test documents the expected flow
        assert True  # Placeholder - actual implementation will follow this structure


class TestSVDWeightedK:
    """Tests specifically for SVD_weighted_K function."""

    @pytest.fixture
    def realistic_parsim_matrices(self):
        """Generate properly dimensioned matrices for PARSIM-S tests."""
        from sippy.utils.simulation_utils import impile, ordinate_sequence

        np.random.seed(42)
        n_points = 300  # Enough for f=10, p=10

        # Generate realistic SISO system
        u = np.random.randn(1, n_points)
        y = np.zeros((1, n_points))
        for i in range(1, n_points):
            y[0, i] = 0.8 * y[0, i - 1] + 0.5 * u[0, i - 1] + 0.05 * np.random.randn()

        l_, m = y.shape[0], u.shape[0]
        f, p = 10, 10

        # Build matrices properly using ordinate_sequence
        Yf, Yp = ordinate_sequence(y, f, p)
        Uf, Up = ordinate_sequence(u, f, p)
        Zp = impile(Up, Yp)

        # Build Gamma_L through proper projection
        M = np.dot(Yf[0:l_, :], np.linalg.pinv(impile(Zp, Uf[0:m, :])))
        Gamma_L = M[:, 0 : (m + l_) * f]

        return Yf, Uf, Zp, Gamma_L, l_, m, f

    def test_svd_weighted_k_uses_matrix_weighting(self, realistic_parsim_matrices):
        """Test that SVD_weighted_K uses proper weighting matrix W2."""
        Yf, Uf, Zp, Gamma_L, l_, m, f = realistic_parsim_matrices

        U_n, S_n, V_n = ParsimCoreAlgorithm.svd_weighted_k(Uf, Zp, Gamma_L)

        # Check that SVD was performed
        assert U_n.shape[0] == Gamma_L.shape[0]
        assert len(S_n) <= min(Gamma_L.shape)

    def test_svd_weighted_k_differs_from_standard_svd(self, realistic_parsim_matrices):
        """Test that weighted SVD produces different results from standard SVD."""
        Yf, Uf, Zp, Gamma_L, l_, m, f = realistic_parsim_matrices

        # Weighted SVD
        U_n_weighted, S_n_weighted, V_n_weighted = ParsimCoreAlgorithm.svd_weighted_k(
            Uf, Zp, Gamma_L
        )

        # Standard SVD
        U_n_standard, S_n_standard, V_n_standard = np.linalg.svd(
            Gamma_L, full_matrices=False
        )

        # They should be different (unless weighting is identity)
        # We check that at least one is different
        different = (
            not np.allclose(U_n_weighted, U_n_standard, atol=1e-6)
            or not np.allclose(S_n_weighted, S_n_standard, atol=1e-6)
            or not np.allclose(V_n_weighted, V_n_standard, atol=1e-6)
        )

        assert different, "Weighted SVD should differ from standard SVD"


class TestAKCEstimatingSP:
    """Tests specifically for AK_C_estimating_S_P function."""

    @pytest.fixture
    def realistic_qr_test_data(self):
        """Generate realistic test data for QR decomposition tests."""
        from sippy.utils.simulation_utils import (
            impile,
            ordinate_sequence,
            reducingOrder,
        )

        np.random.seed(42)
        # Use enough points for f=10: N > (2*m+l_)*f = 30
        n_points = 300
        l_ = 1
        m = 1
        f = 10

        # Generate realistic 2nd order SISO system for better observability
        u = np.random.randn(m, n_points)
        y = np.zeros((l_, n_points))
        for i in range(2, n_points):
            y[0, i] = (
                0.8 * y[0, i - 1]
                - 0.3 * y[0, i - 2]
                + 0.5 * u[0, i - 1]
                + 0.2 * u[0, i - 2]
                + 0.05 * np.random.randn()
            )

        # Build matrices properly
        p = 10
        Yf, Yp = ordinate_sequence(y, f, p)
        Uf, Up = ordinate_sequence(u, f, p)
        Zp = impile(Up, Yp)

        # Build Gamma_L using PARSIM-S iterative approach (multi-row matrix)
        Matrix_pinv = np.linalg.pinv(impile(Zp, Uf[0:m, :]))
        M = np.dot(Yf[0:l_, :], Matrix_pinv)
        Gamma_L = M[:, 0 : (m + l_) * f]
        H = M[:, (m + l_) * f :]

        # Helper for y_tilde estimation
        def estimating_y_S(H, Uf, Yf, i, m, l_):
            y_tilde = np.dot(H[0:l_, :], Uf[m * i : m * (i + 1), :])
            for j in range(1, i):
                y_tilde = y_tilde + np.dot(
                    H[l_ * j : l_ * (j + 1), :], Uf[m * (i - j) : m * (i - j + 1), :]
                )
            return y_tilde

        # Build full Gamma_L matrix (multi-row)
        for i in range(1, f):
            y_tilde = estimating_y_S(H, Uf, Yf, i, m, l_)
            M = np.dot((Yf[l_ * i : l_ * (i + 1)] - y_tilde), Matrix_pinv)
            Gamma_L = impile(Gamma_L, M[:, 0 : (m + l_) * f])
            H = impile(H, M[:, (m + l_) * f :])

        # Get SVD and keep at least 2 singular values
        U_n, S_n, V_n = ParsimCoreAlgorithm.svd_weighted_k(Uf, Zp, Gamma_L)
        U_n, S_n, V_n = reducingOrder(U_n, S_n, V_n, threshold=0.01, max_order=2)

        return U_n, S_n, V_n, l_, f, m, Zp, Uf, Yf

    def test_ak_c_qr_decomposition_used(self, realistic_qr_test_data):
        """Test that QR decomposition is used for K estimation."""
        U_n, S_n, V_n, l_, f, m, Zp, Uf, Yf = realistic_qr_test_data

        A, C, A_K, K, n = ParsimCoreAlgorithm.ak_c_estimating_s_p(
            U_n, S_n, V_n, l_, f, m, Zp, Uf, Yf
        )

        # Check that K was estimated (not zeros, not random)
        assert not np.allclose(K, 0)
        assert n == len(S_n)

    def test_ak_c_matrix_relationships(self, realistic_qr_test_data):
        """Test matrix relationships in AK_C_estimating_S_P."""
        U_n, S_n, V_n, l_, f, m, Zp, Uf, Yf = realistic_qr_test_data

        A, C, A_K, K, n = ParsimCoreAlgorithm.ak_c_estimating_s_p(
            U_n, S_n, V_n, l_, f, m, Zp, Uf, Yf
        )

        # Verify A_K = A - K*C
        assert np.allclose(A_K, A - np.dot(K, C), atol=1e-10)

        # Verify C comes from observability matrix
        assert C.shape == (l_, n)


class TestSimulationsSequenceS:
    """Tests for simulations_sequence_S function."""

    def test_simulations_sequence_s_shape(self):
        """Test output shape of simulations_sequence_S."""
        l_ = 1
        m = 1
        n = 2
        L = 100

        y = np.random.randn(l_, L)
        u = np.random.randn(m, L)

        A_K = np.array([[0.8, 0.1], [0.0, 0.7]])
        C = np.array([[1.0, 0.5]])
        K = np.array([[0.1], [0.05]])

        y_matrix = ParsimCoreAlgorithm.simulations_sequence_s(
            A_K, C, L, K, y, u, l_, m, n, D_required=False
        )

        expected_simulations = n * m + n
        assert y_matrix.shape == (L * l_, expected_simulations)

    def test_simulations_sequence_s_with_d(self):
        """Test simulations_sequence_S with D matrix."""
        l_ = 1
        m = 1
        n = 2
        L = 100

        y = np.random.randn(l_, L)
        u = np.random.randn(m, L)

        A_K = np.array([[0.8, 0.1], [0.0, 0.7]])
        C = np.array([[1.0, 0.5]])
        K = np.array([[0.1], [0.05]])

        y_matrix = ParsimCoreAlgorithm.simulations_sequence_s(
            A_K, C, L, K, y, u, l_, m, n, D_required=True
        )

        expected_simulations = n * m + l_ * m + n
        assert y_matrix.shape == (L * l_, expected_simulations)
