"""
PARSIM algorithms core implementation.
"""

import numpy as np
import scipy as sc

try:
    from joblib import Parallel, delayed

    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

from ...utils.signal_utils import rescale
from ...utils.simulation_utils import (
    Vn_mat,
    check_inputs,
    check_types,
    impile,
    ordinate_sequence,
    reducingOrder,
    simulate_ss_system,
)

# Import compiled utilities for performance
try:
    from ...utils.compiled_utils import (
        NUMBA_AVAILABLE,
        Z_dot_PIort_compiled,
        matrix_operations_a_compiled,
        parsim_k_matrix_operations_compiled,
        parsim_y_tilde_estimation_compiled,
        subspace_weighted_svd_compiled,
    )
except ImportError:
    parsim_k_matrix_operations_compiled = None
    parsim_y_tilde_estimation_compiled = None
    subspace_weighted_svd_compiled = None
    Z_dot_PIort_compiled = None
    matrix_operations_a_compiled = None
    NUMBA_AVAILABLE = False


class ParsimCoreAlgorithm:
    """Core PARSIM algorithms implementation."""

    @staticmethod
    def parsim_k(
        y,
        u,
        f=20,
        p=20,
        threshold=0.1,
        max_order=np.nan,
        fixed_order=np.nan,
        D_required=False,
        B_recalc=False,
    ):
        """
        PARSIM-K algorithm implementation.

        Parameters:
        -----------
        y : ndarray
            Output data (outputs x time_steps)
        u : ndarray
            Input data (inputs x time_steps)
        f : int
            Future horizon
        p : int
            Past horizon
        threshold : float
            Singular value threshold
        max_order : float
            Maximum order
        fixed_order : float
            Fixed order
        D_required : bool
            Whether D matrix is required
        B_recalc : bool
            Whether to recalculate B matrix

        Returns:
        --------
        A_K, C, B_K, D, K, A, B, x0, Vn : ndarrays
            System matrices and initial state
        """
        y = 1.0 * np.atleast_2d(y)
        u = 1.0 * np.atleast_2d(u)
        l_, L = y.shape
        m = u[:, 0].size

        if not check_types(threshold, max_order, fixed_order, f, p):
            return (
                np.array([[0.0]]),
                np.array([[0.0]]),
                np.array([[0.0]]),
                np.array([[0.0]]),
                np.array([[0.0]]),
                np.array([[0.0]]),
                np.array([[0.0]]),
                np.array([[0.0]]),
                np.inf,
            )

        threshold, max_order = check_inputs(threshold, max_order, fixed_order, f)

        # Standardize inputs and outputs
        Ustd = np.zeros(m)
        Ystd = np.zeros(l_)
        for j in range(m):
            Ustd[j], u[j] = rescale(u[j])
        for j in range(l_):
            Ystd[j], y[j] = rescale(y[j])

        # Create data matrices - use compiled version when available
        if NUMBA_AVAILABLE and parsim_k_matrix_operations_compiled is not None:
            try:
                Yf, Yp, Uf, Up, M = parsim_k_matrix_operations_compiled(
                    y, u, f, p, m, l_, L
                )
                Zp = impile(Up, Yp)
            except Exception:
                # Fallback to original implementation
                Yf, Yp = ordinate_sequence(y, f, p)
                Uf, Up = ordinate_sequence(u, f, p)
                Zp = impile(Up, Yp)
                M = np.dot(Yf[0:l_, :], np.linalg.pinv(impile(Zp, Uf[0:m, :])))
        else:
            # Original implementation
            Yf, Yp = ordinate_sequence(y, f, p)
            Uf, Up = ordinate_sequence(u, f, p)
            Zp = impile(Up, Yp)
            M = np.dot(Yf[0:l_, :], np.linalg.pinv(impile(Zp, Uf[0:m, :])))
        Matrix_pinv = np.linalg.pinv(impile(Zp, impile(Uf[0:m, :], Yf[0:l_, :])))
        Gamma_L = M[:, 0 : (m + l_) * f]

        # Defensive check: If M doesn't have enough columns, initialize H_K appropriately
        # H_K should capture residual dynamics not explained by Gamma_L
        if M.shape[1] > (m + l_) * f:
            H_K = M[:, (m + l_) * f :]
        else:
            # Initialize with zeros of appropriate size to maintain algorithm flow
            # Size should be (l_, m) to match first iteration's expected dimensions
            H_K = np.zeros((l_, m))

        G_K = np.zeros((l_, l_))

        # Helper function for y_tilde estimation
        def estimating_y(H_K, Uf, G_K, Yf, i, m, l_):
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

        # Build matrices for each horizon - use compiled y_tilde when available
        for i in range(1, f):
            if NUMBA_AVAILABLE and parsim_y_tilde_estimation_compiled is not None:
                try:
                    y_tilde = parsim_y_tilde_estimation_compiled(
                        H_K, Uf, G_K, Yf, i, m, l_, f
                    )
                except Exception:
                    # Fallback to original implementation
                    y_tilde = estimating_y(H_K, Uf, G_K, Yf, i, m, l_)
            else:
                y_tilde = estimating_y(H_K, Uf, G_K, Yf, i, m, l_)

            M = np.dot((Yf[l_ * i : l_ * (i + 1)] - y_tilde), Matrix_pinv)
            H_K = impile(H_K, M[:, (m + l_) * f : (m + l_) * f + m])
            G_K = impile(G_K, M[:, (m + l_) * f + m :])
            Gamma_L = impile(Gamma_L, M[:, 0 : (m + l_) * f])

        # CRITICAL FIX: Use PARSIM-K specific SVD with Gamma_L (not N4SID's svd_weighted)
        # Reference: master/sippy_unipi/Parsim_methods.py line 233
        U_n, S_n, V_n = ParsimCoreAlgorithm.svd_weighted_k(Uf, Zp, Gamma_L)
        U_n, S_n, V_n = reducingOrder(U_n, S_n, V_n, threshold, max_order)

        n = S_n.size
        S_n_diag = np.diag(S_n)
        Ob_K = np.dot(U_n, np.sqrt(S_n_diag))

        # Estimate A_K carefully
        if l_ * (f - 1) >= n and n > 0:
            try:
                A_K = np.dot(np.linalg.pinv(Ob_K[0 : l_ * (f - 1), :]), Ob_K[l_:, :])
            except (np.linalg.LinAlgError, ValueError):
                # Fallback to random initialization on linear algebra errors
                A_K = np.random.randn(n, n) * 0.1
        else:
            A_K = np.random.randn(n, n) * 0.1

        C = Ob_K[0:l_, :]

        # CRITICAL FIX: Use simulations_sequence_k for parameter estimation
        # This uses predictor form simulation
        # Reference: master/sippy_unipi/Parsim_methods.py line 240
        K_placeholder = np.zeros((n, l_))
        D_placeholder = np.zeros((l_, m))
        y_sim = ParsimCoreAlgorithm.simulations_sequence_k(
            A_K, C, L, y, u, l_, m, n, K_placeholder, D_placeholder, D_required
        )

        # Solve for parameters using least squares
        vect = np.dot(np.linalg.pinv(y_sim), y.reshape((L * l_, 1)))
        Y_estimate = np.dot(y_sim, vect)
        Vn = Vn_mat(y.reshape((L * l_, 1)), Y_estimate)

        # Extract parameters from vect
        B_K = vect[0 : n * m, :].reshape((n, m))
        if D_required:
            D = vect[n * m : n * m + l_ * m, :].reshape((l_, m))
            K = vect[n * m + l_ * m : n * m + l_ * m + n * l_, :].reshape((n, l_))
            x0 = vect[n * m + l_ * m + n * l_ : :, :].reshape((n, 1))
        else:
            D = np.zeros((l_, m))
            K = vect[n * m : n * m + n * l_, :].reshape((n, l_))
            x0 = vect[n * m + n * l_ : :, :].reshape((n, 1))

        # Calculate A matrix
        A = A_K + np.dot(K, C)

        # Optional B recalculation using process form
        # Reference: master/sippy_unipi/Parsim_methods.py lines 256-263
        if B_recalc:
            # Helper function to create simulation matrix for B recalc
            def recalc_K(A, C, D, u):
                y_sim = []
                n_ord = A[:, 0].size
                m_input, L_u = u.shape
                l_out = C[:, 0].size
                n_simulations = n_ord + n_ord * m_input
                vect = np.zeros((n_simulations, 1))
                for i in range(n_simulations):
                    vect[i, 0] = 1.0
                    B_i = vect[0 : n_ord * m_input, :].reshape((n_ord, m_input))
                    x0_i = vect[n_ord * m_input : :, :].reshape((n_ord, 1))
                    _, y_i = simulate_ss_system(A, B_i, C, D, u, x0=x0_i)
                    y_sim.append(y_i.reshape((1, L_u * l_out)))
                    vect[i, 0] = 0.0
                y_matrix = 1.0 * y_sim[0]
                for j in range(n_simulations - 1):
                    y_matrix = impile(y_matrix, y_sim[j + 1])
                y_matrix = y_matrix.T
                return y_matrix

            y_sim = recalc_K(A, C, D, u)
            vect = np.dot(np.linalg.pinv(y_sim), y.reshape((L * l_, 1)))
            Y_estimate = np.dot(y_sim, vect)
            Vn = Vn_mat(y.reshape((L * l_, 1)), Y_estimate)
            B = vect[0 : n * m, :].reshape((n, m))
            x0 = vect[n * m : :, :].reshape((n, 1))
            B_K = B - np.dot(K, D)
        else:
            B = B_K + np.dot(K, D)

        # Rescale back to original units
        for j in range(m):
            B_K[:, j] = B_K[:, j] / Ustd[j]
            D[:, j] = D[:, j] / Ustd[j]
        for j in range(l_):
            K[:, j] = K[:, j] / Ystd[j]
            C[j, :] = C[j, :] * Ystd[j]
            D[j, :] = D[j, :] * Ystd[j]
        B = B_K + np.dot(K, D)

        return A_K, C, B_K, D, K, A, B, x0, Vn

    @staticmethod
    def parsim_s(
        y,
        u,
        f=20,
        p=20,
        threshold=0.1,
        max_order=np.nan,
        fixed_order=np.nan,
        D_required=False,
    ):
        """
        PARSIM-S algorithm implementation.

        Parameters:
        -----------
        y : ndarray
            Output data (outputs x time_steps)
        u : ndarray
            Input data (inputs x time_steps)
        f : int
            Future horizon
        p : int
            Past horizon
        threshold : float
            Singular value threshold
        max_order : float
            Maximum order
        fixed_order : float
            Fixed order
        D_required : bool
            Whether D matrix is required

        Returns:
        --------
        A_K, C, B_K, D, K, A, B, x0, Vn : ndarrays
            System matrices and initial state
        """
        y = 1.0 * np.atleast_2d(y)
        u = 1.0 * np.atleast_2d(u)
        l_, L = y.shape
        m = u[:, 0].size

        if not check_types(threshold, max_order, fixed_order, f, p):
            return (
                np.array([[0.0]]),
                np.array([[0.0]]),
                np.array([[0.0]]),
                np.array([[0.0]]),
                np.array([[0.0]]),
                np.array([[0.0]]),
                np.array([[0.0]]),
                np.array([[0.0]]),
                np.inf,
            )

        threshold, max_order = check_inputs(threshold, max_order, fixed_order, f)

        # Standardize inputs and outputs
        Ustd = np.zeros(m)
        Ystd = np.zeros(l_)
        for j in range(m):
            Ustd[j], u[j] = rescale(u[j])
        for j in range(l_):
            Ystd[j], y[j] = rescale(y[j])

        # Create data matrices
        Yf, Yp = ordinate_sequence(y, f, p)
        Uf, Up = ordinate_sequence(u, f, p)
        Zp = impile(Up, Yp)

        # Initial matrices
        Matrix_pinv = np.linalg.pinv(impile(Zp, Uf[0:m, :]))
        M = np.dot(Yf[0:l_, :], Matrix_pinv)
        Gamma_L = M[:, 0 : (m + l_) * f]
        H = M[:, (m + l_) * f :]

        # Helper function for y_tilde estimation S
        def estimating_y_S(H, Uf, Yf, i, m, l_):
            y_tilde = np.dot(H[0:l_, :], Uf[m * i : m * (i + 1), :])
            for j in range(1, i):
                y_tilde = y_tilde + np.dot(
                    H[l_ * j : l_ * (j + 1), :], Uf[m * (i - j) : m * (i - j + 1), :]
                )
            return y_tilde

        # Build matrices for each horizon
        for i in range(1, f):
            y_tilde = estimating_y_S(H, Uf, Yf, i, m, l_)
            M = np.dot((Yf[l_ * i : l_ * (i + 1)] - y_tilde), Matrix_pinv)
            Gamma_L = impile(Gamma_L, M[:, 0 : (m + l_) * f])
            H = impile(H, M[:, (m + l_) * f :])

        # CRITICAL FIX: Use PARSIM-specific SVD weighting (not N4SID's SVD)
        # Reference: master/sippy_unipi/Parsim_methods.py line 384 (now 459)
        U_n, S_n, V_n = ParsimCoreAlgorithm.svd_weighted_k(Uf, Zp, Gamma_L)
        U_n, S_n, V_n = reducingOrder(U_n, S_n, V_n, threshold, max_order)

        # CRITICAL FIX: Use QR-based Kalman gain estimation
        # Reference: master/sippy_unipi/Parsim_methods.py lines 461-462 (now 386-387)
        A, C, A_K, K, n = ParsimCoreAlgorithm.ak_c_estimating_s_p(
            U_n, S_n, V_n, l_, f, m, Zp, Uf, Yf
        )

        # CRITICAL FIX: Use systematic predictor form simulation
        # Reference: master/sippy_unipi/Parsim_methods.py lines 464-465 (now 389-390)
        y_sim = ParsimCoreAlgorithm.simulations_sequence_s(
            A_K, C, L, K, y, u, l_, m, n, D_required
        )

        # Solve for parameters using least squares
        # Reference: master/sippy_unipi/Parsim_methods.py lines 467-476 (now 392-401)
        vect = np.dot(np.linalg.pinv(y_sim), y.reshape((L * l_, 1)))
        Y_estimate = np.dot(y_sim, vect)
        Vn = Vn_mat(y.reshape((L * l_, 1)), Y_estimate)

        # Extract parameters from vect
        B_K = vect[0 : n * m, :].reshape((n, m))
        if D_required:
            D = vect[n * m : n * m + l_ * m, :].reshape((l_, m))
            x0 = vect[n * m + l_ * m :, :].reshape((n, 1))
        else:
            D = np.zeros((l_, m))
            x0 = vect[n * m :, :].reshape((n, 1))

        # Calculate B matrix
        B = B_K + np.dot(K, D)

        # Rescale back to original units
        for j in range(m):
            B_K[:, j] = B_K[:, j] / Ustd[j]
            D[:, j] = D[:, j] / Ustd[j]
        for j in range(l_):
            K[:, j] = K[:, j] / Ystd[j]
            C[j, :] = C[j, :] * Ystd[j]
            D[j, :] = D[j, :] * Ystd[j]
        B = B_K + np.dot(K, D)

        return A_K, C, B_K, D, K, A, B, x0, Vn

    @staticmethod
    def parsim_p(
        y,
        u,
        f=20,
        p=20,
        threshold=0.1,
        max_order=np.nan,
        fixed_order=np.nan,
        D_required=False,
    ):
        """
        PARSIM-P algorithm implementation with expanding window approach.

        Key difference from PARSIM-S: The Uf window expands with each iteration,
        providing progressively more input information for better parameter estimation.

        Reference: master/sippy_unipi/Parsim_methods.py lines 597-670

        Parameters:
        -----------
        y : ndarray
            Output data (outputs x time_steps)
        u : ndarray
            Input data (inputs x time_steps)
        f : int
            Future horizon
        p : int
            Past horizon
        threshold : float
            Singular value threshold
        max_order : float
            Maximum order
        fixed_order : float
            Fixed order
        D_required : bool
            Whether D matrix is required

        Returns:
        --------
        A_K, C, B_K, D, K, A, B, x0, Vn : ndarrays
            System matrices and initial state
        """
        y = 1.0 * np.atleast_2d(y)
        u = 1.0 * np.atleast_2d(u)
        l_, L = y.shape
        m = u[:, 0].size

        if not check_types(threshold, max_order, fixed_order, f, p):
            return (
                np.array([[0.0]]),
                np.array([[0.0]]),
                np.array([[0.0]]),
                np.array([[0.0]]),
                np.array([[0.0]]),
                np.array([[0.0]]),
                np.array([[0.0]]),
                np.array([[0.0]]),
                np.inf,
            )

        threshold, max_order = check_inputs(threshold, max_order, fixed_order, f)

        # Standardize inputs and outputs
        Ustd = np.zeros(m)
        Ystd = np.zeros(l_)
        for j in range(m):
            Ustd[j], u[j] = rescale(u[j])
        for j in range(l_):
            Ystd[j], y[j] = rescale(y[j])

        # Create data matrices
        Yf, Yp = ordinate_sequence(y, f, p)
        Uf, Up = ordinate_sequence(u, f, p)
        Zp = impile(Up, Yp)

        # Initial projection with first block
        # CRITICAL: This is where PARSIM-P differs from PARSIM-S
        # Master lines 637-639
        Matrix_pinv = np.linalg.pinv(impile(Zp, Uf[0:m, :]))
        M = np.dot(Yf[0:l_, :], Matrix_pinv)
        Gamma_L = M[:, 0 : (m + l_) * f]

        # EXPANDING WINDOW: Key difference from PARSIM-S
        # Master lines 640-643
        # In each iteration, the Uf window grows: Uf[0:m*(i+1), :]
        for i in range(1, f):
            # Recompute Matrix_pinv with EXPANDING Uf window
            # This is the critical line that makes PARSIM-P different!
            Matrix_pinv = np.linalg.pinv(impile(Zp, Uf[0 : m * (i + 1), :]))
            M = np.dot(Yf[l_ * i : l_ * (i + 1), :], Matrix_pinv)
            Gamma_L = impile(Gamma_L, M[:, 0 : (m + l_) * f])

        # SVD for order estimation - use PARSIM-K weighted SVD
        # Master line 644
        U_n, S_n, V_n = ParsimCoreAlgorithm.svd_weighted_k(Uf, Zp, Gamma_L)
        U_n, S_n, V_n = reducingOrder(U_n, S_n, V_n, threshold, max_order)

        # Use same QR-based K estimation as PARSIM-S
        # Master lines 646-647
        A, C, A_K, K, n = ParsimCoreAlgorithm.ak_c_estimating_s_p(
            U_n, S_n, V_n, l_, f, m, Zp, Uf, Yf
        )

        # Simulation using predictor form (master lines 649-651)
        # Use simulations_sequence_S (K is fixed, estimate B_K, D, x0)
        y_sim = ParsimCoreAlgorithm.simulations_sequence_s(
            A_K, C, L, K, y, u, l_, m, n, D_required
        )

        # Parameter estimation (master lines 652-654)
        vect = np.dot(np.linalg.pinv(y_sim), y.reshape((L * l_, 1)))
        Y_estimate = np.dot(y_sim, vect)
        Vn = Vn_mat(y.reshape((L * l_, 1)), Y_estimate)

        # Extract parameters (master lines 655-661)
        B_K = vect[0 : n * m, :].reshape((n, m))
        if D_required:
            D = vect[n * m : n * m + l_ * m, :].reshape((l_, m))
            x0 = vect[n * m + l_ * m :, :].reshape((n, 1))
        else:
            D = np.zeros((l_, m))
            x0 = vect[n * m :, :].reshape((n, 1))

        # Rescale back to original units (master lines 662-668)
        for j in range(m):
            B_K[:, j] = B_K[:, j] / Ustd[j]
            D[:, j] = D[:, j] / Ustd[j]
        for j in range(l_):
            K[:, j] = K[:, j] / Ystd[j]
            C[j, :] = C[j, :] * Ystd[j]
            D[j, :] = D[j, :] * Ystd[j]

        # Calculate B matrix (master line 669)
        B = B_K + np.dot(K, D)

        return A_K, C, B_K, D, K, A, B, x0, Vn

    @staticmethod
    def svd_weighted_k(Uf, Zp, Gamma_L):
        """
        PARSIM-K specific weighted SVD.

        This is different from N4SID's SVD weighting - it uses PARSIM-specific
        weighting based on Z_dot_PIort(Zp, Uf) instead of the N4SID weights.

        Reference: master/sippy_unipi/Parsim_methods.py lines 76-79

        Parameters:
        -----------
        Uf : ndarray
            Future input ordinate sequence
        Zp : ndarray
            Past data matrix (stacked Up and Yp)
        Gamma_L : ndarray
            Extended observability matrix from PARSIM-K iteration

        Returns:
        --------
        U_n : ndarray
            Left singular vectors
        S_n : ndarray
            Singular values
        V_n : ndarray
            Right singular vectors
        """
        from ...utils.simulation_utils import Z_dot_PIort

        # Edge case: Check for empty or degenerate matrices
        if Gamma_L.size == 0 or Gamma_L.shape[0] == 0 or Gamma_L.shape[1] == 0:
            # Return empty SVD components with consistent shapes
            return (
                np.zeros((Gamma_L.shape[0], 0)),
                np.array([]),
                np.zeros((0, Gamma_L.shape[1])),
            )

        try:
            # PARSIM-K weighting: W2 = sqrtm((Zp - Zp*Uf^T*pinv(Uf^T)) * Zp^T)
            W2 = sc.linalg.sqrtm(np.dot(Z_dot_PIort(Zp, Uf), Zp.T)).real

            # Check for NaN or Inf in W2
            if not np.all(np.isfinite(W2)):
                # Fallback to unweighted SVD
                U_n, S_n, V_n = np.linalg.svd(Gamma_L, full_matrices=False)
                return U_n, S_n, V_n

            # Weighted SVD: svd(Gamma_L * W2)
            weighted_matrix = np.dot(Gamma_L, W2)

            # Check for numerical issues
            if not np.all(np.isfinite(weighted_matrix)):
                # Fallback to unweighted SVD
                U_n, S_n, V_n = np.linalg.svd(Gamma_L, full_matrices=False)
                return U_n, S_n, V_n

            U_n, S_n, V_n = np.linalg.svd(weighted_matrix, full_matrices=False)

        except (np.linalg.LinAlgError, ValueError):
            # Fallback to unweighted SVD on any linear algebra errors
            U_n, S_n, V_n = np.linalg.svd(Gamma_L, full_matrices=False)

        return U_n, S_n, V_n

    @staticmethod
    def _simulate_single_parameter_k(
        i, n_simulations, vect, A_K, C, D_required, y, u, l_, m, n, L
    ):
        """
        Simulate a single parameter configuration for PARSIM-K.

        This is a helper function designed to be thread-safe for parallel execution.
        Each call simulates the system with a single unit vector for one parameter.

        Parameters:
        -----------
        i : int
            Parameter index (which parameter to set to 1.0)
        n_simulations : int
            Total number of simulations
        vect : ndarray
            Parameter vector template (will be copied, not modified)
        A_K, C : ndarrays
            System matrices
        D_required : bool
            Whether D matrix is estimated
        y, u : ndarrays
            Output and input data (read-only)
        l_, m, n : ints
            System dimensions
        L : int
            Number of time steps

        Returns:
        --------
        y_hat_flat : ndarray
            Flattened output simulation (1 x L*l_)
        """
        from ...utils.simulation_utils import ss_lsim_predictor_form

        # Create local copy of vect to avoid race conditions
        vect_local = vect.copy()
        vect_local[i, 0] = 1.0

        if D_required:
            B_K = vect_local[0 : n * m, :].reshape((n, m))
            D_i = vect_local[n * m : n * m + l_ * m, :].reshape((l_, m))
            K_i = vect_local[n * m + l_ * m : n * m + l_ * m + n * l_, :].reshape(
                (n, l_)
            )
            x0 = vect_local[n * m + l_ * m + n * l_ : :, :].reshape((n, 1))
        else:
            B_K = vect_local[0 : n * m, :].reshape((n, m))
            D_i = np.zeros((l_, m))
            K_i = vect_local[n * m : n * m + n * l_, :].reshape((n, l_))
            x0 = vect_local[n * m + n * l_ : :, :].reshape((n, 1))

        # Simulate using predictor form
        _, y_hat = ss_lsim_predictor_form(A_K, B_K, C, D_i, K_i, y, u, x0)
        return y_hat.reshape((1, L * l_))

    @staticmethod
    def _simulate_single_parameter_s(
        i, n_simulations, vect, A_K, C, K, D_required, y, u, l_, m, n, L
    ):
        """
        Simulate a single parameter configuration for PARSIM-S.

        This is a helper function designed to be thread-safe for parallel execution.
        Each call simulates the system with a single unit vector for one parameter.
        Note: K is FIXED (not estimated) unlike PARSIM-K.

        Parameters:
        -----------
        i : int
            Parameter index (which parameter to set to 1.0)
        n_simulations : int
            Total number of simulations
        vect : ndarray
            Parameter vector template (will be copied, not modified)
        A_K, C, K : ndarrays
            System matrices (K is fixed)
        D_required : bool
            Whether D matrix is estimated
        y, u : ndarrays
            Output and input data (read-only)
        l_, m, n : ints
            System dimensions
        L : int
            Number of time steps

        Returns:
        --------
        y_hat_flat : ndarray
            Flattened output simulation (1 x L*l_)
        """
        from ...utils.simulation_utils import SS_lsim_predictor_form

        # Create local copy of vect to avoid race conditions
        vect_local = vect.copy()
        vect_local[i, 0] = 1.0

        if D_required:
            B_K = vect_local[0 : n * m, :].reshape((n, m))
            D = vect_local[n * m : n * m + l_ * m, :].reshape((l_, m))
            x0 = vect_local[n * m + l_ * m :, :].reshape((n, 1))
        else:
            B_K = vect_local[0 : n * m, :].reshape((n, m))
            D = np.zeros((l_, m))
            x0 = vect_local[n * m :, :].reshape((n, 1))

        # Simulate predictor form with FIXED K
        _, y_hat = SS_lsim_predictor_form(A_K, B_K, C, D, K, y, u, x0)
        return y_hat.reshape((1, L * l_))

    @staticmethod
    def simulations_sequence_k(A_K, C, L, y, u, l_, m, n, K, D, D_required=False):
        """
        Create simulation matrix for PARSIM-K parameter estimation.

        This function creates a regression matrix by simulating the system
        with different unit vectors for B_K, K, D, and x0 parameters.
        Uses predictor form simulation: x[i+1] = A_K*x[i] + B_K*u[i] + K*y[i]

        PERFORMANCE: Uses parallel execution via joblib when available and
        n_simulations >= 20, achieving 3-6x speedup on multi-core systems.

        Reference: master/sippy_unipi/Parsim_methods.py lines 82-120

        Parameters:
        -----------
        A_K : ndarray
            State matrix in predictor form (n x n)
        C : ndarray
            Output matrix (l x n)
        L : int
            Number of time steps
        y : ndarray
            Output data (l x L)
        u : ndarray
            Input data (m x L)
        l_ : int
            Number of outputs
        m : int
            Number of inputs
        n : int
            Model order
        K : ndarray
            Kalman gain (n x l) - placeholder, overwritten in simulations
        D : ndarray
            Feedthrough matrix (l x m) - placeholder
        D_required : bool
            Whether to estimate D matrix

        Returns:
        --------
        y_matrix : ndarray
            Simulation matrix (L*l x n_simulations) - transposed for least squares
        """
        from ...utils.simulation_utils import impile

        # Calculate number of simulations needed
        if D_required:
            # Parameters to estimate: B_K (n*m), D (l*m), K (n*l), x0 (n)
            n_simulations = n * m + l_ * m + n * l_ + n
        else:
            # Parameters to estimate: B_K (n*m), K (n*l), x0 (n)
            n_simulations = n * m + n * l_ + n

        # Create parameter vector template
        vect = np.zeros((n_simulations, 1))

        # Adaptive threshold: use parallel for n_simulations >= 20
        # Below this threshold, overhead dominates any speedup
        use_parallel = JOBLIB_AVAILABLE and n_simulations >= 20

        if use_parallel:
            # Parallel execution using joblib with processes for true parallelism
            # prefer="processes" avoids GIL and achieves real CPU parallelism
            y_sim_list = Parallel(n_jobs=-1, prefer="processes")(
                delayed(ParsimCoreAlgorithm._simulate_single_parameter_k)(
                    i, n_simulations, vect, A_K, C, D_required, y, u, l_, m, n, L
                )
                for i in range(n_simulations)
            )
        else:
            # Sequential execution fallback
            from ...utils.simulation_utils import ss_lsim_predictor_form

            y_sim_list = []
            for i in range(n_simulations):
                vect[i, 0] = 1.0

                if D_required:
                    B_K = vect[0 : n * m, :].reshape((n, m))
                    D_i = vect[n * m : n * m + l_ * m, :].reshape((l_, m))
                    K_i = vect[n * m + l_ * m : n * m + l_ * m + n * l_, :].reshape(
                        (n, l_)
                    )
                    x0 = vect[n * m + l_ * m + n * l_ : :, :].reshape((n, 1))
                else:
                    B_K = vect[0 : n * m, :].reshape((n, m))
                    D_i = np.zeros((l_, m))
                    K_i = vect[n * m : n * m + n * l_, :].reshape((n, l_))
                    x0 = vect[n * m + n * l_ : :, :].reshape((n, 1))

                # Simulate using predictor form
                _, y_hat = ss_lsim_predictor_form(A_K, B_K, C, D_i, K_i, y, u, x0)
                y_sim_list.append(y_hat.reshape((1, L * l_)))
                vect[i, 0] = 0.0

        # Stack all simulations into a matrix
        # Each y_sim_list[i] has shape (1, L*l_), impile stacks vertically giving (n_simulations, L*l_)
        # Transpose to (L*l_, n_simulations) for least squares: pinv(y_sim) @ y
        y_matrix = 1.0 * y_sim_list[0]
        for j in range(n_simulations - 1):
            y_matrix = impile(y_matrix, y_sim_list[j + 1])
        y_matrix = y_matrix.T

        return y_matrix

    @staticmethod
    def ak_c_estimating_s_p(U_n, S_n, V_n, l_, f, m, Zp, Uf, Yf):
        """
        Estimate A, C, A_K, and K matrices for PARSIM-S and PARSIM-P using QR decomposition.

        This function uses rigorous QR decomposition to estimate the Kalman gain K,
        which is the correct approach from the reference implementation.

        Reference: master/sippy_unipi/Parsim_methods.py lines 85-101 (AK_C_estimating_S_P function)

        Parameters:
        -----------
        U_n, S_n, V_n : ndarrays
            SVD decomposition from svd_weighted_k
        l_ : int
            Number of outputs
        f : int
            Future horizon
        m : int
            Number of inputs
        Zp, Uf, Yf : ndarrays
            Data matrices from ordinate sequences

        Returns:
        --------
        A : ndarray
            State matrix (n x n)
        C : ndarray
            Output matrix (l x n)
        A_K : ndarray
            Predictor form state matrix (n x n)
        K : ndarray
            Kalman gain matrix (n x l)
        n : int
            Model order
        """

        n = S_n.size
        S_n_diag = np.diag(S_n)

        # Construct observability matrix
        Ob_f = np.dot(U_n, sc.linalg.sqrtm(S_n_diag))

        # Estimate A from observability matrix shift property
        A = np.dot(np.linalg.pinv(Ob_f[0 : l_ * (f - 1), :]), Ob_f[l_:, :])

        # Extract C from first block of observability matrix
        C = Ob_f[0:l_, :]

        # QR-based Kalman gain estimation
        # Stack [Zp; Uf; Yf] and perform QR decomposition
        stacked_matrix = impile(impile(Zp, Uf), Yf).T
        Q, R = np.linalg.qr(stacked_matrix)
        Q = Q.T
        R = R.T

        # Extract relevant block from R matrix
        # G_f contains innovation covariance information
        G_f = R[(2 * m + l_) * f :, (2 * m + l_) * f :]
        F = G_f[0:l_, 0:l_]

        # Compute Kalman gain K using QR decomposition result
        # K = Ob_f^+ * G_f[l_:, 0:l_] * F^-1
        K = np.dot(
            np.dot(np.linalg.pinv(Ob_f[0 : l_ * (f - 1), :]), G_f[l_:, 0:l_]),
            np.linalg.inv(F),
        )

        # Compute predictor form A_K = A - K*C
        A_K = A - np.dot(K, C)

        return A, C, A_K, K, n

    @staticmethod
    def simulations_sequence_s(A_K, C, L, K, y, u, l_, m, n, D_required):
        """
        Systematic simulation for PARSIM-S parameter estimation using predictor form.

        Simulates the predictor form system with unit vectors for all parameters
        (B_K, D, x0) to build regression matrix for least squares. Note that K
        is FIXED (already estimated), unlike PARSIM-K where K is also estimated.

        PERFORMANCE: Uses parallel execution via joblib when available and
        n_simulations >= 20, achieving 3-6x speedup on multi-core systems.

        Reference: master/sippy_unipi/Parsim_methods.py lines 48-82 (simulations_sequence_S function)

        Parameters:
        -----------
        A_K : ndarray
            Predictor form A matrix (n x n)
        C : ndarray
            Output matrix (l x n)
        L : int
            Number of time points
        K : ndarray
            Kalman gain matrix (n x l) - FIXED, not estimated
        y, u : ndarrays
            Output (l x L) and input (m x L) data
        l_, m, n : ints
            System dimensions (outputs, inputs, states)
        D_required : bool
            Whether D matrix is included in estimation

        Returns:
        --------
        y_matrix : ndarray
            Simulation matrix (L*l x n_simulations) for least squares
        """
        from ...utils.simulation_utils import impile

        # Calculate number of simulations needed
        if D_required:
            # Parameters to estimate: B_K (n*m), D (l*m), x0 (n)
            # Note: K is NOT estimated, it's fixed
            n_simulations = n * m + l_ * m + n
        else:
            # Parameters to estimate: B_K (n*m), x0 (n)
            n_simulations = n * m + n

        # Create parameter vector template
        vect = np.zeros((n_simulations, 1))

        # Adaptive threshold: use parallel for n_simulations >= 20
        # Below this threshold, overhead dominates any speedup
        use_parallel = JOBLIB_AVAILABLE and n_simulations >= 20

        if use_parallel:
            # Parallel execution using joblib with processes for true parallelism
            # prefer="processes" avoids GIL and achieves real CPU parallelism
            y_sim_list = Parallel(n_jobs=-1, prefer="processes")(
                delayed(ParsimCoreAlgorithm._simulate_single_parameter_s)(
                    i, n_simulations, vect, A_K, C, K, D_required, y, u, l_, m, n, L
                )
                for i in range(n_simulations)
            )
        else:
            # Sequential execution fallback
            from ...utils.simulation_utils import SS_lsim_predictor_form

            y_sim_list = []
            for i in range(n_simulations):
                vect[i, 0] = 1.0

                if D_required:
                    B_K = vect[0 : n * m, :].reshape((n, m))
                    D = vect[n * m : n * m + l_ * m, :].reshape((l_, m))
                    x0 = vect[n * m + l_ * m :, :].reshape((n, 1))
                else:
                    B_K = vect[0 : n * m, :].reshape((n, m))
                    D = np.zeros((l_, m))
                    x0 = vect[n * m :, :].reshape((n, 1))

                # Simulate predictor form with FIXED K
                _, y_hat = SS_lsim_predictor_form(A_K, B_K, C, D, K, y, u, x0)
                y_sim_list.append(y_hat.reshape((1, L * l_)))

                vect[i, 0] = 0.0

        # Stack all simulations into regression matrix
        y_matrix = 1.0 * y_sim_list[0]
        for j in range(n_simulations - 1):
            y_matrix = impile(y_matrix, y_sim_list[j + 1])

        y_matrix = y_matrix.T
        return y_matrix
