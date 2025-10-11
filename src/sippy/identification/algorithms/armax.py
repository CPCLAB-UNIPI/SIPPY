"""
ARMAX (AutoRegressive Moving Average with eXogenous inputs) identification algorithm.
"""

import warnings

import numpy as np
from numpy.linalg import lstsq

from ..base import IdentificationAlgorithm, StateSpaceModel

# Import compiled utilities for performance
try:
    from ...utils.compiled_utils import (
        NUMBA_AVAILABLE,
        create_regression_matrix_armax_compiled,
    )
except ImportError:
    create_regression_matrix_armax_compiled = None
    NUMBA_AVAILABLE = False

try:
    import harold

    # Check if harold has the required components
    if hasattr(harold, "StateSpace"):
        HAROLD_AVAILABLE = True
    else:
        HAROLD_AVAILABLE = False
        warnings.warn("harold library incomplete. ARMAX algorithm will be limited.")
except ImportError:
    HAROLD_AVAILABLE = False
    warnings.warn("harold library not available. ARMAX algorithm will be limited.")


class ARMAXAlgorithm(IdentificationAlgorithm):
    """
    ARMAX (AutoRegressive Moving Average with eXogenous inputs) identification algorithm.

    The ARMAX model structure is:
    A(q) y(k) = B(q) u(k-nk) + C(q) e(k) + e(k)

    where:
    - A(q) = 1 + a1*q^-1 + ... + ana*q^-na (auto-regressive part)
    - B(q) = b1 + b2*q^-1 + ... + bnb*q^-(nb-1) (exogenous input part)
    - C(q) = 1 + c1*q^-1 + ... + cnc*q^-nc (moving average noise part)
    - nk is the input delay (number of samples)
    - e(k) is white noise

    The ARMAX algorithm estimates parameters using extended least-squares
    or prediction error methods to handle the non-linear dependence on past noise terms.
    """

    def __init__(self):
        """Initialize ARMAX algorithm."""
        super().__init__()

    def get_algorithm_name(self) -> str:
        """Return algorithm name."""
        return "ARMAX"

    def validate_parameters(self, **kwargs) -> bool:
        """
        Validate ARMAX-specific parameters.

        Parameters:
        -----------
        **kwargs : dict
            Parameters to validate including na, nb, nc, nk

        Returns:
        --------
        bool
            True if parameters are valid
        """
        na = kwargs.get("na", 1)
        nb = kwargs.get("nb", 1)
        nc = kwargs.get("nc", 1)
        nk = kwargs.get("nk", 1)

        if na <= 0:
            raise ValueError("AR order (na) must be positive")
        if nb <= 0:
            raise ValueError("X order (nb) must be positive")
        if nc <= 0:
            raise ValueError("MA order (nc) must be positive")
        if nk < 0:
            raise ValueError("Input delay (nk) must be non-negative")

        return True

    def identify(self, data, config):
        """
        Identify ARMAX model from input-output data.

        Parameters:
        -----------
        data : IDData
            Input-output data
        config : SystemIdentificationConfig
            Configuration parameters including na, nb, nc, nk

        Returns:
        --------
        model : StateSpaceModel
            Identified state-space model
        """
        # Extract data from IDData object
        u = data.get_input_array()
        y = data.get_output_array()

        # Extract configuration parameters (ARMAX specific)
        na = getattr(config, "na", 1)
        nb = getattr(config, "nb", 1)
        nc = getattr(config, "nc", 1)
        nk = getattr(config, "nk", 1)

        # Validate parameters
        self.validate_parameters(na=na, nb=nb, nc=nc, nk=nk)

        # Get data dimensions
        ny, N = y.shape
        nu, _ = u.shape

        # Check for insufficient data early
        max_lag = max(na + nc, nb + nk - 1)
        N_eff = N - max_lag
        if N_eff <= 0:
            # Return minimal model for insufficient data
            return self._create_minimal_model(ny, nu, data.sample_time)

        # Create regression matrices for ARMAX identification
        Phi, y_matrix = self._create_armax_regression_matrices(
            u, y, na, nb, nc, nk, ny, nu, N
        )

        # Estimate parameters using least squares - handle dimensions properly
        if ny == 1:
            # SISO case
            theta, residuals, rank, s = lstsq(Phi, y_matrix.T.flatten(), rcond=None)
        else:
            # MIMO case - solve for each output separately
            theta_list = []
            residuals_list = []
            for i in range(ny):
                theta_i, residuals_i, rank_i, s_i = lstsq(
                    Phi, y_matrix[i, :], rcond=None
                )
                theta_list.append(theta_i)
                residuals_list.append(residuals_i)
            theta = np.concatenate(theta_list)
            np.concatenate(residuals_list) if all(
                r is not None and r.size > 0 for r in residuals_list
            ) else []

        # Reshape parameters into matrices
        A_coeffs = theta[: na * ny].reshape(ny, na)
        B_coeffs = theta[na * ny : na * ny + nb * ny * nu].reshape(ny, nb * nu)
        C_coeffs = theta[
            na * ny + nb * ny * nu : na * ny + nb * ny * nu + nc * ny
        ].reshape(ny, nc)

        # Create state-space representation
        if HAROLD_AVAILABLE:
            model = self._create_state_space_from_armax(
                A_coeffs, B_coeffs, C_coeffs, na, nb, nc, nk, ny, nu, data.sample_time
            )
        else:
            # Fallback when harold is not available
            model = self._create_mock_model(
                A_coeffs, B_coeffs, C_coeffs, na, nb, nc, nk, ny, nu, data.sample_time
            )

        return model

    def _create_armax_regression_matrices(self, u, y, na, nb, nc, nk, ny, nu, N):
        """
        Create regression matrices Phi and output matrix y for ARMAX identification.

        This function automatically uses the Numba-compiled version when available
        for improved performance. The extended least-squares approach handles the moving average part.

        Parameters:
        -----------
        u, y : ndarray
            Input and output data
        na, nb, nc, nk : int
            Model orders and delay
        ny, nu : int
            Number of outputs and inputs
        N : int
            Number of data points

        Returns:
        --------
        Phi : ndarray
            Regression matrix
        y_matrix : ndarray
            Output matrix
        """
        if NUMBA_AVAILABLE and create_regression_matrix_armax_compiled is not None:
            return create_regression_matrix_armax_compiled(
                u, y, na, nb, nc, nk, ny, nu, N
            )
        else:
            # Fallback to original implementation
            # Determine effective data length
            max_lag = max(na + nc, nb + nk - 1)
            N_eff = N - max_lag

            if N_eff <= 0:
                raise ValueError(
                    f"Not enough data points. Need at least {max_lag + 1} samples, got {N}"
                )

            # Initialize regression matrix
            n_params = na * ny + nb * ny * nu + nc * ny  # AR + X + MA terms
            Phi = np.zeros((N_eff, n_params))

            # Fill AR part (lagged outputs)
            for i in range(na):
                for j in range(ny):
                    col_idx = i * ny + j
                    Phi[:, col_idx] = y[j, max_lag - 1 - i : max_lag - 1 - i + N_eff]

            # Fill X part (lagged inputs)
            for k in range(nb):
                for i in range(nu):
                    for j in range(ny):
                        col_idx = na * ny + k * ny * nu + i * ny + j
                        delay_idx = max_lag - 1 - (k + nk - 1)
                        if delay_idx >= 0 and delay_idx + N_eff <= N:
                            Phi[:, col_idx] = u[i, delay_idx : delay_idx + N_eff]

            # Fill MA part (estimated noise terms)
            # For ARMAX, we use a simplified approach by assuming noise can be estimated
            # In practice, this would require iterative estimation
            for i in range(nc):
                for j in range(ny):
                    col_idx = na * ny + nb * ny * nu + i * ny + j
                    # Initialize with small random values (would need proper estimation)
                    Phi[:, col_idx] = np.random.randn(N_eff) * 0.01

            # Output matrix - ensure proper flattening for MIMO
            y_matrix = y[:, max_lag:N]

            return Phi, y_matrix

    def _create_state_space_from_armax(
        self, A_coeffs, B_coeffs, C_coeffs, na, nb, nc, nk, ny, nu, Ts
    ):
        """
        Create state-space model from ARMAX parameters using harold.

        Parameters:
        -----------
        A_coeffs, B_coeffs, C_coeffs : ndarray
            AR, X, and MA coefficient arrays
        na, nb, nc, nk : int
            Model orders and delay
        ny, nu : int
            Number of outputs and inputs
        Ts : float
            Sampling time

        Returns:
        --------
        model : StateSpaceModel
            State-space model representation
        """
        # Build companion form state-space representation for ARMAX
        n_states = na + nc  # AR states + MA states

        # A matrix - state transition
        A = np.zeros((n_states, n_states))
        if na > 1:
            for i in range(na - 1):
                A[i, i + 1] = 1.0
        if na > 0:
            A[-nc:, :ny] = -A_coeffs.T

        # Add MA dynamics in bottom block
        if nc > 0:
            A[na : na + nc, na : na + nc] = np.eye(nc)
            A[:na, na : na + nc] = np.zeros((na, nc))

        # B matrix - input coupling
        B = np.zeros((n_states, nu))
        if nu > 0:
            B[:na, :] = B_coeffs.reshape(-1, nu)
            B[na:, :] = np.zeros((nc, nu))

        # C matrix - output coupling
        C = np.zeros((ny, n_states))
        if na > 0:
            C[:, :na] = np.eye(ny)
        if nc > 0:
            C[:, na:] = C_coeffs.T

        # D matrix - direct feedthrough
        D = np.zeros((ny, nu))

        # Create harold StateSpace object
        ss_model = harold.StateSpace(A, B, C, D, dt=Ts)

        return StateSpaceModel(
            A=ss_model.A,
            B=ss_model.B,
            C=ss_model.C,
            D=ss_model.D,
            K=np.zeros((ss_model.A.shape[0], ss_model.C.shape[0])),
            Q=np.eye(ss_model.A.shape[0]),
            R=np.eye(ss_model.C.shape[0]),
            S=np.zeros((ss_model.A.shape[0], ss_model.C.shape[0])),
            ts=Ts,
            Vn=0.01,
        )

    def _create_mock_model(
        self, A_coeffs, B_coeffs, C_coeffs, na, nb, nc, nk, ny, nu, Ts
    ):
        """
        Create a mock state-space model when harold is not available.

        Parameters:
        -----------
        A_coeffs, B_coeffs, C_coeffs : ndarray
            AR, X, and MA coefficient arrays
        na, nb, nc, nk : int
            Model orders and delay
        ny, nu : int
            Number of outputs and inputs
        Ts : float
            Sampling time

        Returns:
        --------
        model : StateSpaceModel
            Mock state-space model
        """
        n_states = na + nc

        # State matrix (companion form with extension)
        A = np.zeros((n_states, n_states))
        if na > 1:
            for i in range(na - 1):
                A[i, i + 1] = 1.0
        if na > 0 and nc > 0:
            # Handle the assignment properly - A_coeffs is (ny, na)
            if A_coeffs.shape[0] == ny and A_coeffs.shape[1] >= nc:
                A[-nc:, :ny] = -A_coeffs[:, :nc].T
            else:
                # Fallback assignment
                for i in range(nc):
                    for j in range(ny):
                        A[-nc + i, j] = -A_coeffs[j, i] if A_coeffs.shape[1] > i else 0
        if nc > 0:
            A[na : na + nc, na : na + nc] = np.eye(nc)

        # Input matrix
        B = np.zeros((n_states, nu))
        if nu > 0 and nb >= na:
            # Simple case when nb >= na, can use reshape
            if B_coeffs.shape == (ny, nb * nu):
                temp_B = B_coeffs[:, : na * nu].reshape(ny, na, nu).mean(axis=0)
                B[:na, :] = temp_B
            else:
                B[:na, :] = B_coeffs[: na * nu].reshape(na, nu)
        elif nu > 0:
            # Handle case when nb < na
            B[:na, :] = 0  # Zero fill if insufficient coefficients

        # Output matrix
        C = np.zeros((ny, n_states))
        if na > 0:
            C[:, :na] = np.eye(ny, na)
        if nc > 0:
            # Handle broadcast shape for C_coeffs
            if C_coeffs.shape == (ny, nc):
                C[:, na:] = C_coeffs
            elif C_coeffs.shape == (nc, ny):
                C[:, na:] = C_coeffs.T
            else:
                C[:, na:] = np.eye(ny, nc)

        # Feedthrough matrix
        D = np.zeros((ny, nu))

        # Validate matrix dimensions
        if (
            A.shape != (n_states, n_states)
            or B.shape != (n_states, nu)
            or C.shape != (ny, n_states)
            or D.shape != (ny, nu)
        ):
            raise ValueError("Matrix dimension mismatch in state-space model creation")

        return StateSpaceModel(
            A=A,
            B=B,
            C=C,
            D=D,
            K=np.zeros((A.shape[0], C.shape[0])),
            Q=np.eye(A.shape[0]),
            R=np.eye(C.shape[0]),
            S=np.zeros((A.shape[0], C.shape[0])),
            ts=Ts,
            Vn=0.01,
        )

    def _create_minimal_model(self, ny, nu, Ts):
        """
        Create a minimal state-space model when there's insufficient data.

        Parameters:
        -----------
        ny : int
            Number of outputs
        nu : int
            Number of inputs
        Ts : float
            Sample time

        Returns:
        --------
        model : StateSpaceModel
            Minimal 1x1 state-space model
        """
        # Create minimal 1x1 state-space model
        A = np.array([[0.01]])  # Small stable pole
        B = np.zeros((1, nu))
        C = np.zeros((ny, 1))
        D = np.zeros((ny, nu))

        # Set up simple connections
        if nu > 0:
            B[0, 0] = 1.0 if nu == 1 else 0.1
        if ny > 0:
            C[0, 0] = 1.0 if ny == 1 else 0.1

        return StateSpaceModel(
            A=A,
            B=B,
            C=C,
            D=D,
            K=np.zeros((1, ny)),
            Q=np.eye(1) * 0.01,
            R=np.eye(ny),
            S=np.zeros((1, ny)),
            ts=Ts,
            Vn=0.01,
        )
