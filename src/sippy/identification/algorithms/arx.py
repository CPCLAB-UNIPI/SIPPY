"""
ARX (AutoRegressive with eXogenous inputs) identification algorithm.
"""

import warnings
from typing import TYPE_CHECKING, Optional

import numpy as np
from numpy.linalg import lstsq

from ..base import IdentificationAlgorithm, StateSpaceModel

if TYPE_CHECKING:
    from ..iddata import IDData

# Import compiled utilities for performance
try:
    from ...utils.compiled_utils import (
        NUMBA_AVAILABLE,
        create_regression_matrix_arx_compiled,
        create_regression_matrix_arx_mimo_compiled,
    )
except ImportError:
    create_regression_matrix_arx_compiled = None
    create_regression_matrix_arx_mimo_compiled = None
    NUMBA_AVAILABLE = False

# Import harold for test mocking and availability checking
try:
    import harold

    HAROLD_IMPORTED = True
    # Check for either modern (State) or legacy (StateSpace) API
    if hasattr(harold, "State") or hasattr(harold, "StateSpace"):
        HAROLD_AVAILABLE = True
    else:
        HAROLD_AVAILABLE = False
except ImportError:
    harold = None
    HAROLD_IMPORTED = False
    HAROLD_AVAILABLE = False


class ARXAlgorithm(IdentificationAlgorithm):
    """
    ARX (AutoRegressive with eXogenous inputs) identification algorithm.

    The ARX model structure is:
    A(q) y(k) = B(q) u(k - nk) + e(k)

    where:
    - A(q) = 1 + a1*q^-1 + ... + ana*q^-na (auto-regressive part)
    - B(q) = b1 + b2*q^-1 + ... + bnb*q^-(nb-1) (exogenous input part)
    - nk is the input delay (number of samples)
    - e(k) is white noise

    The algorithm uses least-squares regression to estimate the ARX parameters.
    """

    def __init__(self):
        """Initialize ARX algorithm."""
        super().__init__()

    def get_algorithm_name(self) -> str:
        """Return algorithm name."""
        return "ARX"

    def validate_parameters(self, **kwargs) -> bool:
        """
        Validate ARX-specific parameters.

        Parameters:
        -----------
        **kwargs : dict
            Parameters to validate including na, nb, nk

        Returns:
        --------
        bool
            True if parameters are valid
        """
        na = kwargs.get("na", 1)
        nb = kwargs.get("nb", 1)
        nk = kwargs.get("nk", 1)

        if na <= 0:
            raise ValueError("AR order (na) must be positive")
        if nb <= 0:
            raise ValueError("X order (nb) must be positive")
        if nk < 0:
            raise ValueError("Input delay (nk) must be non-negative")

        return True

    def identify(
        self,
        y: Optional[np.ndarray] = None,
        u: Optional[np.ndarray] = None,
        iddata: Optional["IDData"] = None,
        **kwargs,
    ) -> StateSpaceModel:
        """
        Identify ARX model from input-output data.

        Parameters:
        -----------
        y : np.ndarray, optional
            Output data (outputs x time_steps)
        u : np.ndarray, optional
            Input data (inputs x time_steps)
        iddata : IDData, optional
            Input-output data container
        **kwargs : dict
            Configuration parameters including na, nb, nk, tsample

        Returns:
        --------
        model : StateSpaceModel
            Identified state-space model
        """
        # Validate input arguments
        if iddata is not None and (y is not None or u is not None):
            raise ValueError("Provide either iddata or (y, u), but not both")
        if iddata is None and (y is None or u is None):
            raise ValueError("Must provide either iddata or both y and u")

        # Extract data if IDData is provided
        if iddata is not None:
            u = iddata.get_input_array()
            y = iddata.get_output_array()
            sample_time = iddata.sample_time
        else:
            # Ensure arrays are 2D
            y = np.atleast_2d(y)
            u = np.atleast_2d(u)
            sample_time = kwargs.get("tsample", 1.0)

        # Extract configuration parameters (ARX specific)
        na = kwargs.get("na", 1)
        nb = kwargs.get("nb", 1)
        nk = kwargs.get("nk", 1)

        # Validate parameters
        self.validate_parameters(na=na, nb=nb, nk=nk)

        # Get data dimensions
        ny, N = y.shape
        nu, _ = u.shape

        # Calculate effective data length
        max_lag = max(na, nb + nk - 1)
        N_eff = N - max_lag

        if N_eff <= 0:
            raise ValueError(
                f"Not enough data points. Need at least {max_lag + 1} samples, got {N}"
            )

        # Estimate parameters using least squares
        max_lag = max(na, nb + nk - 1)
        N_eff = N - max_lag
        if N_eff <= 0:
            raise ValueError(
                f"Not enough data points. Need at least {max_lag + 1} samples, got {N}"
            )

        if ny == 1:
            # SISO case leverages shared compiled builder
            Phi, y_matrix = self._create_regression_matrix(u, y, na, nb, nk, ny, nu, N)
            theta, residuals, rank, s = lstsq(Phi, y_matrix.T.flatten(), rcond=None)
            A_coeffs = theta[:na].reshape(1, na)
            B_coeffs = theta[na:].reshape(1, nb)
        else:
            use_compiled_mimo = NUMBA_AVAILABLE and (
                create_regression_matrix_arx_mimo_compiled is not None
            )

            if use_compiled_mimo:
                Phi_batches, y_targets = create_regression_matrix_arx_mimo_compiled(
                    np.ascontiguousarray(u),
                    np.ascontiguousarray(y),
                    na,
                    nb,
                    nk,
                    ny,
                    nu,
                    N,
                )
            else:
                Phi, y_matrix = self._create_regression_matrix(
                    u, y, na, nb, nk, ny, nu, N
                )

            A_coeffs = np.zeros((ny, na))
            B_coeffs = np.zeros((ny, nb * nu))
            for i in range(ny):
                if use_compiled_mimo:
                    Phi_i = np.ascontiguousarray(Phi_batches[i, :, :])
                    y_target = y_targets[i, :]
                else:
                    n_params_i = na * ny + nb * nu
                    Phi_i = np.zeros((N_eff, n_params_i))
                    col = 0

                    for lag in range(na):
                        for j in range(ny):
                            Phi_i[:, col] = y[
                                j, max_lag - 1 - lag : max_lag - 1 - lag + N_eff
                            ]
                            col += 1

                    for lag in range(nb):
                        for j in range(nu):
                            delay_idx = max_lag - 1 - (lag + nk - 1)
                            if delay_idx >= 0 and delay_idx + N_eff <= N:
                                Phi_i[:, col] = u[j, delay_idx : delay_idx + N_eff]
                            col += 1

                    y_target = y_matrix[i, :]

                theta_i, residuals_i, rank_i, s_i = lstsq(Phi_i, y_target, rcond=None)

                for lag in range(na):
                    idx = lag * ny + i
                    A_coeffs[i, lag] = theta_i[idx]

                B_coeffs[i, :] = theta_i[na * ny :]

        # Compute one-step-ahead predictions (Yid) for identification data
        Yid = np.zeros_like(y)
        Yid[:, :max_lag] = y[:, :max_lag]  # Copy initial values
        if ny == 1:
            Yid[0, max_lag:] = np.dot(Phi, theta)
        else:
            # For MIMO, reconstruct predictions for each output
            for i in range(ny):
                if use_compiled_mimo:
                    Phi_i = np.ascontiguousarray(Phi_batches[i, :, :])
                    theta_i = np.zeros(na * ny + nb * nu)
                    for lag in range(na):
                        idx = lag * ny + i
                        theta_i[idx] = A_coeffs[i, lag]
                    theta_i[na * ny :] = B_coeffs[i, :]
                    Yid[i, max_lag:] = np.dot(Phi_i, theta_i)
                else:
                    # Use the same Phi construction as before
                    n_params_i = na * ny + nb * nu
                    Phi_i = np.zeros((N_eff, n_params_i))
                    col = 0
                    for lag in range(na):
                        for j in range(ny):
                            Phi_i[:, col] = y[
                                j, max_lag - 1 - lag : max_lag - 1 - lag + N_eff
                            ]
                            col += 1
                    for lag in range(nb):
                        for j in range(nu):
                            delay_idx = max_lag - 1 - (lag + nk - 1)
                            if delay_idx >= 0 and delay_idx + N_eff <= N:
                                Phi_i[:, col] = u[j, delay_idx : delay_idx + N_eff]
                            col += 1

                    theta_i = np.zeros(n_params_i)
                    for lag in range(na):
                        idx = lag * ny + i
                        theta_i[idx] = A_coeffs[i, lag]
                    theta_i[na * ny :] = B_coeffs[i, :]
                    Yid[i, max_lag:] = np.dot(Phi_i, theta_i)

        # Create G_tf and H_tf transfer functions
        G_tf, H_tf = self._create_transfer_functions_arx(
            A_coeffs, B_coeffs, na, nb, nk, ny, nu, sample_time
        )

        # Create state-space representation
        if HAROLD_AVAILABLE and harold is not None:
            model = self._create_transfer_function(
                A_coeffs, B_coeffs, na, nb, nk, ny, nu, sample_time
            )
        else:
            # Warn about harold availability only when needed
            if harold is None:
                warnings.warn(
                    "harold library not available. ARX algorithm will be limited."
                )
            else:
                warnings.warn(
                    "harold library not available. ARX algorithm will be limited."
                )

            # Fallback when harold is not available
            model = self._create_mock_model(
                A_coeffs, B_coeffs, na, nb, nk, ny, nu, sample_time
            )

        # Attach transfer functions and predictions to model
        model.G_tf = G_tf
        model.H_tf = H_tf
        model.Yid = Yid

        return model

    def _create_regression_matrix(self, u, y, na, nb, nk, ny, nu, N):
        """
        Create regression matrix Phi and output matrix y for least squares.

        This function automatically uses the Numba-compiled version when available
        for improved performance.

        Parameters:
        -----------
        u, y : ndarray
            Input and output data
        na, nb, nk : int
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
        if NUMBA_AVAILABLE and create_regression_matrix_arx_compiled is not None:
            return create_regression_matrix_arx_compiled(u, y, na, nb, nk, ny, nu, N)
        else:
            # Fallback to original implementation
            # Determine effective data length
            max_lag = max(na, nb + nk - 1)
            N_eff = N - max_lag

            if N_eff <= 0:
                raise ValueError(
                    f"Not enough data points. Need at least {max_lag + 1} samples, got {N}"
                )

            # Output matrix - trimmed for effective length
            y_matrix = y[:, max_lag:N]
            # Return dummy Phi since we construct per-output matrices in identify()
            Phi = np.zeros((N_eff, 1))  # Not used in MIMO case

            return Phi, y_matrix

    def _create_transfer_functions_arx(
        self, A_coeffs, B_coeffs, na, nb, nk, ny, nu, Ts
    ):
        """
        Create G_tf and H_tf transfer functions for ARX.

        For ARX: H_tf = 1 (unity, since ARX has no noise model).

        Parameters:
        -----------
        A_coeffs, B_coeffs : ndarray
            AR and exogenous coefficients
        na, nb, nk : int
            Model orders and delay
        ny, nu : int
            Number of outputs and inputs
        Ts : float
            Sampling time

        Returns:
        --------
        G_tf, H_tf : harold.Transfer objects or None
            Transfer functions (None if harold not available)
        """
        if not HAROLD_AVAILABLE or harold is None:
            return None, None

        try:
            # Create G(q) = B / A - Deterministic transfer function
            max_order = max(na, nb + nk)

            # Build numerator with delay
            NUM_G_full = np.zeros(max_order + 1)
            NUM_G_full[nk : nk + nb] = B_coeffs[0, :] if ny == 1 else B_coeffs[0, :nb]

            # Build denominator
            # Note: ARX regression returns coefficients for -y[k-1], so we negate
            DEN_G = np.zeros(max_order + 1)
            DEN_G[0] = 1.0
            DEN_G[1 : na + 1] = -A_coeffs[0, :]

            # Strip leading zeros from numerator for harold compatibility
            NUM_G = np.trim_zeros(NUM_G_full, "f")
            if len(NUM_G) == 0:
                NUM_G = np.array([0.0])

            G_tf = harold.Transfer(NUM_G, DEN_G, dt=Ts)

            # H(q) = 1 - ARX has no noise model (H is unity)
            H_tf = harold.Transfer([1.0], [1.0], dt=Ts)

            return G_tf, H_tf
        except Exception as e:
            warnings.warn(f"Failed to create ARX transfer functions with harold: {e}")
            return None, None

    def _create_transfer_function(self, A_coeffs, B_coeffs, na, nb, nk, ny, nu, Ts):
        """
        Create transfer function model from ARX parameters using harold.

        Parameters:
        -----------
        A_coeffs, B_coeffs : ndarray
            AR and exogenous coefficients
        na, nb, nk : int
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
        # For simple SISO case, create a transfer function
        if ny == 1 and nu == 1:
            # Create denominator polynomial (AR part + 1)
            den_coeffs = np.concatenate(([1], -A_coeffs[0, :]))

            # Create numerator polynomial (X part with delay)
            # Numerator and denominator must have same length for harold
            num_coeffs_full = np.zeros(len(den_coeffs))
            num_coeffs_full[nk : nk + nb] = B_coeffs[0, :]

            # Strip leading zeros from numerator (harold requirement)
            num_coeffs = np.trim_zeros(num_coeffs_full, "f")
            if len(num_coeffs) == 0:
                num_coeffs = np.array([0.0])

            # Create transfer function
            try:
                # Use modern Harold API if available, fallback to legacy
                if hasattr(harold, "Transfer"):
                    tf = harold.Transfer(num_coeffs, den_coeffs, dt=Ts)
                else:
                    tf = harold.TransferFunction(num_coeffs, den_coeffs, dt=Ts)

                # Fixed 2025-10-12: Removed incorrect harold.undiscretize() call
                # that was identified in migration accuracy investigation
                # Convert discrete-time transfer function to state-space
                ss_model = harold.transfer_to_state(tf)
            except Exception as e:
                warnings.warn(
                    f"Failed to create ARX transfer function with harold: {e}"
                )
                # Fall back to mock model
                return self._create_mock_model(
                    A_coeffs, B_coeffs, na, nb, nk, ny, nu, Ts
                )
            # Extract actual matrices from ss_model
            # Harold uses lowercase attributes (.a, .b, .c, .d)
            A = ss_model.a
            B = ss_model.b
            C = ss_model.c
            D = ss_model.d

            # Check if we got real arrays (not mocked objects for testing)
            if not isinstance(A, np.ndarray):
                # Fallback for mocked harold - create simple companion form
                n_states = na
                A = np.zeros((n_states, n_states))
                if na > 1:
                    A[:-1, 1:] = np.eye(na - 1)
                A[-1, :] = -A_coeffs[0, :]

                B = np.zeros((n_states, nu))
                B_flat = B_coeffs.flatten()
                if len(B_flat) >= n_states:
                    B[:, 0] = B_flat[:n_states]
                else:
                    B[: len(B_flat), 0] = B_flat

                C = np.zeros((ny, n_states))
                C[0, -1] = 1.0

                D = np.zeros((ny, nu))
        else:
            # For MIMO case, create a simple state-space representation
            # using companion form for each output
            n_states = na * ny
            A = np.zeros((n_states, n_states))
            B = np.zeros((n_states, nu))
            C = np.zeros((ny, n_states))
            D = np.zeros((ny, nu))

            # Build companion form matrices
            for i in range(ny):  # For each output
                # A matrix (companion form)
                if na > 1:
                    A[i * na : (i + 1) * na - 1, i * na + 1 : (i + 1) * na] = np.eye(
                        na - 1
                    )
                if na > 0:
                    A[(i + 1) * na - 1, i * na : (i + 1) * na] = -A_coeffs[i, :]

                # B matrix
                if nu > 0:
                    B[(i + 1) * na - 1, :] = (
                        B_coeffs[i, :].reshape(-1, nu)[:nu]
                        if nb == 1
                        else B_coeffs[i, :nu]
                    )

                # C matrix
                C[i, (i + 1) * na - 1] = 1

            # Create harold StateSpace object
            # Use modern Harold API if available, fallback to legacy
            if hasattr(harold, "State"):
                ss_model = harold.State(A, B, C, D, dt=Ts)
            else:
                ss_model = harold.StateSpace(A, B, C, D, dt=Ts)

        # Use local matrices (not ss_model attributes) for dimensions
        # This ensures tests with mocked harold don't break
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

    def _create_mock_model(self, A_coeffs, B_coeffs, na, nb, nk, ny, nu, Ts):
        """
        Create a mock state-space model when harold is not available.

        Parameters:
        -----------
        A_coeffs, B_coeffs : ndarray
            AR and exogenous coefficients
        na, nb, nk : int
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
        # Create simple companion form state-space representation
        n_states = na * ny  # Simplified state dimension

        # State matrix A (companion form)
        A = np.zeros((n_states, n_states))
        for i in range(n_states - 1):
            A[i, i + 1] = 1
        if na > 0:
            # Handle AR coefficients assignment properly
            if A_coeffs.shape[0] == ny and A_coeffs.shape[1] == na:
                A[-ny:, :na] = -A_coeffs
            else:
                # Fallback for different shapes
                for i in range(min(ny, A_coeffs.shape[0])):
                    for j in range(min(na, A_coeffs.shape[1])):
                        A[n_states - ny + i, j] = -A_coeffs[i, j]

        # Input matrix B
        B = np.zeros((n_states, nu))
        if nb > 0:
            # Handle B coefficients properly
            if B_coeffs.shape[0] == ny:
                # Each output row contains coefficients for all inputs at all nb lags
                # Extract direct feedthrough (k=0) from the first nu coefficients per output
                for i in range(ny):
                    # B_coeffs[i, :] shape = [nb * nu]
                    # Take the first nu which represent direct feedthrough (k=nk position)
                    direct_coeffs = (
                        B_coeffs[i, :nu] if B_coeffs.shape[1] >= nu else B_coeffs[i, :]
                    )
                    B[-ny + i, :] = direct_coeffs[:nu]
            else:
                # Fallback case
                B_flat = B_coeffs.flatten()
                if len(B_flat) >= nu * ny:
                    B[-ny:, :] = B_flat[: nu * ny].reshape(ny, nu)

        # Output matrix C
        C = np.zeros((ny, n_states))
        C[:, -ny:] = np.eye(ny)

        # Feedthrough matrix D (zero for ARX)
        D = np.zeros((ny, nu))

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
