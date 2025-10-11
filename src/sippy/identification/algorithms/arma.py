"""
ARMA (AutoRegressive Moving Average) identification algorithm.
"""

import warnings

import numpy as np
from numpy.linalg import lstsq

from ..base import IdentificationAlgorithm, StateSpaceModel

try:
    import harold

    # Check if harold has the required components
    if hasattr(harold, "State") or hasattr(harold, "StateSpace"):
        HAROLD_AVAILABLE = True
    else:
        HAROLD_AVAILABLE = False
        warnings.warn("harold library incomplete. ARMA algorithm will be limited.")
except ImportError:
    HAROLD_AVAILABLE = False
    warnings.warn("harold library not available. ARMA algorithm will be limited.")


class ARMAAlgorithm(IdentificationAlgorithm):
    """
    ARMA (AutoRegressive Moving Average) identification algorithm.

    The ARMA model structure is:
    A(q) y(k) = C(q) e(k)

    where:
    - A(q) = 1 + a1*q^-1 + ... + ana*q^-na (auto-regressive part)
    - C(q) = 1 + c1*q^-1 + ... + cnc*q^-nc (moving average part)
    - e(k) is white noise

    ARMA models are used for time series analysis and forecasting,
    capturing both the auto-regressive and moving average components
    of stochastic processes.

    The algorithm uses extended least-squares methods to estimate
    the AR and MA parameters simultaneously.
    """

    def __init__(self):
        """Initialize ARMA algorithm."""
        super().__init__()

    def get_algorithm_name(self) -> str:
        """Return algorithm name."""
        return "ARMA"

    def validate_parameters(self, **kwargs) -> bool:
        """
        Validate ARMA-specific parameters.

        Parameters:
        -----------
        **kwargs : dict
            Parameters to validate including na, nc

        Returns:
        --------
        bool
            True if parameters are valid
        """
        na = kwargs.get("na", 1)
        nc = kwargs.get("nc", 1)

        if na <= 0:
            raise ValueError("AR order (na) must be positive")
        if nc <= 0:
            raise ValueError("MA order (nc) must be positive")

        return True

    def identify(self, data, config):
        """
        Identify ARMA model from output data (time series).

        Parameters:
        -----------
        data : IDData
            Output time series data (inputs ignored for ARMA)
        config : SystemIdentificationConfig
            Configuration parameters including na, nc

        Returns:
        --------
        model : StateSpaceModel
            Identified state-space model
        """
        # Extract data from IDData object
        u = data.get_input_array()
        y = data.get_output_array()

        # Extract configuration parameters (ARMA specific)
        na = getattr(config, "na", 1)
        nc = getattr(config, "nc", 1)

        # Validate parameters
        self.validate_parameters(na=na, nc=nc)

        # Get data dimensions (ARMA is typically SISO but support MIMO too)
        ny, N = y.shape
        nu, _ = u.shape

        # For ARMA, inputs are not used - it's a time series model
        # but we handle the possibility by ignoring inputs

        # Calculate effective data length
        max_lag = max(na, nc)
        N_eff = N - max_lag

        if N_eff <= 0:
            raise ValueError(
                f"Not enough data points. Need at least {max_lag + 1} samples, got {N}"
            )

        # MIMO case - handle each output separately (ARMA is typically SISO time series)
        AR_coeffs = np.zeros((ny, na))
        MA_coeffs = np.zeros((ny, nc))
        residuals_list = []

        for i in range(ny):
            # For each output channel (typically just one for ARMA)
            # Construct regression matrix for ARMA estimation
            n_params = na + nc  # Only AR and MA coefficients
            Phi = np.zeros((N_eff, n_params))
            col = 0

            # AR part: lagged outputs
            for lag in range(na):
                Phi[:, col] = y[i, max_lag - 1 - lag : max_lag - 1 - lag + N_eff]
                col += 1

            # MA part: since we don't have access to noise terms directly,
            # we use an iterative approach or approximate method
            # For simplicity, use residuals from initial AR fit as noise estimate
            if nc > 0:
                # Initial AR-only estimate to get residuals
                Phi_ar = Phi[:, :na]
                theta_ar, _, _, _ = lstsq(
                    Phi_ar, y[i, max_lag : max_lag + N_eff], rcond=None
                )
                ar_pred = Phi_ar @ theta_ar
                residuals = y[i, max_lag : max_lag + N_eff] - ar_pred

                # MA part: use residuals as approximation of past noise terms
                for lag in range(nc):
                    if lag == 0:
                        # Current residual is not available in MA part
                        Phi[:, col] = 0
                    else:
                        # Adjust indices to avoid out-of-bounds access
                        start_idx = max_lag - 1 - lag
                        end_idx = start_idx + N_eff
                        if start_idx >= 0 and end_idx <= len(residuals):
                            Phi[:, col] = residuals[start_idx:end_idx]
                        else:
                            Phi[:, col] = 0
                    col += 1
            else:
                # No MA coefficients needed
                pass

            # Solve for ARMA parameters
            theta, residuals_i, rank, s = lstsq(
                Phi, y[i, max_lag : max_lag + N_eff], rcond=None
            )
            residuals_list.append(residuals_i)

            # Extract AR and MA coefficients
            AR_coeffs[i, :] = theta[:na]
            if nc > 0:
                MA_coeffs[i, :] = theta[na:]

        # Create state-space representation
        if HAROLD_AVAILABLE:
            model = self._create_state_space_from_arma(
                AR_coeffs, MA_coeffs, na, nc, ny, data.sample_time
            )
        else:
            # Fallback when harold is not available
            model = self._create_mock_model(
                AR_coeffs, MA_coeffs, na, nc, ny, data.sample_time
            )

        return model

    def _create_state_space_from_arma(self, AR_coeffs, MA_coeffs, na, nc, ny, Ts):
        """
        Create state-space model from ARMA coefficients using harold.

        Parameters:
        -----------
        AR_coeffs, MA_coeffs : ndarray
            AR and MA coefficients
        na, nc : int
            AR and MA orders
        ny : int
            Number of outputs
        Ts : float
            Sampling time

        Returns:
        --------
        model : StateSpaceModel
            State-space model representation
        """
        # For ARMA, we create a transfer function representation
        # A(q) y(k) = C(q) e(k) -> y(k) = C(q)/A(q) * e(k)

        # Create state-space model for each output (SISO for simplicity)
        # Use companion form for the AR part
        n_states = max(na, nc)
        A = np.zeros((n_states, n_states))
        B = np.zeros((n_states, ny))  # Noise input
        C = np.zeros((ny, n_states))
        D = np.eye(ny)  # Direct feedthrough from noise

        # Build companion form for AR coefficients
        for i in range(ny):
            A_sub = np.zeros((max(na, nc), max(na, nc)))

            # Companion form for AR part
            if na > 0:
                if na > 1:
                    A_sub[: na - 1, 1:na] = np.eye(na - 1)
                A_sub[na - 1, :na] = -AR_coeffs[i, :]

            # Handle C matrix for output
            row_offset = i * max(na, nc)
            C_i = np.zeros((1, max(na, nc)))
            C_i[0, max(na, nc) - 1] = 1
            C[row_offset : row_offset + 1, :] = C_i

            # Set B matrix (noise input)
            if nc > 0:
                # MA coefficients affect the noise input
                B_i = np.zeros((max(na, nc), 1))
                B_i[:nc, 0] = MA_coeffs[i, :]
                B[row_offset : row_offset + max(na, nc), i] = B_i.flatten()

            # Assemble into full state space
            if ny == 1:
                A = A_sub
                B = B_i
                C = C_i
            else:
                # For MIMO, stack the SISO systems
                if i == 0:
                    A = A_sub
                    B = B_i
                    C = C_i
                else:
                    A_block = np.zeros(
                        (A.shape[0] + A_sub.shape[0], A.shape[1] + A_sub.shape[1])
                    )
                    A_block[: A.shape[0], : A.shape[1]] = A
                    A_block[A.shape[0] :, A.shape[1] :] = A_sub
                    A = A_block

                    B_block = np.zeros(
                        (B.shape[0] + B_i.shape[0], max(B.shape[1], B_i.shape[1]) + 1)
                    )
                    B_block[: B.shape[0], : B.shape[1]] = B
                    B_block[B.shape[0] :, i] = B_i.flatten()
                    B = B_block

                    C_block = np.vstack([C, C_i])
                    C = C_block

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

    def _create_mock_model(self, AR_coeffs, MA_coeffs, na, nc, ny, Ts):
        """
        Create a mock state-space model when harold is not available.

        Parameters:
        -----------
        AR_coeffs, MA_coeffs : ndarray
            AR and MA coefficients
        na, nc : int
            AR and MA orders
        ny : int
            Number of outputs
        Ts : float
            Sampling time

        Returns:
        --------
        model : StateSpaceModel
            Mock state-space model
        """
        # Create simple companion form state-space representation
        n_states = max(na, nc)

        # State matrix A (companion form)
        A = np.zeros((n_states, n_states))
        if na > 1:
            A[: na - 1, 1:na] = np.eye(na - 1)
        if na > 0:
            A[na - 1, :na] = -AR_coeffs[0, :] if ny == 1 else -AR_coeffs[0, :]

        # Input matrix B (noise input)
        B = np.zeros((n_states, ny))
        if nc > 0:
            num_b_elements = min(nc, n_states)
            B[:num_b_elements, :] = (
                MA_coeffs[0:num_b_elements, :].T
                if ny == 1
                else np.zeros((num_b_elements, ny))
            )

        # Output matrix C
        C = np.zeros((ny, n_states))
        C[:, -1] = 1  # Last state is the output

        # Feedthrough matrix D (from noise to output)
        D = np.eye(ny)

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
