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
    if hasattr(harold, "State"):
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

        # Compute one-step-ahead predictions (Yid) for identification data
        Yid = np.zeros_like(y)
        Yid[:, :max_lag] = y[:, :max_lag]  # Copy initial values

        for i in range(ny):
            # Reconstruct predictions for each output using AR and MA terms
            n_params = na + nc
            Phi_i = np.zeros((N_eff, n_params))
            col = 0

            # AR part
            for lag in range(na):
                Phi_i[:, col] = y[i, max_lag - 1 - lag : max_lag - 1 - lag + N_eff]
                col += 1

            # MA part (using residuals from AR fit)
            if nc > 0:
                Phi_ar = Phi_i[:, :na]
                ar_pred = Phi_ar @ AR_coeffs[i, :]
                residuals = y[i, max_lag : max_lag + N_eff] - ar_pred

                for lag in range(nc):
                    if lag == 0:
                        Phi_i[:, col] = 0
                    else:
                        start_idx = max_lag - 1 - lag
                        end_idx = start_idx + N_eff
                        if start_idx >= 0 and end_idx <= len(residuals):
                            Phi_i[:, col] = residuals[start_idx:end_idx]
                        else:
                            Phi_i[:, col] = 0
                    col += 1

            theta_i = np.concatenate([AR_coeffs[i, :], MA_coeffs[i, :] if nc > 0 else []])
            Yid[i, max_lag:] = np.dot(Phi_i, theta_i).flatten()

        # Create G_tf and H_tf transfer functions
        G_tf, H_tf = self._create_transfer_functions_arma(
            AR_coeffs, MA_coeffs, na, nc, ny, data.sample_time
        )

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

        # Attach transfer functions and predictions to model
        model.G_tf = G_tf
        model.H_tf = H_tf
        model.Yid = Yid

        return model

    def _create_transfer_functions_arma(self, AR_coeffs, MA_coeffs, na, nc, ny, Ts):
        """
        Create G_tf and H_tf transfer functions for ARMA.

        For ARMA: G_tf = None (no input, time series only), H_tf = C(q)/A(q).

        Parameters:
        -----------
        AR_coeffs, MA_coeffs : ndarray
            AR and MA coefficient arrays
        na, nc : int
            AR and MA orders
        ny : int
            Number of outputs
        Ts : float
            Sampling time

        Returns:
        --------
        G_tf, H_tf : harold.Transfer objects or None
            Transfer functions (None if harold not available)
        """
        if not HAROLD_AVAILABLE:
            return None, None

        try:
            import harold

            # G_tf = None - ARMA is a time series model with no input
            G_tf = None

            # H_tf = C(q)/A(q) - Noise transfer function
            max_order = max(na, nc)

            NUM_H = np.zeros(max_order + 1)
            NUM_H[0] = 1.0
            NUM_H[1:nc + 1] = MA_coeffs[0, :] if ny == 1 else MA_coeffs[0, :]

            DEN_H = np.zeros(max_order + 1)
            DEN_H[0] = 1.0
            DEN_H[1:na + 1] = AR_coeffs[0, :] if ny == 1 else AR_coeffs[0, :]

            H_tf = harold.Transfer(NUM_H, DEN_H, dt=Ts)

            return G_tf, H_tf
        except Exception as e:
            warnings.warn(f"Failed to create ARMA transfer functions with harold: {e}")
            return None, None

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

        # Build companion form state-space for each output channel
        # Strategy: Build complete subsystems first, then assemble
        n_states_per_output = max(na, nc)

        # Initialize matrices - for SISO or first step of MIMO
        A = None
        B = None
        C = None
        D = np.eye(ny)  # Direct feedthrough from noise

        # Build companion form for each output channel
        for i in range(ny):
            # Build subsystem for output i
            A_sub = np.zeros((n_states_per_output, n_states_per_output))

            # Companion form for AR part
            if na > 0:
                if na > 1:
                    A_sub[: na - 1, 1:na] = np.eye(na - 1)
                A_sub[na - 1, :na] = -AR_coeffs[i, :]

            # Build B_sub for this output (noise input)
            B_sub = np.zeros((n_states_per_output, 1))
            if nc > 0:
                # MA coefficients affect the noise input
                B_sub[:nc, 0] = MA_coeffs[i, :]

            # Assemble into full state space
            if ny == 1:
                # SISO case - use subsystem directly
                A = A_sub
                B = B_sub
                # C_sub for SISO: picks last state
                C = np.zeros((1, n_states_per_output))
                C[0, n_states_per_output - 1] = 1
            else:
                # MIMO case - build block-diagonal structure
                if i == 0:
                    # First output - initialize
                    A = A_sub
                    B = B_sub
                    # C for first output - picks last state of first block
                    C = np.zeros((1, n_states_per_output))
                    C[0, n_states_per_output - 1] = 1
                else:
                    # Subsequent outputs - expand block-diagonally
                    # Expand A block-diagonally
                    A_block = np.zeros(
                        (A.shape[0] + A_sub.shape[0], A.shape[1] + A_sub.shape[1])
                    )
                    A_block[: A.shape[0], : A.shape[1]] = A
                    A_block[A.shape[0] :, A.shape[1] :] = A_sub
                    A = A_block

                    # Expand B - each output has its own noise input column
                    B_block = np.zeros((B.shape[0] + B_sub.shape[0], i + 1))
                    B_block[: B.shape[0], :i] = B
                    B_block[B.shape[0] :, i] = B_sub.flatten()
                    B = B_block

                    # Expand C - need to expand existing rows and add new row
                    # Existing rows get padded with zeros to the right
                    C_expanded = np.zeros((C.shape[0], A.shape[1]))
                    C_expanded[:, : C.shape[1]] = C
                    # New row for output i - picks last state of new block
                    C_new = np.zeros((1, A.shape[1]))
                    C_new[0, A.shape[1] - 1] = 1
                    C = np.vstack([C_expanded, C_new])

        # Create harold State object (for validation)
        _ss_model = harold.State(A, B, C, D, dt=Ts)

        # Use local matrices (not _ss_model attributes) for dimensions
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
        # Use same logic as _create_state_space_from_arma but without harold
        n_states_per_output = max(na, nc)

        # Initialize matrices
        A = None
        B = None
        C = None
        D = np.eye(ny)

        # Build companion form for each output channel
        for i in range(ny):
            # Build subsystem for output i
            A_sub = np.zeros((n_states_per_output, n_states_per_output))

            # Companion form for AR part
            if na > 0:
                if na > 1:
                    A_sub[: na - 1, 1:na] = np.eye(na - 1)
                A_sub[na - 1, :na] = -AR_coeffs[i, :]

            # Build B_sub for this output (noise input)
            B_sub = np.zeros((n_states_per_output, 1))
            if nc > 0:
                B_sub[:nc, 0] = MA_coeffs[i, :]

            # Assemble into full state space
            if ny == 1:
                # SISO case
                A = A_sub
                B = B_sub
                C = np.zeros((1, n_states_per_output))
                C[0, n_states_per_output - 1] = 1
            else:
                # MIMO case - build block-diagonal structure
                if i == 0:
                    A = A_sub
                    B = B_sub
                    C = np.zeros((1, n_states_per_output))
                    C[0, n_states_per_output - 1] = 1
                else:
                    # Expand A block-diagonally
                    A_block = np.zeros(
                        (A.shape[0] + A_sub.shape[0], A.shape[1] + A_sub.shape[1])
                    )
                    A_block[: A.shape[0], : A.shape[1]] = A
                    A_block[A.shape[0] :, A.shape[1] :] = A_sub
                    A = A_block

                    # Expand B
                    B_block = np.zeros((B.shape[0] + B_sub.shape[0], i + 1))
                    B_block[: B.shape[0], :i] = B
                    B_block[B.shape[0] :, i] = B_sub.flatten()
                    B = B_block

                    # Expand C - need to expand existing rows and add new row
                    C_expanded = np.zeros((C.shape[0], A.shape[1]))
                    C_expanded[:, : C.shape[1]] = C
                    # New row for output i - picks last state of new block
                    C_new = np.zeros((1, A.shape[1]))
                    C_new[0, A.shape[1] - 1] = 1
                    C = np.vstack([C_expanded, C_new])

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
