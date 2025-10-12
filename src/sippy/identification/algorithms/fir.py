"""
FIR (Finite Impulse Response) identification algorithm.
"""

import warnings

import numpy as np
from numpy.linalg import lstsq

from ..base import IdentificationAlgorithm, StateSpaceModel

# Import compiled utilities for performance
try:
    from ...utils.compiled_utils import (
        NUMBA_AVAILABLE,
        create_regression_matrix_fir_compiled,
    )
except ImportError:
    create_regression_matrix_fir_compiled = None
    NUMBA_AVAILABLE = False

try:
    import harold

    # Check if harold has the required components
    if hasattr(harold, "State"):
        HAROLD_AVAILABLE = True
    else:
        HAROLD_AVAILABLE = False
        warnings.warn("harold library incomplete. FIR algorithm will be limited.")
except ImportError:
    HAROLD_AVAILABLE = False
    warnings.warn("harold library not available. FIR algorithm will be limited.")


class FIRAlgorithm(IdentificationAlgorithm):
    """
    FIR (Finite Impulse Response) identification algorithm.

    The FIR model structure is:
    y(k) = b1*u(k-nk) + b2*u(k-nk-1) + ... + bnb*u(k-nk-nb+1) + e(k)

    where:
    - b1, b2, ..., bnb are the FIR coefficients
    - nb is the number of FIR coefficients
    - nk is the input delay (number of samples)
    - e(k) is white noise

    The algorithm uses least-squares regression to estimate the FIR coefficients.
    """

    def __init__(self):
        """Initialize FIR algorithm."""
        super().__init__()

    def get_algorithm_name(self) -> str:
        """Return algorithm name."""
        return "FIR"

    def validate_parameters(self, **kwargs) -> bool:
        """
        Validate FIR-specific parameters.

        Parameters:
        -----------
        **kwargs : dict
            Parameters to validate including nb, nk

        Returns:
        --------
        bool
            True if parameters are valid
        """
        nb = kwargs.get("nb", 1)
        nk = kwargs.get("nk", 1)

        if nb <= 0:
            raise ValueError("Number of FIR coefficients must be positive")
        if nk < 0:
            raise ValueError("Input delay (nk) must be non-negative")

        return True

    def identify(self, data, config):
        """
        Identify FIR model from input-output data.

        Parameters:
        -----------
        data : IDData
            Input-output data
        config : SystemIdentificationConfig
            Configuration parameters including nb, nk

        Returns:
        --------
        model : StateSpaceModel
            Identified state-space model
        """
        # Extract data from IDData object
        u = data.get_input_array()
        y = data.get_output_array()

        # Extract configuration parameters (FIR specific)
        nb = getattr(config, "nb", 1)
        nk = getattr(config, "nk", 1)

        # Validate parameters
        self.validate_parameters(nb=nb, nk=nk)

        # Get data dimensions
        ny, N = y.shape
        nu, _ = u.shape

        # Calculate effective data length
        N_eff = N - nb - nk + 1

        if N_eff <= 0:
            raise ValueError(
                f"Not enough data points. Need at least {nb + nk} samples, got {N}"
            )

        # MIMO case - construct output-specific regression matrices
        fir_coeffs = np.zeros((ny, nb * nu))
        residuals_list = []

        for i in range(ny):
            # For output i, construct regression matrix for this output
            Phi_i = np.zeros((N_eff, nb * nu))
            col = 0

            # Input part: all lagged inputs affect this output
            for lag in range(nb):
                for j in range(nu):
                    delay_idx = N_eff + nk - 1 - lag
                    if delay_idx >= 0 and delay_idx + N_eff <= N:
                        Phi_i[:, col] = u[
                            j, delay_idx - N_eff + 1 : delay_idx - N_eff + N_eff + 1
                        ]
                    else:
                        Phi_i[:, col] = 0
                    col += 1

            # Solve for output i
            theta_i, residuals_i, rank_i, s_i = lstsq(
                Phi_i, y[i, nk + nb - 1 : nk + nb - 1 + N_eff], rcond=None
            )
            residuals_list.append(residuals_i)

            # Extract input coefficients for output i
            fir_coeffs[i, :] = theta_i

        # Compute one-step-ahead predictions (Yid) for identification data
        N_eff_yid = N - nb - nk + 1
        Yid = np.zeros_like(y)
        Yid[:, :nk + nb - 1] = y[:, :nk + nb - 1]  # Copy initial values

        # Compute predictions for each output
        for i in range(ny):
            Phi_i = np.zeros((N_eff_yid, nb * nu))
            col = 0
            for lag in range(nb):
                for j in range(nu):
                    delay_idx = N_eff_yid + nk - 1 - lag
                    if delay_idx >= 0 and delay_idx + N_eff_yid <= N:
                        Phi_i[:, col] = u[j, delay_idx - N_eff_yid + 1 : delay_idx - N_eff_yid + N_eff_yid + 1]
                    col += 1

            Yid[i, nk + nb - 1:] = np.dot(Phi_i, fir_coeffs[i, :]).flatten()

        # Create G_tf and H_tf transfer functions
        G_tf, H_tf = self._create_transfer_functions_fir(
            fir_coeffs, nb, nk, ny, nu, data.sample_time
        )

        # Create state-space representation
        if HAROLD_AVAILABLE:
            model = self._create_state_space_from_fir(
                fir_coeffs, nb, nk, ny, nu, data.sample_time
            )
        else:
            # Fallback when harold is not available
            model = self._create_mock_model(
                fir_coeffs, nb, nk, ny, nu, data.sample_time
            )

        # Attach transfer functions and predictions to model
        model.G_tf = G_tf
        model.H_tf = H_tf
        model.Yid = Yid

        return model

    def _create_regression_matrix(self, u, y, nb, nk, ny, nu, N):
        """
        Create regression matrix Phi and output matrix y for least squares.

        This function automatically uses the Numba-compiled version when available
        for improved performance.

        Parameters:
        -----------
        u, y : ndarray
            Input and output data
        nb, nk : int
            Model coefficients count and delay
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
        if NUMBA_AVAILABLE and create_regression_matrix_fir_compiled is not None:
            return create_regression_matrix_fir_compiled(u, y, nb, nk, ny, nu, N)
        else:
            # Fallback to original implementation
            # Determine effective data length
            max_lag = nb + nk - 1
            N_eff = N - max_lag

            if N_eff <= 0:
                raise ValueError(
                    f"Not enough data points. Need at least {max_lag + 1} samples, got {N}"
                )

            # Initialize regression matrix
            n_params = nb * ny * nu
            Phi = np.zeros((N_eff, n_params))

            # Fill FIR part (lagged inputs)
            for k in range(nb):
                for i in range(nu):
                    # For MIMO, each input affects all outputs
                    for j in range(ny):
                        col_idx = k * ny * nu + i * ny + j
                        delay_idx = max_lag - 1 - k
                        if delay_idx >= 0 and delay_idx + N_eff <= N:
                            Phi[:, col_idx] = u[i, delay_idx : delay_idx + N_eff]

            # Output matrix
            y_matrix = y[:, max_lag:N]

            return Phi, y_matrix

    def _create_transfer_functions_fir(self, fir_coeffs, nb, nk, ny, nu, Ts):
        """
        Create G_tf and H_tf transfer functions for FIR.

        For FIR: G_tf = B(q) (FIR polynomial), H_tf = 1 (white noise only).

        Parameters:
        -----------
        fir_coeffs : ndarray
            FIR coefficient array (ny x nb*nu)
        nb, nk : int
            Number of FIR coefficients and delay
        ny, nu : int
            Number of outputs and inputs
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

            # Create G(q) = B(q) - FIR transfer function with delay
            # For FIR, numerator is the FIR coefficients, denominator is 1
            NUM_G = np.zeros(nb + nk)
            NUM_G[nk:nk + nb] = fir_coeffs[0, :nb] if ny == 1 else fir_coeffs[0, :nb]

            DEN_G = np.array([1.0])  # FIR has unity denominator

            G_tf = harold.Transfer(NUM_G, DEN_G, dt=Ts)

            # H(q) = 1 - FIR has no noise model (white noise only)
            H_tf = harold.Transfer([1.0], [1.0], dt=Ts)

            return G_tf, H_tf
        except Exception as e:
            warnings.warn(f"Failed to create FIR transfer functions with harold: {e}")
            return None, None

    def _create_state_space_from_fir(self, fir_coeffs, nb, nk, ny, nu, Ts):
        """
        Create state-space model from FIR coefficients using harold.

        Parameters:
        -----------
        fir_coeffs : ndarray
            FIR coefficients array
        nb, nk : int
            Number of coefficients and input delay
        ny, nu : int
            Number of outputs and inputs
        Ts : float
            Sampling time

        Returns:
        --------
        model : StateSpaceModel
            State-space model representation
        """
        # Create state matrices for FIR model
        n_states = nb

        # For each input-output pair, we have nb coefficients
        # State representation of FIR using delay chain
        A = np.zeros((n_states, n_states))
        if n_states > 1:
            for i in range(n_states - 1):
                A[i, i + 1] = 1  # Shift register

        B = np.zeros((n_states, nu))
        # Input enters at the beginning of the shift register
        B[0, :] = np.ones(nu) if nk == 0 else np.zeros(nu)

        C = np.zeros((ny, n_states))
        D = np.zeros((ny, nu))

        # Fill C and D matrices from FIR coefficients
        for i in range(ny):  # For each output
            for j in range(nu):  # For each input
                for k in range(nb):
                    if k < fir_coeffs.shape[1]:
                        if k == 0:
                            # Direct term
                            if nk == 0:
                                D[i, j] = fir_coeffs[i, k * nu + j]
                            else:
                                D[i, j] = 0
                                if k == nk - 1:
                                    D[i, j] = fir_coeffs[i, k * nu + j]
                        else:
                            # Feed-through through states
                            C[i, k - 1] = (
                                fir_coeffs[i, k * nu + j]
                                if k * nu + j < fir_coeffs.shape[1]
                                else 0
                            )

        # Create harold State object
        ss_model = harold.State(A, B, C, D, dt=Ts)

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

    def _create_mock_model(self, fir_coeffs, nb, nk, ny, nu, Ts):
        """
        Create a mock state-space model when harold is not available.

        Parameters:
        -----------
        fir_coeffs : ndarray
            FIR coefficients array
        nb, nk : int
            Number of coefficients and input delay
        ny, nu : int
            Number of outputs and inputs
        Ts : float
            Sampling time

        Returns:
        --------
        model : StateSpaceModel
            Mock state-space model
        """
        # Create simple delay chain state-space representation
        n_states = nb

        # State matrix (shift register)
        A = np.zeros((n_states, n_states))
        if n_states > 1:
            for i in range(n_states - 1):
                A[i, i + 1] = 1

        # Input matrix (input enters at the beginning)
        B = np.zeros((n_states, nu))
        B[0, :] = np.ones(nu)

        # Output matrix (weighted sum of states + direct feedthrough)
        C = np.zeros((ny, n_states))
        D = np.zeros((ny, nu))

        # Fill C and D from FIR coefficients
        for i in range(ny):  # For each output
            for j in range(nu):  # For each input
                # Simple mapping - this is a simplified version
                for k in range(nb):
                    if k < fir_coeffs.shape[1]:
                        if k == 0:
                            D[i, j] = fir_coeffs[i, k * nu + j]
                        else:
                            C[i, k - 1] = fir_coeffs[i, k * nu + j]

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
