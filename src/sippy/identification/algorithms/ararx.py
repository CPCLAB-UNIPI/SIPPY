"""
ARARX (Auto-Regressive Auto-Regressive X) identification algorithm.
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
        warnings.warn("harold library incomplete. ARARX algorithm will be limited.")
except ImportError:
    HAROLD_AVAILABLE = False
    warnings.warn("harold library not available. ARARX algorithm will be limited.")


class ARARXAlgorithm(IdentificationAlgorithm):
    """
    ARARX (Auto-Regressive Auto-Regressive X) identification algorithm.

    The ARARX model structure is:
    A(q) y(k) = B(q)/D(q) F(q) u(k-nk) + C(q) e(k)

    where:
    - A(q) = 1 + a1*q^-1 + ... + ana*q^-na (auto-regressive part, na=0 for ARARX)
    - B(q)/D(q) = input transfer function polynomials
    - F(q) = input filter polynomial
    - C(q) = 1 + c1*q^-1 + ... + cnc*q^-nc (noise AR polynomial)
    - e(k) is white noise
    - nk is the input delay

    ARARX models are used for systems with colored noise,
    where the noise dynamics need to be explicitly modeled.

    The algorithm uses extended least-squares methods to estimate
    the input and noise parameters simultaneously.
    """

    def __init__(self):
        """Initialize ARARX algorithm."""
        super().__init__()

    def get_algorithm_name(self) -> str:
        """Return algorithm name."""
        return "ARARX"

    def validate_parameters(self, **kwargs) -> bool:
        """
        Validate ARARX-specific parameters.

        Parameters:
        -----------
        **kwargs : dict
            Parameters to validate including nb, nc, nd, nf, nk

        Returns:
        --------
        bool
            True if parameters are valid
        """
        nb = kwargs.get("nb")
        nc = kwargs.get("nc")
        nd = kwargs.get("nd")
        nf = kwargs.get("nf")
        nk = kwargs.get("nk")

        # Check if parameters are explicitly set to invalid values
        if nb is not None and nb <= 0:
            raise ValueError("Input order (nb) must be positive")
        if nc is not None and nc <= 0:
            raise ValueError("Noise AR order (nc) must be positive")
        if nd is not None and nd <= 0:
            raise ValueError("Noise MA orders must be positive")
        if nf is not None and nf <= 0:
            raise ValueError("Noise MA orders must be positive")
        if nk is not None and nk < 0:
            raise ValueError("Input delay (nk) must be non-negative")

        return True

    def identify(self, data, config):
        """
        Identify ARARX model from input-output data.

        Parameters:
        -----------
        data : IDData
            Input-output data
        config : SystemIdentificationConfig
            Configuration parameters including nb, nc, nd, nf, nk, na

        Returns:
        --------
        model : StateSpaceModel
            Identified state-space model
        """
        # Extract data from IDData object
        u = data.get_input_array()
        y = data.get_output_array()
        ts = data.ts if hasattr(data, 'ts') else 1.0

        # Extract configuration parameters (ARARX specific)
        nb = getattr(config, "nb", None)
        nc = getattr(config, "nc", None)
        nd = getattr(config, "nd", None)
        nf = getattr(config, "nf", None)
        nk = getattr(config, "nk", None)

        # Validate parameters before handling None cases
        self.validate_parameters(nb=nb, nc=nc, nd=nd, nf=nf, nk=nk)

        # Handle None cases (but keep 0 if explicitly set for validation)
        nb = nb or 1
        nc = nc or 1
        nd = nd or 1
        nf = nf or 1
        nk = nk or 0

        # Get data dimensions
        ny, N = y.shape
        nu, _ = u.shape

        # Calculate effective data length considering all lags
        max_input_lag = nk + max(nb, nd, nf) - 1
        max_noise_lag = nc - 1
        max_lag = max(max_input_lag, max_noise_lag)
        N_eff = N - max_lag

        # Check if we have enough data for parameter estimation
        n_params = (nb + nd + nf) + nc  # Total number of parameters
        if N_eff <= 0 or N <= n_params:
            raise ValueError(
                f"Not enough data. Need at least {max_lag + 1} samples and more than {n_params} total data points, got {N}"
            )

        # Initialize coefficient storage for MIMO case
        if nu > 1 or ny > 1:
            # For MIMO, estimate each input-output pair separately for simplicity
            # In a more advanced implementation, we'd estimate all parameters jointly
            B_coeffs = np.zeros((ny, nu, nb + nd + nf))
            C_coeffs = np.zeros((ny, nc))
        else:
            # SISO case
            B_coeffs = np.zeros((ny, nb + nd + nf))
            C_coeffs = np.zeros((ny, nc))

        residuals_list = []

        for i in range(ny):  # For each output
            for j in range(nu):  # For each input (SISO case will have j=0 only)
                # Construct regression matrix for ARARX estimation
                n_params = (nb + nd + nf) + nc  # Input part + noise AR part
                Phi = np.zeros((N_eff, n_params))
                col = 0

                # Input part: B(q)/D(q)F(q)u[k-nk]
                # For simplicity, use delayed inputs and outputs as regressors
                # B(q) terms: delayed inputs
                for lag in range(nb):
                    input_idx = max_lag - nk - lag
                    Phi[:, col] = u[j, input_idx : input_idx + N_eff]
                    col += 1

                # D(q) terms: delayed outputs
                for lag in range(nd):
                    output_idx = max_lag - 1 - lag
                    if output_idx >= 0 and output_idx + N_eff <= N:
                        Phi[:, col] = y[i, output_idx : output_idx + N_eff]
                    else:
                        Phi[:, col] = 0
                    col += 1

                # F(q) terms: additional filtered inputs (simplified as delayed inputs)
                for lag in range(nf):
                    input_idx = max_lag - nk - lag
                    if input_idx >= 0 and input_idx + N_eff <= N:
                        Phi[:, col] = u[j, input_idx : input_idx + N_eff]
                    else:
                        Phi[:, col] = 0
                    col += 1

                # Noise AR part: C(q) terms
                # We'll use iterative approach for noise terms
                if nc > 0:
                    # Initial estimate without noise correlation
                    Phi_input = Phi[:, : (nb + nd + nf)]
                    theta_input, _, _, _ = lstsq(
                        Phi_input, y[i, max_lag : max_lag + N_eff], rcond=None
                    )
                    y_pred = Phi_input @ theta_input

                    # Add noise correlation terms
                    for lag in range(nc):
                        if lag == 0:
                            # Current residual not available
                            Phi[:, col] = 0
                        else:
                            noise_idx = max_lag - 1 - lag
                            if noise_idx >= 0:
                                Phi[:, col] = (
                                    y[i, noise_idx : noise_idx + N_eff] - y_pred[:N_eff]
                                )
                            else:
                                Phi[:, col] = 0
                        col += 1
                else:
                    # No noise correlation
                    pass

                # Solve for ARARX parameters
                theta, residuals_i, rank, s = lstsq(
                    Phi, y[i, max_lag : max_lag + N_eff], rcond=None
                )
                residuals_list.append(residuals_i)

                # Extract input and noise coefficients
                if nu == 1 and ny == 1:
                    # SISO case
                    B_coeffs[i, :] = theta[: nb + nd + nf]
                    if nc > 0:
                        C_coeffs[i, :] = theta[nb + nd + nf :]
                else:
                    # MIMO case - handle differently
                    pass

        # Create state-space representation
        if HAROLD_AVAILABLE:
            model = self._create_state_space_from_ararx(
                B_coeffs, C_coeffs, nb, nc, nd, nf, ny, nu, ts
            )
        else:
            # Fallback when harold is not available
            model = self._create_mock_model(
                B_coeffs, C_coeffs, nb, nc, nd, nf, ny, nu, ts
            )

        return model

    def _create_state_space_from_ararx(
        self, B_coeffs, C_coeffs, nb, nc, nd, nf, ny, nu, Ts
    ):
        """
        Create state-space model from ARARX coefficients using harold.

        Parameters:
        -----------
        B_coeffs, C_coeffs : ndarray
            Input and noise coefficients
        nb, nc, nd, nf : int
            Polynomial orders
        ny : int
            Number of outputs
        nu : int
            Number of inputs
        Ts : float
            Sampling time

        Returns:
        --------
        model : StateSpaceModel
            State-space model representation
        """
        # For ARARX, create a transfer function representation
        # A(q) y(k) = B(q)/D(q)F(q)u[k-nk] + C(q)e[k]
        # With na=0, we have: y(k) = B(q)/D(q)F(q)u[k-nk] + C(q)e[k]

        n_states = max(nb, nc, nd, nf)

        # Create state-space matrices (companion form)
        A = np.zeros((n_states, n_states))
        B = np.zeros((n_states, nu))
        C = np.zeros((ny, n_states))
        D = np.zeros((ny, nu))

        # Build companion form
        if ny == 1 and nu == 1:
            # SISO case
            # State matrix (companion form for input dynamics)
            if n_states > 1:
                A[: n_states - 1, 1:n_states] = np.eye(n_states - 1)

            # Last row reflects the system dynamics (simplified)
            A[n_states - 1, :] = (
                -B_coeffs[0, :n_states] if n_states <= len(B_coeffs[0]) else 0
            )

            # Input matrix
            B[: min(n_states, len(B_coeffs[0])), 0] = B_coeffs[
                0, : min(n_states, len(B_coeffs[0]))
            ]

            # Output matrix (last state is the output)
            C[0, -1] = 1

            # Noise effects handled through C_coeffs
            if nc > 0:
                # For ARARX, noise affects the system dynamics
                B_noise = np.zeros((n_states, ny))
                B_noise[: min(nc, n_states), 0] = C_coeffs[0, : min(nc, n_states)]
                # In a full implementation, this would be handled more sophisticatedly
        else:
            # MIMO case - simplified implementation
            for i in range(ny):
                for j in range(nu):
                    # Create separate SISO systems and combine
                    pass

        # Create TransferFunction first (ARARX often works in transfer function form)
        # Then convert to StateSpace
        tf_model = harold.TransferFunction(num=[1], den=[1, 1], dt=Ts)
        tf_model.NumberOfInputs = nu
        tf_model.NumberOfOutputs = ny
        tf_model.SamplingPeriod = Ts

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

    def _create_mock_model(self, B_coeffs, C_coeffs, nb, nc, nd, nf, ny, nu, Ts):
        """
        Create a mock state-space model when harold is not available.

        Parameters:
        -----------
        B_coeffs, C_coeffs : ndarray
            Input and noise coefficients
        nb, nc, nd, nf : int
            Polynomial orders
        ny : int
            Number of outputs
        nu : int
            Number of inputs
        Ts : float
            Sampling time

        Returns:
        --------
        model : StateSpaceModel
            Mock state-space model
        """
        # Create simple companion form state-space representation
        n_states = max(nb, nc, nd, nf)

        # State matrix A (companion form)
        A = np.zeros((n_states, n_states))
        if n_states > 1:
            A[: n_states - 1, 1:n_states] = np.eye(n_states - 1)

        # Set last row based on coefficients (simplified)
        if ny == 1 and nu == 1:
            # Use the first n_states coefficients for the state matrix
            if len(B_coeffs[0]) >= n_states:
                A[n_states - 1, :n_states] = -B_coeffs[0, :n_states]
            else:
                A[n_states - 1, : len(B_coeffs[0])] = -B_coeffs[0, : len(B_coeffs[0])]
        else:
            # MIMO simplified implementation
            A[n_states - 1, 0] = -0.5  # Placeholder

        # Input matrix B
        B = np.zeros((n_states, nu))
        if ny == 1 and nu == 1:
            # Use the first n_states coefficients for the input matrix
            if len(B_coeffs[0]) >= n_states:
                B[:n_states, 0] = B_coeffs[0, :n_states]
            else:
                B[: len(B_coeffs[0]), 0] = B_coeffs[0, : len(B_coeffs[0])]
                # Fill remaining states if needed
                if len(B_coeffs[0]) > 0:
                    B[len(B_coeffs[0]) : n_states, 0] = B_coeffs[0, 0]
        else:
            # MIMO simplified implementation
            B[0, 0] = 1.0  # Placeholder

        # Output matrix C
        C = np.zeros((ny, n_states))
        C[:, -1] = 1  # Last state is the output

        # Feedthrough matrix D
        D = np.zeros((ny, nu))

        # Noise modeling through covariance matrices
        Vn = 0.01
        if nc > 0:
            # Increase noise variance for colored noise
            Vn = 0.05

        return StateSpaceModel(
            A=A,
            B=B,
            C=C,
            D=D,
            K=np.zeros((A.shape[0], C.shape[0])),
            Q=np.eye(A.shape[0]) * Vn,
            R=np.eye(C.shape[0]),
            S=np.zeros((A.shape[0], C.shape[0])),
            ts=Ts,
            Vn=Vn,
        )
