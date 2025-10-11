"""
ARX (AutoRegressive with eXogenous inputs) identification algorithm.
"""
import warnings

import numpy as np
from numpy.linalg import lstsq

from ..base import IdentificationAlgorithm, StateSpaceModel

try:
    import harold
    # Check if harold has the required components
    if hasattr(harold, 'StateSpace'):
        HAROLD_AVAILABLE = True
    else:
        HAROLD_AVAILABLE = False
        warnings.warn("harold library incomplete. ARX algorithm will be limited.")
except ImportError:
    HAROLD_AVAILABLE = False
    warnings.warn("harold library not available. ARX algorithm will be limited.")


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
        na = kwargs.get('na', 1)
        nb = kwargs.get('nb', 1)
        nk = kwargs.get('nk', 1)

        if na <= 0:
            raise ValueError("AR order (na) must be positive")
        if nb <= 0:
            raise ValueError("X order (nb) must be positive")
        if nk < 0:
            raise ValueError("Input delay (nk) must be non-negative")

        return True

    def identify(self, data, config):
        """
        Identify ARX model from input-output data.

        Parameters:
        -----------
        data : IDData
            Input-output data
        config : SystemIdentificationConfig
            Configuration parameters including na, nb, nk

        Returns:
        --------
        model : StateSpaceModel
            Identified state-space model
        """
        # Extract data from IDData object
        u = data.get_input_array()
        y = data.get_output_array()

        # Extract configuration parameters (ARX specific)
        na = getattr(config, 'na', 1)
        nb = getattr(config, 'nb', 1)
        nk = getattr(config, 'nk', 1)

        # Validate parameters
        self.validate_parameters(na=na, nb=nb, nk=nk)

        # Get data dimensions
        ny, N = y.shape
        nu, _ = u.shape

        # Create regression matrix
        Phi, y_matrix = self._create_regression_matrix(u, y, na, nb, nk, ny, nu, N)

        # Estimate parameters using least squares
        theta, residuals, rank, s = lstsq(Phi, y_matrix.T.flatten(), rcond=None)

        # Reshape parameters into matrices
        A_coeffs = theta[:na * ny].reshape(ny, na)
        B_coeffs = theta[na * ny:].reshape(ny, nb * nu)

        # Create state-space representation
        if HAROLD_AVAILABLE:
            model = self._create_transfer_function(A_coeffs, B_coeffs, na, nb, nk, ny, nu, data.sample_time)
        else:
            # Fallback when harold is not available
            model = self._create_mock_model(A_coeffs, B_coeffs, na, nb, nk, ny, nu, data.sample_time)

        return model

    def _create_regression_matrix(self, u, y, na, nb, nk, ny, nu, N):
        """
        Create regression matrix Phi and output matrix y for least squares.

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
        # Determine effective data length
        max_lag = max(na, nb + nk - 1)
        N_eff = N - max_lag

        if N_eff <= 0:
            raise ValueError(f"Not enough data points. Need at least {max_lag + 1} samples, got {N}")

        # Initialize regression matrix
        n_params = na * ny + nb * ny * nu
        Phi = np.zeros((N_eff, n_params))

        # Fill AR part (lagged outputs)
        for i in range(na):
            for j in range(ny):
                col_idx = i * ny + j
                Phi[:, col_idx] = y[j, max_lag - 1 - i : max_lag - 1 - i + N_eff]

        # Fill X part (lagged inputs)
        for k in range(nb):
            for i in range(nu):
                # For MIMO, each input affects all outputs
                for j in range(ny):
                    col_idx = na * ny + k * ny * nu + i * ny + j
                    delay_idx = max_lag - 1 - (k + nk - 1)
                    if delay_idx >= 0 and delay_idx + N_eff <= N:
                        Phi[:, col_idx] = u[i, delay_idx : delay_idx + N_eff]

        # Output matrix
        y_matrix = y[:, max_lag : N]

        return Phi, y_matrix

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
            num_coeffs = np.zeros(nb + nk)
            num_coeffs[nk:] = B_coeffs[0, :]

            # Create transfer function
            tf = harold.TransferFunction(num_coeffs, den_coeffs, dt=Ts)

            # Convert to state-space
            ss_model = harold.undiscretize(tf, method='backward euler')
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
                    A[i*na:(i+1)*na-1, i*na+1:(i+1)*na] = np.eye(na-1)
                if na > 0:
                    A[(i+1)*na-1, i*na:(i+1)*na] = -A_coeffs[i, :]

                # B matrix
                if nu > 0:
                    B[(i+1)*na-1, :] = B_coeffs[i, :].reshape(-1, nu)[:nu] if nb == 1 else B_coeffs[i, :nu]

                # C matrix
                C[i, (i+1)*na-1] = 1

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
            Vn=0.01
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
            A[-ny:, :] = -A_coeffs.T  # AR coefficients in last rows

        # Input matrix B
        B = np.zeros((n_states, nu))
        if nb > 0:
            B[-ny:, :] = B_coeffs[:, 0, :].T if len(B_coeffs.shape) > 2 else B_coeffs.T

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
            Vn=0.01
        )
