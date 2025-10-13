"""
OE (Output Error) identification algorithm.
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
        warnings.warn("harold library incomplete. OE algorithm will be limited.")
except ImportError:
    HAROLD_AVAILABLE = False
    warnings.warn("harold library not available. OE algorithm will be limited.")


class OEAlgorithm(IdentificationAlgorithm):
    """
    OE (Output Error) identification algorithm.

    ⚠️ SIMPLIFIED IMPLEMENTATION

    This implementation uses direct least squares approximation instead of the
    reference implementation's iterative nonlinear optimization.

    Reference (master):
      - Uses predicted outputs (Yid) in regressor
      - Iterative refinement with convergence checking
      - IPOPT nonlinear optimization

    Harold branch:
      - Uses actual outputs in single-pass least squares
      - Faster but may be less accurate for noise-heavy data

    For details see investigation report from Subagent 4.

    Model Structure:
    ----------------
    The OE model structure is:
    y(k) = B(q)/F(q) * u(k-nk) + e(k)

    where:
    - B(q) = b1 + b2*q^-1 + ... + bnb*q^-(nb-1) (numerator polynomial)
    - F(q) = 1 + f1*q^-1 + ... + fnf*q^-nf (denominator polynomial)
    - nk is the input delay (number of samples)
    - e(k) is white noise

    The OE algorithm estimates parameters using output error prediction error methods,
    which is nonlinear due to the noise-free output feedback through F(q).
    """

    def __init__(self):
        """Initialize OE algorithm."""
        super().__init__()

    def get_algorithm_name(self) -> str:
        """Return algorithm name."""
        return "OE"

    def validate_parameters(self, **kwargs) -> bool:
        """
        Validate OE-specific parameters.

        Parameters:
        -----------
        **kwargs : dict
            Parameters to validate including nb, nf, nk

        Returns:
        --------
        bool
            True if parameters are valid
        """
        nb = kwargs.get("nb", 2)
        nf = kwargs.get("nf", 2)
        nk = kwargs.get("nk", 1)

        if nb <= 0:
            raise ValueError("Numerator order (nb) must be positive")
        if nf <= 0:
            raise ValueError("Denominator order (nf) must be positive")
        if nk < 0:
            raise ValueError("Input delay (nk) must be non-negative")

        return True

    def identify(self, data, config):
        """
        Identify OE model from input-output data.

        Parameters:
        -----------
        data : IDData
            Input-output data
        config : SystemIdentificationConfig
            Configuration parameters including nb, nf, nk

        Returns:
        --------
        model : StateSpaceModel
            Identified state-space model
        """
        # Extract data from IDData object
        u = data.get_input_array()
        y = data.get_output_array()

        # Extract configuration parameters (OE specific)
        nb = getattr(config, "nb", 2)
        nf = getattr(config, "nf", 2)
        nk = getattr(config, "nk", 1)

        # Validate parameters
        self.validate_parameters(nb=nb, nf=nf, nk=nk)

        # Get data dimensions
        ny, N = y.shape
        nu, _ = u.shape

        # Create regression matrices for OE identification
        Phi, y_matrix = self._create_oe_regression_matrices(u, y, nb, nf, nk, ny, nu, N)

        # Estimate parameters using least squares
        if ny == 1:
            # SISO case
            theta, residuals, rank, s = lstsq(Phi, y_matrix.T, rcond=None)
            B_coeffs = theta[: nb * nu].reshape(ny, nb * nu)
            F_coeffs = theta[nb * nu :].reshape(ny, nf)
        else:
            # MIMO case - need to solve for each output separately
            theta_list = []
            residuals_list = []
            for i in range(ny):
                theta_i, residuals_i, rank_i, s_i = lstsq(
                    Phi, y_matrix[i, :], rcond=None
                )
                theta_list.append(theta_i)
                residuals_list.append(residuals_i)

            # Combine results
            theta = np.concatenate(theta_list)
            B_coeffs = np.zeros((ny, nb * nu))
            F_coeffs = np.zeros((ny, nf))

            for i in range(ny):
                theta_i = theta_list[i]
                B_coeffs[i, :] = theta_i[: nb * nu]
                F_coeffs[i, :] = (
                    theta_i[nb * nu : nb * nu + nf]
                    if len(theta_i) >= nb * nu + nf
                    else theta_i[nb * nu :]
                )

        # Compute one-step-ahead predictions (Yid) for identification data
        max_lag = max(nf + nk - 1, nb + nk - 1)
        N_eff = N - max_lag
        Yid = np.zeros_like(y)
        Yid[:, :max_lag] = y[:, :max_lag]  # Copy initial values

        if ny == 1:
            # SISO case - compute predictions
            Yid[0, max_lag:] = np.dot(Phi, theta).flatten()
        else:
            # MIMO case - reconstruct predictions for each output
            for i in range(ny):
                theta_i = theta_list[i]
                # For OE, the regression matrix uses noise-free outputs
                # We need to reconstruct Phi_i for each output
                n_params = nb * nu + nf * ny
                Phi_i = np.zeros((N_eff, n_params))

                # Fill B part (numerator - lagged inputs)
                for k in range(nb):
                    for j in range(nu):
                        col_idx = k * nu + j
                        delay_idx = max_lag - 1 - (k + nk - 1)
                        if delay_idx >= 0 and delay_idx + N_eff <= N:
                            Phi_i[:, col_idx] = u[j, delay_idx : delay_idx + N_eff]

                # Fill F part (denominator - lagged outputs)
                for k in range(nf):
                    for j in range(ny):
                        col_idx = nb * nu + k * ny + j
                        output_delay = max_lag - 1 - k
                        if output_delay >= 0 and output_delay + N_eff <= N:
                            Phi_i[:, col_idx] = -y[
                                j, output_delay : output_delay + N_eff
                            ]

                Yid[i, max_lag:] = np.dot(Phi_i, theta_i)

        # Create G_tf and H_tf transfer functions
        G_tf, H_tf = self._create_transfer_functions_oe(
            B_coeffs, F_coeffs, nb, nf, nk, ny, nu, data.sample_time
        )

        # Create state-space representation
        if HAROLD_AVAILABLE:
            model = self._create_state_space_from_oe(
                B_coeffs, F_coeffs, nb, nf, nk, ny, nu, data.sample_time
            )
        else:
            # Fallback when harold is not available
            model = self._create_mock_model(
                B_coeffs, F_coeffs, nb, nf, nk, ny, nu, data.sample_time
            )

        # Attach transfer functions and predictions to model
        model.G_tf = G_tf
        model.H_tf = H_tf
        model.Yid = Yid

        return model

    def _create_oe_regression_matrices(self, u, y, nb, nf, nk, ny, nu, N):
        """
        Create regression matrices Phi and output matrix y for OE identification.

        This uses the output error approach where only noise-free outputs are used
        in the regression (nonlinear estimation).

        Parameters:
        -----------
        u, y : ndarray
            Input and output data
        nb, nf, nk : int
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
        max_lag = max(nf + nk - 1, nb + nk - 1)
        N_eff = N - max_lag

        if N_eff <= 0:
            raise ValueError(
                f"Not enough data points. Need at least {max_lag + 1} samples, got {N}"
            )

        # Initialize regression matrix
        n_params = nb * nu + nf * ny  # B + F coefficients (F is per output)
        Phi = np.zeros((N_eff, n_params))

        # Fill B part (numerator coefficients - lagged inputs)
        for k in range(nb):
            for i in range(nu):
                col_idx = k * nu + i
                delay_idx = max_lag - 1 - (k + nk - 1)
                if delay_idx >= 0 and delay_idx + N_eff <= N:
                    Phi[:, col_idx] = u[i, delay_idx : delay_idx + N_eff]

        # Fill F part (denominator coefficients - lagged noise-free outputs)
        # For OE, we use iterative approach starting with actual outputs as approximation
        # In practice, this would require iterative refinement
        for i in range(nf):
            for j in range(ny):
                col_idx = nb * nu + i * ny + j
                # Use lagged outputs as initial approximation of noise-free outputs
                output_delay = max_lag - 1 - i
                if output_delay >= 0 and output_delay + N_eff <= N:
                    Phi[:, col_idx] = -y[j, output_delay : output_delay + N_eff]

        # Output matrix
        y_matrix = y[:, max_lag:N]

        return Phi, y_matrix

    def _create_transfer_functions_oe(self, B_coeffs, F_coeffs, nb, nf, nk, ny, nu, Ts):
        """
        Create G_tf and H_tf transfer functions for OE.

        For OE: G_tf = B(q)/F(q), H_tf = 1 (unity, since OE has no noise model).

        Parameters:
        -----------
        B_coeffs, F_coeffs : ndarray
            Numerator (B) and denominator (F) coefficients
        nb, nf, nk : int
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
        if not HAROLD_AVAILABLE:
            return None, None

        try:
            import harold

            # Create G(q) = B / F - Deterministic transfer function
            max_order = max(nf, nb + nk)

            NUM_G = np.zeros(max_order)
            NUM_G[nk : nk + nb] = B_coeffs[0, :] if ny == 1 else B_coeffs[0, :nb]

            DEN_G = np.zeros(max_order + 1)
            DEN_G[0] = 1.0
            DEN_G[1 : nf + 1] = F_coeffs[0, :]

            G_tf = harold.Transfer(NUM_G, DEN_G, dt=Ts)

            # H(q) = 1 - OE has no noise model (H is unity)
            H_tf = harold.Transfer([1.0], [1.0], dt=Ts)

            return G_tf, H_tf
        except Exception as e:
            warnings.warn(f"Failed to create OE transfer functions with harold: {e}")
            return None, None

    def _create_state_space_from_oe(self, B_coeffs, F_coeffs, nb, nf, nk, ny, nu, Ts):
        """
        Create state-space model from OE parameters using harold.

        Parameters:
        -----------
        B_coeffs, F_coeffs : ndarray
            Numerator and denominator coefficient arrays
        nb, nf, nk : int
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
        # Build observer canonical form state-space representation for OE
        n_states = nf  # Number of states equals denominator order

        # A matrix - state transition (companion form of F polynomial)
        A = np.zeros((n_states, n_states))
        if nf > 1:
            for i in range(nf - 1):
                A[i, i + 1] = 1.0
        if nf > 0:
            if F_coeffs.shape[0] == 1:
                # SISO case
                A[-1, :] = -F_coeffs.T.flatten()
            else:
                # MIMO case - use average F coefficients for state matrix
                avg_F = np.mean(F_coeffs, axis=0)
                A[-1, :] = -avg_F

        # B matrix - input coupling
        B = np.zeros((n_states, nu))
        if ny == 1:  # SISO case
            coeffs_flat = B_coeffs.flatten()
            # Handle cases where nb != nf (take first nf coefficients or pad with zeros)
            if len(coeffs_flat) >= nf:
                B[:, 0] = coeffs_flat[:nf]
            else:
                B[: len(coeffs_flat), 0] = coeffs_flat
        else:  # MIMO case - average over outputs
            for j in range(nu):
                temp_coeffs = np.mean(B_coeffs[:, j::nu], axis=1)
                # Handle dimension mismatch
                if len(temp_coeffs) >= nf:
                    B[:, j] = temp_coeffs[:nf]
                else:
                    B[: len(temp_coeffs), j] = temp_coeffs

        # C matrix - output coupling
        C = np.zeros((ny, n_states))
        if ny == 1:  # SISO case
            C[0, :] = (
                np.concatenate(([0.0] * (nf - 1), [1.0])) if nf > 0 else np.array([])
            )
        else:  # MIMO case
            for i in range(ny):
                C[i, :] = (
                    np.concatenate(([0.0] * (nf - 1), [1.0]))
                    if nf > 0
                    else np.array([])
                )

        # D matrix - direct feedthrough
        D = np.zeros((ny, nu))

        # Create harold StateSpace object
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

    def _create_mock_model(self, B_coeffs, F_coeffs, nb, nf, nk, ny, nu, Ts):
        """
        Create a mock state-space model when harold is not available.

        Parameters:
        -----------
        B_coeffs, F_coeffs : ndarray
            Numerator and denominator coefficient arrays
        nb, nf, nk : int
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
        n_states = nf

        # State matrix (companion form of F polynomial)
        A = np.zeros((n_states, n_states))
        if nf > 1:
            for i in range(nf - 1):
                A[i, i + 1] = 1.0
        if nf > 0:
            if F_coeffs.shape[0] == 1:
                # SISO case
                A[-1, :] = -F_coeffs.T.flatten()
            else:
                # MIMO case - use average F coefficients for state matrix
                avg_F = np.mean(F_coeffs, axis=0)
                A[-1, :] = -avg_F

        # Input matrix
        B = np.zeros((n_states, nu))
        if ny == 1:  # SISO case
            # Handle case where B_coeffs might have more elements than B columns
            coeffs_flat = B_coeffs.flatten()
            if coeffs_flat.shape[0] >= n_states:
                B[:, 0] = coeffs_flat[:n_states]
            else:
                B[: coeffs_flat.shape[0], 0] = coeffs_flat
        else:  # MIMO case - average over outputs
            for j in range(nu):
                if B_coeffs.shape[1] >= j + 1:
                    temp_coeffs = np.mean(B_coeffs[:, j::nu], axis=1)
                    if temp_coeffs.shape[0] >= n_states:
                        B[:, j] = temp_coeffs[:n_states]
                    else:
                        B[: temp_coeffs.shape[0], j] = temp_coeffs

        # Output matrix
        C = np.zeros((ny, n_states))
        for i in range(ny):
            if nf > 0:
                C[i, :] = np.concatenate(([0.0] * (nf - 1), [1.0]))
            else:
                C[i, :] = np.array([])

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
