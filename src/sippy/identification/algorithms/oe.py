"""
OE (Output Error) identification algorithm.
"""

import warnings
from typing import TYPE_CHECKING, Optional

import numpy as np
from numpy.linalg import lstsq

from ..base import IdentificationAlgorithm, StateSpaceModel

if TYPE_CHECKING:
    from ..iddata import IDData

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

# Check for CasADi availability for NLP-based identification
try:
    import casadi  # noqa: F401

    CASADI_AVAILABLE = True
except ImportError:
    CASADI_AVAILABLE = False


class OEAlgorithm(IdentificationAlgorithm):
    """
    OE (Output Error) identification algorithm.

    Implements two identification methods:

    1. **NLP Method** (CasADi + IPOPT) - DEFAULT when available:
       - Uses predicted outputs (Yid) in regressor (iterative, nonlinear)
       - Matches master branch reference implementation
       - Decision variables: [b, f, Yid] coefficients + auxiliary time series
       - Objective: minimize ||Y - Yid||^2
       - Equality constraint: symbolic Yid - optimization Yid = 0
       - Optional stability constraints via companion matrix norms
       - Exact maximum likelihood estimates

    2. **Simplified Method** (Direct LS) - Fallback when CasADi unavailable:
       - Uses actual outputs in single-pass least squares
       - 30-100x faster but may be less accurate for noise-heavy data
       - Approximation of true OE solution

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

    def identify(
        self,
        y: Optional[np.ndarray] = None,
        u: Optional[np.ndarray] = None,
        iddata: Optional["IDData"] = None,
        **kwargs,
    ) -> StateSpaceModel:
        """
        Identify OE model from input-output data.

        Automatically selects NLP method (CasADi + IPOPT) if available,
        otherwise falls back to simplified direct least squares method.

        Parameters:
        -----------
        y : np.ndarray, optional
            Output data (outputs x time_steps)
        u : np.ndarray, optional
            Input data (inputs x time_steps)
        iddata : IDData, optional
            Input-output data container
        **kwargs : dict
            Configuration parameters including:
            - nb: int, numerator order
            - nf: int, denominator order
            - nk: int, input delay
            - tsample: float, sampling time
            - max_iterations: int, IPOPT max iterations (default 200)
            - stability_constraint: bool, enable stability constraints (default False)
            - stability_margin: float, stability margin for constraints (default 1.0)

        Returns:
        --------
        model : StateSpaceModel
            Identified state-space model with G_tf, H_tf, Yid attributes
        """
        # Backward compatibility: detect old API (data, config) vs new API (y, u, **kwargs)
        from ..base import SystemIdentificationConfig
        from ..iddata import IDData as IDDataClass

        if y is not None and isinstance(y, IDDataClass) and u is not None and isinstance(u, SystemIdentificationConfig):
            # Old API: identify(data, config)
            iddata = y
            config = u
            y = None
            u = None
            # Extract parameters from config
            kwargs = {
                'nb': getattr(config, 'nb', 2),
                'nf': getattr(config, 'nf', 2),
                'nk': getattr(config, 'nk', 1),
                'max_iterations': getattr(config, 'max_iterations', 200),
            }

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

        # Extract configuration parameters (OE specific)
        nb = kwargs.get("nb", 2)
        nf = kwargs.get("nf", 2)
        nk = kwargs.get("nk", 1)

        # Validate parameters
        self.validate_parameters(nb=nb, nf=nf, nk=nk)

        # Route to appropriate implementation
        # Remove nb, nf, nk from kwargs to avoid duplicate argument errors
        kwargs_filtered = {k: v for k, v in kwargs.items() if k not in ['nb', 'nf', 'nk', 'tsample']}

        if CASADI_AVAILABLE:
            # Use NLP method (matches master branch)
            try:
                return self._identify_nlp(y, u, nb, nf, nk, sample_time, **kwargs_filtered)
            except Exception as e:
                warnings.warn(
                    f"NLP identification failed: {e}. Falling back to simplified LS method."
                )
                return self._identify_ills(y, u, nb, nf, nk, sample_time, **kwargs_filtered)
        else:
            # Fall back to simplified iterative least squares
            warnings.warn(
                "CasADi not available. Using simplified LS method (may be less accurate than master branch)."
            )
            return self._identify_ills(y, u, nb, nf, nk, sample_time, **kwargs_filtered)

    def _identify_ills(
        self, y, u, nb, nf, nk, sample_time, **kwargs
    ) -> StateSpaceModel:
        """
        Simplified OE identification using iterative least squares.

        This is the fallback method when CasADi is not available.
        Uses actual outputs in regressor (single-pass approximation).

        Parameters:
        -----------
        y, u : np.ndarray
            Output and input data
        nb, nf, nk : int
            Model orders and delay
        sample_time : float
            Sampling time
        **kwargs : dict
            Additional parameters

        Returns:
        --------
        model : StateSpaceModel
            Identified model
        """

        # Get data dimensions
        ny, N = y.shape
        nu, _ = u.shape

        # Create regression matrices for OE identification (uses actual outputs as approximation)
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
            B_coeffs, F_coeffs, nb, nf, nk, ny, nu, sample_time
        )

        # Create state-space representation
        if HAROLD_AVAILABLE:
            model = self._create_state_space_from_oe(
                B_coeffs, F_coeffs, nb, nf, nk, ny, nu, sample_time
            )
        else:
            # Fallback when harold is not available
            model = self._create_mock_model(
                B_coeffs, F_coeffs, nb, nf, nk, ny, nu, sample_time
            )

        # Attach transfer functions and predictions to model
        model.G_tf = G_tf
        model.H_tf = H_tf
        model.Yid = Yid

        return model

    def _identify_nlp(
        self, y, u, nb, nf, nk, sample_time, **kwargs
    ) -> StateSpaceModel:
        """
        OE identification using NLP (CasADi + IPOPT).

        This method matches the master branch reference implementation exactly.
        Uses predicted outputs (Yidw) in regressor for true iterative output error estimation.

        Parameters:
        -----------
        y, u : np.ndarray
            Output and input data (outputs/inputs x time_steps)
        nb, nf, nk : int
            Model orders and delay
        sample_time : float
            Sampling time
        **kwargs : dict
            Additional parameters including max_iterations, stability_constraint, etc.

        Returns:
        --------
        model : StateSpaceModel
            Identified OE model with G_tf, H_tf, Yid
        """
        import casadi as ca  # noqa: F401 - used extensively in this function

        # Get data dimensions
        ny, N = y.shape
        nu, _ = u.shape

        # Check SISO constraint for NLP method
        if ny > 1 or nu > 1:
            raise ValueError(
                "NLP method currently supports SISO only. Use simplified method for MIMO."
            )

        # Flatten to 1D for SISO
        y_flat = y.flatten()
        u_flat = u.flatten()

        # Extract optional parameters
        max_iterations = kwargs.get("max_iterations", 200)
        stability_constraint = kwargs.get("stability_constraint", False)
        stability_margin = kwargs.get("stability_margin", 1.0)

        # Build and solve NLP problem
        solution = self._build_oe_nlp(
            u_flat,
            y_flat,
            nb,
            nf,
            nk,
            N,
            max_iterations,
            stability_constraint,
            stability_margin,
        )

        # Extract coefficients from solution
        B_coeffs = solution["b"].reshape(1, nb)
        F_coeffs = solution["f"].reshape(1, nf)
        Yid = solution["Yid"].reshape(1, N)
        Vn = solution["Vn"]

        # Create G_tf and H_tf transfer functions
        G_tf, H_tf = self._create_transfer_functions_oe(
            B_coeffs, F_coeffs, nb, nf, nk, ny, nu, sample_time
        )

        # Create state-space representation
        if HAROLD_AVAILABLE:
            model = self._create_state_space_from_oe(
                B_coeffs, F_coeffs, nb, nf, nk, ny, nu, sample_time
            )
        else:
            model = self._create_mock_model(
                B_coeffs, F_coeffs, nb, nf, nk, ny, nu, sample_time
            )

        # Attach results
        model.G_tf = G_tf
        model.H_tf = H_tf
        model.Yid = Yid
        model.Vn = Vn
        model.B_coeffs = B_coeffs
        model.F_coeffs = F_coeffs

        return model

    def _build_oe_nlp(
        self, u, y, nb, nf, nk, N, max_iterations, stability_constraint, stability_margin
    ):
        """
        Build and solve OE NLP problem using CasADi + IPOPT.

        Follows master branch implementation exactly:
        - Decision variables: [b, f, Yidw]
        - Regressor uses Yidw[k-nf:k] (predicted outputs, not actual outputs)
        - Equality constraint: symbolic Yid - Yidw = 0
        - Optional stability constraint: norm_inf(companion_F) <= stability_margin

        Parameters:
        -----------
        u, y : np.ndarray
            Flattened input and output data (1D arrays, SISO)
        nb, nf, nk : int
            Model orders and delay
        N : int
            Number of data points
        max_iterations : int
            IPOPT max iterations
        stability_constraint : bool
            Enable stability constraints
        stability_margin : float
            Stability margin for companion matrix norm

        Returns:
        --------
        solution : dict
            Dictionary with keys: 'b', 'f', 'Yid', 'Vn'
        """
        import casadi as ca

        # Number of coefficients
        n_coeff = nb + nf

        # Decision variables: [b (nb), f (nf), Yidw (N)]
        n_opt = n_coeff + N
        w_opt = ca.SX.sym("w", n_opt)

        # Extract coefficient variables
        b = w_opt[0:nb]  # Numerator coefficients
        f = w_opt[nb : nb + nf]  # Denominator coefficients

        # Extract auxiliary variable (predicted outputs)
        Yidw = w_opt[n_coeff : n_coeff + N]

        # Initialize symbolic Yid (will be computed in loop)
        Yid = y * ca.SX.ones(1)  # Start with actual outputs

        # Maximum lag for causality
        n_tr = max(nb + nk - 1, nf)  # Number of non-identifiable initial samples

        # Build symbolic regressor loop (matches master branch)
        for k in range(n_tr, N):
            # Build regressor for time step k
            # OE structure: y[k] = B(q)/F(q) * u[k-nk] + e[k]
            # Regressor: phi = [u[k-nk], ..., u[k-nk-nb+1], -Yidw[k-1], ..., -Yidw[k-nf]]

            phi_parts = []

            # Input terms: lagged inputs u[k-nk:k-nk-nb:-1]
            for i in range(nb):
                idx = k - nk - i
                if idx >= 0:
                    phi_parts.append(u[idx])
                else:
                    phi_parts.append(0.0)

            # Output terms: lagged PREDICTED outputs -Yidw[k-1:k-nf-1:-1]
            # This is the KEY difference from simplified LS - uses Yidw, not y!
            for i in range(nf):
                idx = k - 1 - i
                if idx >= 0:
                    phi_parts.append(-Yidw[idx])
                else:
                    phi_parts.append(0.0)

            # Stack regressor
            phi = ca.vertcat(*phi_parts)

            # Compute prediction: Yid[k] = phi' * coeff
            coeff = ca.vertcat(b, f)
            Yid[k] = ca.mtimes(phi.T, coeff)

        # Objective function: minimize ||Y - Yidw||^2
        DY = y - Yidw
        f_obj = (1.0 / N) * ca.mtimes(DY.T, DY)

        # Constraints
        g = []
        g_lb = []
        g_ub = []

        # Equality constraint: Yid - Yidw = 0 (multiple shooting)
        g.append(Yid - Yidw)
        g_lb.extend([-1e-7] * N)
        g_ub.extend([1e-7] * N)

        # Optional stability constraint: norm_inf(companion_F) <= stability_margin
        if stability_constraint and nf > 0:
            # Companion matrix for F polynomial: F(q) = 1 + f1*q^-1 + ... + fnf*q^-nf
            compF = ca.SX.zeros(nf, nf)
            if nf > 1:
                diagF = ca.SX.eye(nf - 1)
                compF[:-1, 1:] = diagF
            compF[-1, :] = -f[::-1]  # Reverse and negate

            # Infinity norm (upper bound on spectral radius)
            norm_CompF = ca.norm_inf(compF)

            g.append(norm_CompF)
            g_lb.append(-1e-7)
            g_ub.append(stability_margin)

        # Stack constraints
        g_vec = ca.vertcat(*g)

        # Bounds on decision variables
        w_lb = -1e2 * ca.DM.ones(n_opt)
        w_ub = 1e2 * ca.DM.ones(n_opt)

        # NLP problem
        nlp = {"x": w_opt, "f": f_obj, "g": g_vec}

        # Solver options
        opts = {
            "ipopt.max_iter": max_iterations,
            "ipopt.print_level": 0,
            "ipopt.sb": "yes",
            "print_time": 0,
        }

        # Create solver
        solver = ca.nlpsol("solver", "ipopt", nlp, opts)

        # Initial guess: [zeros for coefficients, actual outputs for Yid]
        w_0 = np.zeros(n_opt)
        w_0[n_coeff : n_coeff + N] = y  # Initialize Yidw with actual outputs

        # Solve NLP
        sol = solver(lbx=w_lb, ubx=w_ub, x0=w_0, lbg=g_lb, ubg=g_ub)

        # Extract solution
        x_opt = sol["x"].full().flatten()

        b_opt = x_opt[0:nb]
        f_opt = x_opt[nb : nb + nf]
        Yid_opt = x_opt[n_coeff : n_coeff + N]

        # Compute noise variance
        Vn = np.linalg.norm(y - Yid_opt, 2) ** 2 / (2 * N)

        return {"b": b_opt, "f": f_opt, "Yid": Yid_opt, "Vn": Vn}

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

        # Use local matrices for test mocking compatibility
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
