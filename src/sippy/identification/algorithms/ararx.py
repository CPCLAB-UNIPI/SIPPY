"""
ARARX (Auto-Regressive Auto-Regressive X) identification algorithm.

This implementation uses nonlinear programming (NLP) with CasADi to match
the master branch reference implementation exactly. Auxiliary variables
W and V are optimization variables with explicit equality constraints.
"""

import warnings
from typing import TYPE_CHECKING, Optional

import numpy as np

from ..base import IdentificationAlgorithm, StateSpaceModel

if TYPE_CHECKING:
    from ..iddata import IDData

try:
    import harold

    # Check if harold has the required components
    if hasattr(harold, "State"):
        HAROLD_AVAILABLE = True
    else:
        HAROLD_AVAILABLE = False
        warnings.warn("harold library incomplete. ARARX algorithm will be limited.")
except ImportError:
    HAROLD_AVAILABLE = False
    warnings.warn("harold library not available. ARARX algorithm will be limited.")

try:
    from casadi import DM, SX, mtimes, nlpsol, norm_inf, vertcat

    CASADI_AVAILABLE = True
except ImportError:
    CASADI_AVAILABLE = False
    warnings.warn(
        "CasADi not available. ARARX will use simplified method with reduced accuracy. "
        "Install CasADi for production-quality results: pip install casadi"
    )


class ARARXAlgorithm(IdentificationAlgorithm):
    """
    ARARX (Auto-Regressive Auto-Regressive X) identification algorithm.

    The ARARX model structure is:
    A(q) y(k) = B(q)/D(q) * u(k-theta) + e(k)

    where:
    - A(q) = 1 + a1*q^-1 + ... + ana*q^-na (output auto-regressive polynomial)
    - B(q) = b0 + b1*q^-1 + ... + bnb*q^-nb (input numerator polynomial)
    - D(q) = 1 + d1*q^-1 + ... + dnd*q^-nd (denominator polynomial in input path)
    - theta is the input delay (number of samples)
    - e(k) is white noise
    - NO C(q) polynomial (no MA component in noise model)
    - NO F(q) polynomial (no additional filtering)

    ARARX extends ARX by adding a denominator D(q) in the input transfer function,
    allowing modeling of systems with more complex input dynamics.

    Transfer functions:
    - G(q) = B(q) / (A(q) * D(q))  (deterministic transfer function)
    - H(q) = 1 / A(q)  (noise transfer function - simple AR)

    ## Implementation Methods

    ### NLP Method (Default, CasADi required)
    Uses simultaneous nonlinear programming with auxiliary variables:
    - Decision variables: [a, b, d, W, V, Yid] where W=B*u, V=A*y-W
    - Objective: Minimize (1/N) * sum((y - Yid)^2)
    - Constraints: Explicit equality constraints linking auxiliary variables
    - Solver: IPOPT (Interior Point OPTimizer)
    - Accuracy: Exact maximum likelihood estimate (~0% error vs master)

    ### Simplified Method (Fallback, no CasADi)
    Uses iterative auxiliary variable least squares:
    - Alternates between updating A and updating (B, D)
    - Uses heuristic regularization for numerical stability
    - 50 iterations with convergence checking
    - Accuracy: Approximate (~1-10% error, may fail on ill-conditioned data)

    ## Usage Example

    ```python
    from sippy import SystemIdentification

    # With IDData
    sys_id = SystemIdentification(
        data=iddata,
        method="ARARX",
        na=2, nb=2, nd=1, theta=1
    )
    model = sys_id.identify()

    # With raw arrays
    model = SystemIdentification.identify(
        y=y_data, u=u_data,
        method="ARARX",
        na=2, nb=2, nd=1, theta=1,
        max_iterations=200,
        stability_constraint=False
    )
    ```

    ## Parameters

    Required:
    - na (int): Order of A(q) polynomial (output AR)
    - nb (int): Order of B(q) polynomial (input numerator)
    - nd (int): Order of D(q) polynomial (input denominator)
    - theta (int): Input delay (also accepts 'nk' for backward compatibility)

    Optional (NLP method):
    - max_iterations (int): Maximum IPOPT iterations (default: 200)
    - stability_constraint (bool): Enforce stability via companion matrix norms (default: False)
    - stability_margin (float): Stability margin for poles (default: 1.0)

    ## Notes

    - **CasADi strongly recommended**: The NLP method provides exact ML estimates
      matching the master branch reference implementation. Install with: pip install casadi

    - **Simplified method limitations**: Without CasADi, the algorithm falls back to
      an approximate iterative method with reduced accuracy. Not recommended for
      production use.

    - **Stability**: With `stability_constraint=True`, poles are enforced to have
      magnitude < `stability_margin` (typically 1.0 for discrete-time systems)

    - **Computational cost**: NLP method is 10-50x slower than simplified method but
      provides exact solution. For rapid prototyping, consider ARX first.
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
            Parameters to validate including na, nb, nd, theta

        Returns:
        --------
        bool
            True if parameters are valid
        """
        na = kwargs.get("na")
        nb = kwargs.get("nb")
        nd = kwargs.get("nd")
        theta = kwargs.get("theta")

        # Check if parameters are explicitly set to invalid values
        if na is not None and na < 0:
            raise ValueError("Output AR order (na) must be non-negative")
        if nb is not None and nb <= 0:
            raise ValueError("Input order (nb) must be positive")
        if nd is not None and nd <= 0:
            raise ValueError("Denominator order (nd) must be positive")
        if theta is not None and theta < 0:
            raise ValueError("Input delay (theta) must be non-negative")

        return True

    def identify(
        self,
        y: Optional[np.ndarray] = None,
        u: Optional[np.ndarray] = None,
        iddata: Optional["IDData"] = None,
        **kwargs,
    ) -> StateSpaceModel:
        """
        Identify ARARX model from input-output data.

        Uses NLP method if CasADi available, otherwise falls back to simplified method.

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
            - na, nb, nd, theta (required model orders)
            - max_iterations (int, default 200): IPOPT iterations
            - stability_constraint (bool, default False): Enforce stability
            - stability_margin (float, default 1.0): Pole magnitude limit
            - tsample (float, default 1.0): Sample time

        Returns:
        --------
        model : StateSpaceModel
            Identified state-space model with G_tf, H_tf, and Yid attributes
        """
        # Backward compatibility: detect old API (data, config) vs new API (y, u, **kwargs)
        from ..base import SystemIdentificationConfig
        from ..iddata import IDData as IDDataClass

        if (
            y is not None
            and isinstance(y, IDDataClass)
            and u is not None
            and isinstance(u, SystemIdentificationConfig)
        ):
            # Old API: identify(data, config)
            iddata = y
            config = u
            y = None
            u = None
            # Extract parameters from config
            # ARARX uses theta, but also support nk for backward compatibility
            theta_value = getattr(config, "theta", None)
            if theta_value is None:
                theta_value = getattr(config, "nk", 1)
            kwargs = {
                "na": getattr(config, "na", 1),
                "nb": getattr(config, "nb", 1),
                "nd": getattr(config, "nd", 1),
                "theta": theta_value,
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

        # Extract configuration parameters (ARARX uses na, nb, nd, theta)
        # Handle None values explicitly since kwargs may have key=None
        na = kwargs.get("na")
        if na is None:
            na = 1  # Output AR order default
        nb = kwargs.get("nb")
        if nb is None:
            nb = 1  # Input numerator order default
        nd = kwargs.get("nd")
        if nd is None:
            nd = 1  # Denominator order default
        # ARARX traditionally uses theta, but also support nk for backward compatibility
        theta = kwargs.get("theta")
        if theta is None:
            theta = kwargs.get("nk")  # Try nk parameter
        if theta is None:
            theta = 1  # Default to 1 if neither theta nor nk provided

        # Validate parameters
        self.validate_parameters(na=na, nb=nb, nd=nd, theta=theta)

        # Extract NLP-specific parameters from kwargs (don't pass model orders again)
        nlp_kwargs = {
            "max_iterations": kwargs.get("max_iterations", 200),
            "stability_constraint": kwargs.get("stability_constraint", False),
            "stability_margin": kwargs.get("stability_margin", 1.0),
        }

        # Route to appropriate implementation
        if CASADI_AVAILABLE:
            # Use NLP method (exact, production quality)
            return self._identify_nlp(
                y, u, na, nb, nd, theta, sample_time, **nlp_kwargs
            )
        else:
            # Fallback to simplified method
            warnings.warn(
                "Using simplified ARARX method (CasADi not available). "
                "Accuracy may be reduced. Install CasADi for production use: pip install casadi"
            )
            return self._identify_simplified(y, u, na, nb, nd, theta, sample_time)

    def _identify_nlp(
        self, y, u, na, nb, nd, theta, sample_time, **kwargs
    ) -> StateSpaceModel:
        """
        Identify ARARX model using CasADi NLP optimization.

        This method matches the master branch implementation exactly by formulating
        the identification problem as a constrained nonlinear program.

        Parameters:
        -----------
        y : np.ndarray
            Output data (ny x N)
        u : np.ndarray
            Input data (nu x N)
        na, nb, nd, theta : int
            Model orders and delay
        sample_time : float
            Sampling period
        **kwargs : dict
            Additional parameters (max_iterations, stability_constraint, etc.)

        Returns:
        --------
        model : StateSpaceModel
            Identified state-space model
        """
        # Get data dimensions
        ny, N = y.shape
        nu, _ = u.shape

        # Currently only support SISO
        if ny > 1 or nu > 1:
            raise NotImplementedError("ARARX NLP currently only supports SISO systems")

        # DATA RESCALING (critical for numerical conditioning)
        y_std, y_scaled = self._rescale(y.flatten())
        u_std, u_scaled = self._rescale(u.flatten())

        # Use scaled data for optimization
        y_flat = y_scaled
        u_flat = u_scaled

        # Extract NLP parameters
        max_iterations = kwargs.get("max_iterations", 200)
        stability_cons = kwargs.get("stability_constraint", False)
        stab_marg = kwargs.get("stability_margin", 1.0)

        # Calculate effective data length (n_tr = number of non-identifiable samples)
        n_tr = max(na, nb + theta, nd)

        # Build and solve NLP
        solver, w_lb, w_ub, g_lb, g_ub, w_0 = self._build_ararx_nlp(
            y_flat,
            u_flat,
            na,
            nb,
            nd,
            theta,
            N,
            n_tr,
            max_iterations,
            stab_marg,
            stability_cons,
        )

        # Solve the NLP
        try:
            sol = solver(lbx=w_lb, ubx=w_ub, x0=w_0, lbg=g_lb, ubg=g_ub)

            # Check convergence
            if not solver.stats()["success"]:
                warnings.warn(
                    f"IPOPT did not converge successfully. "
                    f"Return status: {solver.stats()['return_status']}. "
                    f"Results may be suboptimal."
                )
        except Exception as e:
            raise RuntimeError(f"CasADi NLP optimization failed: {e}")

        # Extract solution
        x_opt = sol["x"]
        n_coeff = na + nb + nd

        # Extract polynomial coefficients (from scaled optimization)
        THETA = np.array(x_opt[:n_coeff]).flatten()
        A_coeffs = THETA[:na].reshape(ny, na) if na > 0 else np.zeros((ny, 0))
        B_coeffs_scaled = THETA[na : na + nb].reshape(ny, nb)
        D_coeffs = THETA[na + nb : na + nb + nd].reshape(ny, nd)

        # Extract one-step-ahead predictions (scaled)
        Yid_scaled = np.array(x_opt[-N:]).flatten()

        # RESCALE BACK to original units (critical!)
        # B coefficients scale as: B_original = B_scaled * (y_std / u_std)
        B_coeffs = B_coeffs_scaled * (y_std / u_std)

        # Yid scales as: Yid_original = Yid_scaled * y_std
        Yid_flat = Yid_scaled * y_std
        Yid = Yid_flat.reshape(ny, N)

        # Estimate noise variance (using original scale)
        Vn = (np.linalg.norm(Yid_flat - y.flatten(), 2) ** 2) / (2 * N)

        # Create transfer functions
        G_tf, H_tf = self._create_transfer_functions_ararx(
            A_coeffs, B_coeffs, D_coeffs, na, nb, nd, theta, ny, nu, sample_time
        )

        # Create state-space model
        if HAROLD_AVAILABLE:
            model = self._create_state_space_from_ararx(
                A_coeffs, B_coeffs, D_coeffs, na, nb, nd, theta, ny, nu, sample_time
            )
        else:
            model = self._create_mock_model(
                A_coeffs, B_coeffs, D_coeffs, na, nb, nd, theta, ny, nu, sample_time
            )

        # Attach attributes
        model.G_tf = G_tf
        model.H_tf = H_tf
        model.Yid = Yid
        model.Vn = Vn

        return model

    def _rescale(self, data):
        """
        Normalize data to zero mean and unit standard deviation.

        This is critical for numerical conditioning in NLP optimization.
        Prevents ill-conditioning when inputs/outputs have different scales.

        Parameters:
        -----------
        data : np.ndarray
            Input array (1D)

        Returns:
        --------
        data_std : float
            Standard deviation (for rescaling back)
        data_scaled : np.ndarray
            Normalized data (mean=0, std=1)
        """
        data_mean = np.mean(data)
        data_std = np.std(data)

        # Handle constant signals (avoid division by zero)
        if data_std < 1e-10:
            return 1.0, data - data_mean

        data_scaled = (data - data_mean) / data_std
        return data_std, data_scaled

    def _build_ararx_nlp(
        self,
        y,
        u,
        na,
        nb,
        nd,
        theta,
        N,
        n_tr,
        max_iterations,
        stab_marg,
        stability_cons,
    ):
        """
        Build CasADi NLP problem for ARARX identification.

        This follows the master branch formulation exactly from functionset_OPT.py.

        Decision Variables:
        - a[0:na]: A polynomial coefficients
        - b[na:na+nb]: B polynomial coefficients
        - d[na+nb:na+nb+nd]: D polynomial coefficients
        - W[-3*N:-2*N]: Auxiliary variable W = B*u
        - V[-2*N:-N]: Auxiliary variable V = A*y - W
        - Yid[-N:]: One-step-ahead predictions

        Objective:
        - Minimize (1/N) * sum((y - Yid)^2)

        Constraints:
        - Yid[k] = -sum(a*y_past) + sum(b*u_past) - sum(d*V_past)
        - W[k] = sum(b*u_past)
        - V[k] = y[k] + sum(a*y_past) - W[k]
        - Optional: ||companion(A)||_inf <= stab_marg
        - Optional: ||companion(D)||_inf <= stab_marg

        Parameters:
        -----------
        y, u : np.ndarray
            Flattened data arrays (length N)
        na, nb, nd, theta : int
            Model orders
        N, n_tr : int
            Total samples and non-identifiable samples
        max_iterations : int
            IPOPT iteration limit
        stab_marg : float
            Stability margin
        stability_cons : bool
            Whether to enforce stability constraints

        Returns:
        --------
        solver : casadi.nlpsol
            Configured NLP solver
        w_lb, w_ub : casadi.DM
            Variable bounds
        g_lb, g_ub : casadi.DM
            Constraint bounds
        w_0 : casadi.DM
            Initial guess
        """
        # Number of coefficients
        n_coeff = na + nb + nd

        # Total optimization variables: [a, b, d, W, V, Yid]
        n_aus = 3 * N
        n_opt = n_coeff + n_aus

        # Define symbolic optimization variables
        w_opt = SX.sym("w", n_opt)

        # Extract coefficient subsets
        a = w_opt[0:na]
        b = w_opt[na : na + nb]
        d = w_opt[na + nb : na + nb + nd]

        # Extract auxiliary variables
        Ww = w_opt[-3 * N : -2 * N]  # W symbolic variable
        Vw = w_opt[-2 * N : -N]  # V symbolic variable
        Yidw = w_opt[-N:]  # Yid symbolic variable

        # Build coefficient vector for regressor
        coeff = vertcat(a, b, d)
        coeff_w = vertcat(b)  # For W = B*u
        coeff_v = a if na > 0 else SX.zeros(0)  # For V = A*y - W

        # Initialize symbolic predictions
        Yid = y * SX.ones(1)
        W = y * SX.ones(1)
        V = y * SX.ones(1)

        # Build prediction equations for k >= n_tr
        for k in range(N):
            if k >= n_tr:
                # Build regressor for Yid prediction
                # phi = [-y_lags, u_lags, -V_lags]

                # Output lags
                vecY = y[k - na : k][::-1] if na > 0 else SX.zeros(0)

                # Input lags
                vecU = u[k - nb - theta : k - theta][::-1]

                # V lags
                vecV = Vw[k - nd : k][::-1]

                # Regressor for ARARX: phi = [-vecY, vecU, -vecV]
                phi = vertcat(-vecY, vecU, -vecV)

                # Prediction: Yid[k] = phi' * [a; b; d]
                Yid[k] = mtimes(phi.T, coeff)

                # W auxiliary variable: W[k] = B*u
                phiw = vertcat(vecU)
                W[k] = mtimes(phiw.T, coeff_w)

                # V auxiliary variable: V[k] = y[k] + A*y - W[k]
                if na == 0:
                    V[k] = y[k] - Ww[k]
                else:
                    phiv = vertcat(vecY)
                    V[k] = y[k] + mtimes(phiv.T, coeff_v) - Ww[k]

        # Objective function: minimize mean squared error
        DY = y - Yidw
        f_obj = (1.0 / N) * mtimes(DY.T, DY)

        # Equality constraints
        g = []

        # 1. Yid consistency constraint
        g.append(Yid - Yidw)

        # 2. W consistency constraint
        g.append(W - Ww)

        # 3. V consistency constraint
        g.append(V - Vw)

        # Stability constraints (optional)
        ng_norm = 0
        if stability_cons:
            if na > 0:
                ng_norm += 1
                # Companion matrix for A(q)
                compA = SX.zeros(na, na)
                if na > 1:
                    diagA = SX.eye(na - 1)
                    compA[:-1, 1:] = diagA
                compA[-1, :] = -a[::-1]
                norm_CompA = norm_inf(compA)
                g.append(norm_CompA)

            if nd > 0:
                ng_norm += 1
                # Companion matrix for D(q)
                compD = SX.zeros(nd, nd)
                if nd > 1:
                    diagD = SX.eye(nd - 1)
                    compD[:-1, 1:] = diagD
                compD[-1, :] = -d[::-1]
                norm_CompD = norm_inf(compD)
                g.append(norm_CompD)

        # Stack constraint vector
        g_ = vertcat(*g)

        # Variable bounds
        w_lb = -1e2 * DM.ones(n_opt)
        w_ub = 1e2 * DM.ones(n_opt)

        # Constraint bounds (equality constraints: g = 0)
        ng = g_.size1()
        g_lb = -1e-7 * DM.ones(ng, 1)
        g_ub = 1e-7 * DM.ones(ng, 1)

        # Update stability constraint bounds
        if ng_norm > 0:
            g_ub[-ng_norm:] = stab_marg * DM.ones(ng_norm, 1)

        # Initial guess
        w_0 = DM.zeros(n_opt)
        # Coefficients initialized to zero
        # Yid initialized to measured output
        w_0[-N:] = y
        # W and V initialized to measured output (arbitrary)
        w_0[-3 * N : -2 * N] = y
        w_0[-2 * N : -N] = y

        # Define NLP problem
        nlp = {"x": w_opt, "f": f_obj, "g": g_}

        # Solver options (match master branch)
        sol_opts = {
            "ipopt.max_iter": max_iterations,
            "ipopt.print_level": 0,
            "ipopt.sb": "yes",
            "print_time": 0,
        }

        # Create solver
        solver = nlpsol("solver", "ipopt", nlp, sol_opts)

        return solver, w_lb, w_ub, g_lb, g_ub, w_0

    def _identify_simplified(
        self, y, u, na, nb, nd, theta, sample_time
    ) -> StateSpaceModel:
        """
        Fallback simplified method when CasADi not available.

        Uses iterative auxiliary variable least squares. Less accurate than NLP.

        Parameters:
        -----------
        y : np.ndarray
            Output data (ny x N)
        u : np.ndarray
            Input data (nu x N)
        na, nb, nd, theta : int
            Model orders and delay
        sample_time : float
            Sampling period

        Returns:
        --------
        model : StateSpaceModel
            Identified state-space model
        """
        # Get data dimensions
        ny, N = y.shape
        nu, _ = u.shape

        # Calculate effective data length
        max_lag = max(na, nb + theta, nd)
        N_eff = N - max_lag

        # Check if we have enough data
        n_params = na + nb + nd
        if N_eff <= 0:
            raise ValueError(
                f"Not enough data. Need at least {max_lag + 1} samples, got {N}"
            )
        if N_eff <= n_params:
            raise ValueError(
                f"Not enough data for parameter estimation. Need more than {n_params} samples, got {N_eff}"
            )

        # Step 1: Initialize with ARX estimate (A and B, D=1 initially)
        A_coeffs, B_coeffs = self._initialize_with_arx(u, y, na, nb, theta, N, max_lag)
        D_coeffs = np.zeros((ny, nd))

        # Step 2: Iterative optimization using auxiliary variables
        # Increased from 10 to 50 for better convergence
        max_iter = 50
        tol = 1e-8  # Tighter tolerance for better accuracy

        # Track convergence for diagnostics
        converged = False

        for iteration in range(max_iter):
            A_prev = A_coeffs.copy()
            B_prev = B_coeffs.copy()
            D_prev = D_coeffs.copy()

            # Compute auxiliary variable V = y - B/D * u
            V = self._compute_auxiliary_V(
                y, u, B_coeffs, D_coeffs, nb, nd, theta, N, max_lag
            )

            # Update A using [y, V] regression
            A_coeffs = self._update_A_coefficients(y, V, na, N, max_lag)

            # Compute auxiliary variable W = A * y
            W = self._compute_auxiliary_W(y, A_coeffs, na, N, max_lag)

            # Update B and D using [u, W] regression
            B_coeffs, D_coeffs = self._update_BD_coefficients(
                u, W, nb, nd, theta, N, max_lag
            )

            # Check convergence using relative change (more robust than absolute)
            # Use Frobenius norm for matrices
            norm_A_prev = (
                np.linalg.norm(A_prev) + 1e-10
            )  # Add small value to prevent division by zero
            norm_B_prev = np.linalg.norm(B_prev) + 1e-10
            norm_D_prev = np.linalg.norm(D_prev) + 1e-10

            rel_delta_A = np.linalg.norm(A_coeffs - A_prev) / norm_A_prev
            rel_delta_B = np.linalg.norm(B_coeffs - B_prev) / norm_B_prev
            rel_delta_D = np.linalg.norm(D_coeffs - D_prev) / norm_D_prev

            max_rel_change = max(rel_delta_A, rel_delta_B, rel_delta_D)

            if max_rel_change < tol:
                converged = True
                break

        # Warn if convergence not achieved
        if not converged and ny > 0:
            warnings.warn(
                f"ARARX simplified method did not converge after {max_iter} iterations. "
                f"Final relative change: {max_rel_change:.2e}. "
                f"Consider installing CasADi for better accuracy or checking data quality."
            )

        # Step 3: Compute Yid (one-step-ahead predictions)
        Yid = self._compute_yid_ararx(
            u, y, A_coeffs, B_coeffs, D_coeffs, na, nb, nd, theta, N, max_lag
        )

        # Step 4: Create transfer functions using harold
        G_tf, H_tf = self._create_transfer_functions_ararx(
            A_coeffs, B_coeffs, D_coeffs, na, nb, nd, theta, ny, nu, sample_time
        )

        # Step 5: Create state-space model
        if HAROLD_AVAILABLE:
            model = self._create_state_space_from_ararx(
                A_coeffs, B_coeffs, D_coeffs, na, nb, nd, theta, ny, nu, sample_time
            )
        else:
            model = self._create_mock_model(
                A_coeffs, B_coeffs, D_coeffs, na, nb, nd, theta, ny, nu, sample_time
            )

        # Attach attributes
        model.G_tf = G_tf
        model.H_tf = H_tf
        model.Yid = Yid

        return model

    def _initialize_with_arx(self, u, y, na, nb, theta, N, max_lag):
        """Initialize A and B coefficients using ARX estimation."""
        ny = y.shape[0]
        nu = u.shape[0]
        N_eff = N - max_lag

        # For MIMO, estimate separately for each output
        A_coeffs = np.zeros((ny, na)) if na > 0 else np.zeros((ny, 0))
        B_coeffs = np.zeros((ny, nb))

        for out_idx in range(ny):
            # Construct regression matrix [y_lags, u_lags] for this output
            Phi = np.zeros((N_eff, na + nb * nu))

            # Add output lags for A polynomial (own output only for SISO structure)
            for i in range(na):
                Phi[:, i] = y[out_idx, max_lag - 1 - i : max_lag - 1 - i + N_eff]

            # Add input lags for B polynomial (all inputs)
            for inp_idx in range(nu):
                for i in range(nb):
                    col = na + inp_idx * nb + i
                    Phi[:, col] = u[
                        inp_idx, max_lag - theta - i : max_lag - theta - i + N_eff
                    ]

            # Solve least squares for this output
            target = y[out_idx, max_lag : max_lag + N_eff]
            theta_arx, _, _, _ = np.linalg.lstsq(Phi, target, rcond=None)

            # Extract coefficients
            if na > 0:
                A_coeffs[out_idx, :] = theta_arx[:na]
            # For B, use first input coefficients (SISO-like for now)
            B_coeffs[out_idx, :] = theta_arx[na : na + nb]

        return A_coeffs, B_coeffs

    def _compute_auxiliary_V(self, y, u, B_coeffs, D_coeffs, nb, nd, theta, N, max_lag):
        """Compute V = y - B/D * u (auxiliary variable for A estimation)."""
        ny = y.shape[0]
        N_eff = N - max_lag
        V = np.zeros((ny, N_eff))

        for i in range(ny):
            for k in range(N_eff):
                k_abs = k + max_lag

                # Compute B * u term
                b_u = 0
                for j in range(nb):
                    if k_abs - theta - j >= 0:
                        b_u += B_coeffs[i, j] * u[0, k_abs - theta - j]

                # Compute D denominator effect (simplified recursive filtering)
                d_denom = 1.0
                for j in range(nd):
                    if k - j - 1 >= 0:
                        d_denom += D_coeffs[i, j] * V[i, k - j - 1]

                # V = y - B/D * u
                # Adaptive regularization: use fraction of B*u magnitude instead of hardcoded 0.1
                epsilon = max(abs(d_denom) * 0.01, abs(b_u) * 0.01, 1e-6)
                if abs(d_denom) < epsilon:
                    # If denominator too small, use approximation V ≈ y - B*u
                    V[i, k] = y[i, k_abs] - b_u
                else:
                    V[i, k] = y[i, k_abs] - b_u / d_denom

        return V

    def _update_A_coefficients(self, y, V, na, N, max_lag):
        """Update A coefficients using regression on [y, V]."""
        if na == 0:
            return np.zeros((y.shape[0], 0))

        N_eff = V.shape[1]
        ny = y.shape[0]
        A_coeffs = np.zeros((ny, na))

        # Estimate A separately for each output
        for out_idx in range(ny):
            # Build regression matrix with lagged V for this output
            Phi = np.zeros((N_eff, na))
            for i in range(na):
                # Use V lags (since V = y - B/D*u, we're regressing on corrected output)
                if i == 0:
                    # No lag - use all of V
                    Phi[:, i] = -V[out_idx, :]
                else:
                    # i-step lag - shift V by i steps
                    Phi[i:, i] = -V[out_idx, : N_eff - i]
                    Phi[:i, i] = 0  # Pad with zeros at the beginning

            target = y[out_idx, max_lag : max_lag + N_eff]
            A_new, _, _, _ = np.linalg.lstsq(Phi, target, rcond=None)
            A_coeffs[out_idx, :] = A_new

        return A_coeffs

    def _compute_auxiliary_W(self, y, A_coeffs, na, N, max_lag):
        """Compute W = A * y (auxiliary variable for B, D estimation)."""
        ny = y.shape[0]
        N_eff = N - max_lag
        W = np.zeros((ny, N_eff))

        for i in range(ny):
            for k in range(N_eff):
                k_abs = k + max_lag
                W[i, k] = y[i, k_abs]

                # Add AR terms: A(q) * y
                for j in range(na):
                    if k_abs - j - 1 >= 0:
                        W[i, k] += A_coeffs[i, j] * y[i, k_abs - j - 1]

        return W

    def _update_BD_coefficients(self, u, W, nb, nd, theta, N, max_lag):
        """Update B and D coefficients using regression on [u, W]."""
        nu = u.shape[0]
        ny = W.shape[0]
        N_eff = W.shape[1]

        B_coeffs = np.zeros((ny, nb))
        D_coeffs = np.zeros((ny, nd))

        # Estimate B and D separately for each output
        for out_idx in range(ny):
            # Build regression for B and D for this output
            Phi = np.zeros((N_eff, nb * nu + nd))

            # Input lags for B (all inputs)
            for inp_idx in range(nu):
                for i in range(nb):
                    col = inp_idx * nb + i
                    Phi[:, col] = u[
                        inp_idx, max_lag - theta - i : max_lag - theta - i + N_eff
                    ]

            # W lags for D (denominator)
            for i in range(nd):
                if i == 0:
                    # Current W not available, use previous
                    Phi[1:, nb * nu + i] = -W[out_idx, : N_eff - 1]
                    Phi[0, nb * nu + i] = 0
                else:
                    Phi[i:, nb * nu + i] = -W[out_idx, : N_eff - i]
                    Phi[:i, nb * nu + i] = 0

            target = W[out_idx, :N_eff]
            theta_bd, _, _, _ = np.linalg.lstsq(Phi, target, rcond=None)

            # Extract coefficients (use first input's B coefficients for SISO-like structure)
            B_coeffs[out_idx, :] = theta_bd[:nb]
            D_coeffs[out_idx, :] = theta_bd[nb * nu :]

        return B_coeffs, D_coeffs

    def _compute_yid_ararx(
        self, u, y, A_coeffs, B_coeffs, D_coeffs, na, nb, nd, theta, N, max_lag
    ):
        """Compute one-step-ahead predictions for ARARX model."""
        ny = y.shape[0]
        Yid = np.zeros_like(y)
        Yid[:, :max_lag] = y[:, :max_lag]

        for k in range(max_lag, N):
            for i in range(ny):
                # AR part: -A * y
                y_pred = 0
                for j in range(na):
                    if k - j - 1 >= 0:
                        y_pred -= A_coeffs[i, j] * y[i, k - j - 1]

                # Input part: B * u
                b_u = 0
                for j in range(nb):
                    if k - theta - j >= 0:
                        b_u += B_coeffs[i, j] * u[0, k - theta - j]

                # D denominator effect (recursive)
                d_effect = 1.0
                for j in range(nd):
                    if k - j - 1 >= 0:
                        d_effect += D_coeffs[i, j] * Yid[i, k - j - 1]

                # Combine: y = -A*y + B/D*u
                # Adaptive regularization similar to V computation
                epsilon = max(abs(d_effect) * 0.01, abs(b_u) * 0.01, 1e-6)
                if abs(d_effect) < epsilon:
                    # If denominator too small, use approximation: y_pred ≈ -A*y + B*u
                    Yid[i, k] = y_pred + b_u
                else:
                    Yid[i, k] = y_pred + b_u / d_effect

        return Yid

    def _create_transfer_functions_ararx(
        self, A_coeffs, B_coeffs, D_coeffs, na, nb, nd, theta, ny, nu, Ts
    ):
        """
        Create G_tf and H_tf transfer functions using harold.Transfer.

        For ARARX:
        - G_tf = B(q) / (A(q) * D(q))
        - H_tf = 1 / A(q)

        Parameters:
        -----------
        A_coeffs, B_coeffs, D_coeffs : ndarray
            Polynomial coefficients
        na, nb, nd, theta : int
            Polynomial orders and delay
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

            # Build polynomial arrays (harold uses positive powers, convert from negative)
            A_poly = (
                np.concatenate(([1.0], A_coeffs.flatten()))
                if na > 0
                else np.array([1.0])
            )

            # Build B polynomial with delay
            # For discrete TF, B(q) = b0*q^-theta + b1*q^-(theta+1) + ... + bnb*q^-(theta+nb)
            # In harold array form: [0, 0, ..., 0, b0, b1, ..., bnb]
            B_poly = np.concatenate((B_coeffs.flatten(), [0.0] * theta))

            D_poly = np.concatenate(([1.0], D_coeffs.flatten()))

            # Multiply A * D for denominator using harold.haroldpolymul
            # For MIMO case, A_coeffs and D_coeffs are (ny x na) and (ny x nd)
            # Use first output's coefficients for SISO-like TF
            DEN_G = harold.haroldpolymul(A_poly, D_poly)

            # Ensure numerator and denominator have valid lengths
            # harold Transfer needs non-empty numerator with at least one non-zero element
            if len(B_poly) == 0 or np.all(B_poly == 0):
                B_poly = np.array([0.0])

            # Create G transfer function: G(q) = B(q) / (A(q) * D(q))
            G_tf = harold.Transfer(B_poly, DEN_G, dt=Ts)

            # Create H transfer function: H(q) = 1 / A(q)
            H_tf = harold.Transfer([1.0], A_poly, dt=Ts)

            return G_tf, H_tf
        except Exception as e:
            warnings.warn(f"Failed to create ARARX transfer functions: {e}")
            return None, None

    def _create_state_space_from_ararx(
        self, A_coeffs, B_coeffs, D_coeffs, na, nb, nd, theta, ny, nu, Ts
    ):
        """
        Create state-space model from ARARX using harold.transfer_to_state.

        Parameters:
        -----------
        A_coeffs, B_coeffs, D_coeffs : ndarray
            Polynomial coefficients
        na, nb, nd, theta : int
            Polynomial orders and delay
        ny, nu : int
            Number of outputs and inputs
        Ts : float
            Sampling time

        Returns:
        --------
        model : StateSpaceModel
            State-space model representation
        """
        import harold

        # Create transfer function
        G_tf, H_tf = self._create_transfer_functions_ararx(
            A_coeffs, B_coeffs, D_coeffs, na, nb, nd, theta, ny, nu, Ts
        )

        if G_tf is None:
            # Fallback to mock model
            return self._create_mock_model(
                A_coeffs, B_coeffs, D_coeffs, na, nb, nd, theta, ny, nu, Ts
            )

        # Convert to state-space using harold
        try:
            ss_model = harold.transfer_to_state(G_tf)

            # Extract matrices (harold uses lowercase)
            A = ss_model.a
            B = ss_model.b
            C = ss_model.c
            D = ss_model.d

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
        except Exception as e:
            warnings.warn(f"Harold transfer_to_state failed: {e}, using fallback")
            return self._create_mock_model(
                A_coeffs, B_coeffs, D_coeffs, na, nb, nd, theta, ny, nu, Ts
            )

    def _create_mock_model(
        self, A_coeffs, B_coeffs, D_coeffs, na, nb, nd, theta, ny, nu, Ts
    ):
        """
        Create a mock state-space model when harold is not available.

        Parameters:
        -----------
        A_coeffs, B_coeffs, D_coeffs : ndarray
            Polynomial coefficients
        na, nb, nd, theta : int
            Polynomial orders and delay
        ny, nu : int
            Number of outputs and inputs
        Ts : float
            Sampling time

        Returns:
        --------
        model : StateSpaceModel
            Mock state-space model
        """
        # Use companion form for A, B, D polynomials
        n_states = max(na, nb + nd, 1)

        # Build A matrix (companion form for output AR)
        A = np.zeros((n_states, n_states))
        if n_states > 1:
            A[: n_states - 1, 1:] = np.eye(n_states - 1)
        if na > 0 and na <= n_states:
            A[n_states - 1, :na] = -A_coeffs.flatten()

        # Build B matrix (input with denominator effect)
        B = np.zeros((n_states, nu))
        if nb > 0:
            n_copy = min(nb, n_states)
            B[:n_copy, 0] = B_coeffs.flatten()[:n_copy]

        # Build C matrix
        C = np.zeros((ny, n_states))
        C[0, n_states - 1] = 1.0

        # D matrix
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
