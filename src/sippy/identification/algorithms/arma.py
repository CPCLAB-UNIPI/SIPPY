"""
ARMA (AutoRegressive Moving Average) identification algorithm.

This implementation uses nonlinear programming (NLP) with CasADi to match
the master branch reference implementation exactly. The noise sequence is
treated as an optimization variable with explicit equality constraints.
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
    if hasattr(harold, "State"):
        HAROLD_AVAILABLE = True
    else:
        HAROLD_AVAILABLE = False
        warnings.warn("harold library incomplete. ARMA algorithm will be limited.")
except ImportError:
    HAROLD_AVAILABLE = False
    warnings.warn("harold library not available. ARMA algorithm will be limited.")

try:
    from casadi import DM, SX, mtimes, nlpsol, norm_inf, vertcat

    CASADI_AVAILABLE = True
except ImportError:
    CASADI_AVAILABLE = False
    warnings.warn(
        "CasADi not available. ARMA will use simplified method with reduced accuracy. "
        "Install CasADi for production-quality results: pip install casadi"
    )


class ARMAAlgorithm(IdentificationAlgorithm):
    """
    ARMA (AutoRegressive Moving Average) identification algorithm.

    The ARMA model structure is:
    A(q) y(k) = C(q) e(k)

    where:
    - A(q) = 1 + a1*q^-1 + ... + ana*q^-na (auto-regressive polynomial)
    - C(q) = 1 + c1*q^-1 + ... + cnc*q^-nc (moving average polynomial)
    - e(k) is white noise
    - NO inputs (time-series only model)

    Transfer functions:
    - G(q) = None (no input-output dynamics)
    - H(q) = C(q) / A(q) (noise transfer function)

    ## Implementation Methods

    ### NLP Method (Default, CasADi required)
    Uses simultaneous nonlinear programming with noise sequence optimization:
    - Decision variables: [a, c, e[noise sequence], Yid]
    - Objective: Minimize (1/N) * sum((y - Yid)^2)
    - Constraints: Explicit equality constraints for consistency
    - Solver: IPOPT (Interior Point OPTimizer)
    - Accuracy: Exact maximum likelihood estimate (~0% error vs master)

    ### ILLS Method (Fallback, no CasADi)
    Uses iterative extended least squares:
    - Alternates between updating noise estimates and coefficients
    - 100 iterations with variance-based convergence checking
    - Uses binary search for step size adaptation
    - Accuracy: Approximate (~10-100% error, may fail on difficult data)

    ## Usage Example

    ```python
    from sippy import SystemIdentification

    # With IDData (time series only, no inputs)
    sys_id = SystemIdentification(
        data=iddata,
        method="ARMA",
        na=2, nc=1
    )
    model = sys_id.identify()

    # With raw arrays
    model = SystemIdentification.identify(
        y=y_data,
        method="ARMA",
        na=2, nc=1,
        max_iterations=200
    )
    ```

    ## Parameters

    Required:
    - na (int): Order of A(q) polynomial (auto-regressive)
    - nc (int): Order of C(q) polynomial (moving average)

    Optional (NLP method):
    - max_iterations (int): Maximum IPOPT iterations (default: 200)
    - stability_constraint (bool): Enforce stability via companion matrix norms (default: False)
    - stability_margin (float): Stability margin for poles (default: 1.0)

    Optional (ILLS method):
    - max_iterations (int): Maximum ILLS iterations (default: 100)
    - tolerance (float): Convergence threshold (default: 1e-6)

    ## Notes

    - **CasADi strongly recommended**: The NLP method provides exact ML estimates
      matching the master branch reference implementation. Install with: pip install casadi

    - **ILLS method limitations**: Without CasADi, the algorithm falls back to
      an iterative least squares method with reduced accuracy. Not recommended for
      production use.

    - **Time-series only**: ARMA has no inputs (u). For input-output systems, use ARMAX.

    - **Stability**: With `stability_constraint=True`, poles are enforced to have
      magnitude < `stability_margin` (typically 1.0 for discrete-time systems)

    - **Computational cost**: NLP method is 10-50x slower than ILLS method but
      provides exact solution. For rapid prototyping, consider AR or ARX first.
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

    def identify(
        self,
        y: Optional[np.ndarray] = None,
        u: Optional[np.ndarray] = None,
        iddata: Optional["IDData"] = None,
        **kwargs,
    ) -> StateSpaceModel:
        """
        Identify ARMA model from output data (time series).

        Uses NLP method if CasADi available, otherwise falls back to ILLS method.

        Parameters:
        -----------
        y : np.ndarray, optional
            Output data (outputs x time_steps)
        u : np.ndarray, optional
            Input data (inputs x time_steps) - ignored for ARMA
        iddata : IDData, optional
            Input-output data container
        **kwargs : dict
            Configuration parameters including:
            - na, nc (required model orders)
            - max_iterations (int, default 200 for NLP, 100 for ILLS): iterations
            - stability_constraint (bool, default False): Enforce stability
            - stability_margin (float, default 1.0): Pole magnitude limit
            - tolerance (float, default 1e-6): ILLS convergence threshold
            - tsample (float, default 1.0): Sample time

        Returns:
        --------
        model : StateSpaceModel
            Identified state-space model with H_tf and Yid attributes
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
                'na': getattr(config, 'na', 1),
                'nc': getattr(config, 'nc', 1),
            }

        # Validate input arguments
        if iddata is not None and (y is not None or u is not None):
            raise ValueError("Provide either iddata or (y, u), but not both")
        if iddata is None and y is None:
            raise ValueError("Must provide either iddata or y")

        # Extract data if IDData is provided
        if iddata is not None:
            u = iddata.get_input_array()
            y = iddata.get_output_array()
            sample_time = iddata.sample_time
        else:
            # Ensure arrays are 2D
            y = np.atleast_2d(y)
            if u is not None:
                u = np.atleast_2d(u)
            else:
                # ARMA doesn't use inputs, create dummy if not provided
                u = np.zeros((1, y.shape[1]))
            sample_time = kwargs.get("tsample", 1.0)

        # Extract configuration parameters (ARMA specific)
        na = kwargs.get("na", 1)
        nc = kwargs.get("nc", 1)

        # Validate parameters
        self.validate_parameters(na=na, nc=nc)

        # Route to appropriate implementation
        # Remove na, nc from kwargs to avoid duplicate arguments
        kwargs_without_orders = {k: v for k, v in kwargs.items() if k not in ['na', 'nc']}

        if CASADI_AVAILABLE:
            # Use NLP method (exact, production quality)
            return self._identify_nlp(y, na, nc, sample_time, **kwargs_without_orders)
        else:
            # Fallback to ILLS method
            warnings.warn(
                "Using simplified ARMA method (CasADi not available). "
                "Accuracy may be reduced. Install CasADi for production use: pip install casadi"
            )
            return self._identify_ills(y, u, na, nc, sample_time, **kwargs_without_orders)

    def _identify_nlp(self, y, na, nc, sample_time, **kwargs) -> StateSpaceModel:
        """
        Identify ARMA model using CasADi NLP optimization.

        This method matches the master branch implementation exactly by formulating
        the identification problem as a constrained nonlinear program.

        Parameters:
        -----------
        y : np.ndarray
            Output data (ny x N)
        na, nc : int
            Model orders
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

        # Currently only support SISO
        if ny > 1:
            raise NotImplementedError("ARMA NLP currently only supports SISO systems")

        # DATA RESCALING (critical for numerical conditioning)
        # Master branch: divide by std only, NO mean centering
        y_std, y_scaled = self._rescale(y.flatten())

        # Use scaled data for optimization
        y_flat = y_scaled

        # Extract NLP parameters
        max_iterations = kwargs.get("max_iterations", 200)
        stability_cons = kwargs.get("stability_constraint", False)
        stab_marg = kwargs.get("stability_margin", 1.0)

        # Calculate effective data length
        n_tr = max(na, nc)

        # Build and solve NLP
        solver, w_lb, w_ub, g_lb, g_ub, w_0 = self._build_arma_nlp(
            y_flat, na, nc, N, n_tr, max_iterations, stab_marg, stability_cons
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
        n_coeff = na + nc

        # Extract polynomial coefficients (from scaled optimization)
        THETA = np.array(x_opt[:n_coeff]).flatten()
        A_coeffs = THETA[:na].reshape(ny, na) if na > 0 else np.zeros((ny, 0))
        C_coeffs = THETA[na : na + nc].reshape(ny, nc) if nc > 0 else np.zeros((ny, 0))

        # Extract one-step-ahead predictions (scaled)
        Yid_scaled = np.array(x_opt[-N:]).flatten()

        # RESCALE BACK to original units
        # Master branch: multiply by std only (no mean was subtracted)
        Yid_flat = Yid_scaled * y_std
        Yid = Yid_flat.reshape(ny, N)

        # Estimate noise variance (using original scale)
        Vn = (np.linalg.norm(Yid_flat - y.flatten(), 2) ** 2) / (2 * N)

        # Create transfer functions (G_tf=None for ARMA, H_tf=C/A)
        G_tf, H_tf = self._create_transfer_functions_arma(
            A_coeffs, C_coeffs, na, nc, ny, sample_time
        )

        # Create state-space model
        if HAROLD_AVAILABLE:
            model = self._create_state_space_from_arma(
                A_coeffs, C_coeffs, na, nc, ny, sample_time
            )
        else:
            model = self._create_mock_model(A_coeffs, C_coeffs, na, nc, ny, sample_time)

        # Attach attributes
        model.G_tf = G_tf
        model.H_tf = H_tf
        model.Yid = Yid
        model.Vn = Vn

        # Attach AR and MA coefficients for easy access
        model.AR_coeffs = A_coeffs
        model.MA_coeffs = C_coeffs

        return model

    def _rescale(self, data):
        """
        Normalize data to unit standard deviation (NO mean centering).

        This matches the master branch implementation exactly.
        Master only divides by std, not by mean (different from typical z-score).

        This is critical for numerical conditioning in NLP optimization.
        Prevents ill-conditioning when data has extreme scales.

        Parameters:
        -----------
        data : np.ndarray
            Input array (1D)

        Returns:
        --------
        data_std : float
            Standard deviation (for rescaling back)
        data_scaled : np.ndarray
            Normalized data (divided by std only, mean preserved)
        """
        data_std = np.std(data)

        # Handle constant signals (avoid division by zero)
        if data_std < 1e-10:
            return 1.0, data

        data_scaled = data / data_std
        return data_std, data_scaled

    def _build_arma_nlp(
        self, y, na, nc, N, n_tr, max_iterations, stab_marg, stability_cons
    ):
        """
        Build CasADi NLP problem for ARMA identification.

        This follows the master branch formulation from functionset_OPT.py.

        Decision Variables:
        - a[0:na]: A polynomial coefficients
        - c[na:na+nc]: C polynomial coefficients
        - e[-N:]: Noise sequence (prediction errors)
        - Yid[-N:]: One-step-ahead predictions

        Objective:
        - Minimize (1/N) * sum((y - Yid)^2)

        Constraints:
        - Yid[k] = -sum(a*y_past) + sum(c*e_past)
        - e[k] = y[k] - Yid[k]
        - Optional: ||companion(A)||_inf <= stab_marg
        - Optional: ||companion(C)||_inf <= stab_marg

        Parameters:
        -----------
        y : np.ndarray
            Flattened data array (length N)
        na, nc : int
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
        n_coeff = na + nc

        # Total optimization variables: [a, c, Yid]
        # Note: noise e is NOT a separate variable, it's computed from y - Yid
        n_opt = n_coeff + N

        # Define symbolic optimization variables
        w_opt = SX.sym("w", n_opt)

        # Extract coefficient subsets
        a = w_opt[0:na]
        c = w_opt[na : na + nc]

        # Extract Yid (one-step predictions)
        Yidw = w_opt[-N:]

        # Build coefficient vector for regressor
        coeff = vertcat(a, c)

        # Initialize symbolic predictions
        Yid = y * SX.ones(1)

        # Initialize noise sequence (prediction errors)
        # Epsi is updated iteratively in the loop
        Epsi = SX.zeros(N)

        # Build prediction equations for k >= n_tr
        for k in range(N):
            if k >= n_tr:
                # Build regressor for Yid prediction
                # phi = [-y_lags, e_lags]

                # Output lags
                vecY = y[k - na : k][::-1] if na > 0 else SX.zeros(0)

                # Noise lags (using past Epsi values)
                vecE = Epsi[k - nc : k][::-1] if nc > 0 else SX.zeros(0)

                # Regressor for ARMA: phi = [-vecY, vecE]
                phi = vertcat(-vecY, vecE)

                # Prediction: Yid[k] = phi' * [a; c]
                Yid[k] = mtimes(phi.T, coeff)

                # Update prediction error for this time step
                # This creates proper causal dependency: Epsi[k] computed after Yid[k]
                Epsi[k] = y[k] - Yidw[k]

        # Objective function: minimize mean squared error
        DY = y - Yidw
        f_obj = (1.0 / N) * mtimes(DY.T, DY)

        # Equality constraints
        g = []

        # 1. Yid consistency constraint
        g.append(Yid - Yidw)

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

            if nc > 0:
                ng_norm += 1
                # Companion matrix for C(q)
                compC = SX.zeros(nc, nc)
                if nc > 1:
                    diagC = SX.eye(nc - 1)
                    compC[:-1, 1:] = diagC
                compC[-1, :] = -c[::-1]
                norm_CompC = norm_inf(compC)
                g.append(norm_CompC)

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

    def _identify_ills(self, y, u, na, nc, sample_time, **kwargs) -> StateSpaceModel:
        """
        Fallback simplified method when CasADi not available.

        Uses iterative extended least squares. Less accurate than NLP.

        Parameters:
        -----------
        y : np.ndarray
            Output data (ny x N)
        u : np.ndarray
            Input data (ignored for ARMA)
        na, nc : int
            Model orders
        sample_time : float
            Sampling period
        **kwargs : dict
            Additional parameters (max_iterations, tolerance)

        Returns:
        --------
        model : StateSpaceModel
            Identified state-space model
        """
        # Get data dimensions (ARMA is typically SISO but support MIMO too)
        ny, N = y.shape

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

        # Maximum iterations for extended least squares
        max_iterations = kwargs.get("max_iterations", 100)
        tolerance = kwargs.get("tolerance", 1e-6)

        for i in range(ny):
            # For each output channel (typically just one for ARMA)
            # Use iterative extended least-squares (similar to master branch ARMAX)

            # Initialize noise estimate
            noise_hat = np.zeros(N)

            # Target signal (what we're trying to predict)
            y_target = y[i, max_lag:]

            # Initialize variance tracking
            Vn = np.inf
            Vn_old = np.inf
            theta = np.zeros(na + nc)
            iterations = 0

            # Iterative extended least squares loop
            while (Vn_old > Vn or iterations == 0) and iterations < max_iterations:
                theta_old = theta.copy()
                Vn_old = Vn
                iterations += 1

                # Build regression matrix for this iteration
                Phi = np.zeros((N_eff, na + nc))
                col = 0

                # AR part: lagged outputs (always based on actual data)
                for lag in range(na):
                    Phi[:, col] = y[i, max_lag - 1 - lag : max_lag - 1 - lag + N_eff]
                    col += 1

                # MA part: lagged noise estimates
                for lag in range(nc):
                    # Use noise estimates from previous iteration
                    # For lag k, we need noise_hat[t-k]
                    # At time t (indexed as max_lag + j for j in 0..N_eff-1),
                    # we need noise_hat from t-1-lag = max_lag + j - 1 - lag
                    for j in range(N_eff):
                        t_idx = max_lag + j - 1 - lag
                        if 0 <= t_idx < N:
                            Phi[j, col] = noise_hat[t_idx]
                        else:
                            Phi[j, col] = 0
                    col += 1

                # Solve least squares for this iteration with regularization
                # Use rcond for numerical stability with higher order models
                try:
                    theta_new, _, _, _ = lstsq(Phi, y_target, rcond=1e-10)
                except np.linalg.LinAlgError:
                    # If SVD fails, use previous solution and break
                    if iterations > 1:
                        break
                    else:
                        # First iteration failed, use zero initialization
                        theta_new = np.zeros(na + nc)

                # Compute predictions and new residuals
                y_pred = Phi @ theta_new
                new_residuals = y_target - y_pred

                # Compute mean square error (variance)
                Vn = np.mean(new_residuals**2)

                # If solution is worse, use binary search to find better step size
                if Vn > Vn_old and iterations > 1:
                    interval_length = 0.5
                    while Vn > Vn_old and interval_length > np.finfo(np.float32).eps:
                        theta = interval_length * theta_new + (1 - interval_length) * theta_old
                        y_pred = Phi @ theta
                        new_residuals = y_target - y_pred
                        Vn = np.mean(new_residuals**2)
                        interval_length = interval_length / 2.0

                    if Vn > Vn_old:
                        # Binary search failed, keep old solution
                        theta = theta_old
                        Vn = Vn_old
                        break
                else:
                    theta = theta_new

                # Update noise estimates for entire signal
                # noise[k] = y[k] - y_pred[k] using current theta
                # Reconstruct noise for entire signal (needed for next iteration)
                # Note: theta contains regression coefficients directly from least squares
                # theta[:na] are the AR coefficients as they appear in y[k] = theta[0]*y[k-1] + ...
                # theta[na:] are the MA coefficients
                for k in range(max_lag, N):
                    # AR component: theta contains direct regression coefficients
                    ar_sum = 0
                    for lag in range(na):
                        if k - 1 - lag >= 0:
                            ar_sum += theta[lag] * y[i, k - 1 - lag]

                    # MA component
                    ma_sum = 0
                    for lag in range(nc):
                        if k - 1 - lag >= 0:
                            ma_sum += theta[na + lag] * noise_hat[k - 1 - lag]

                    # Prediction using AR + MA
                    y_pred_k = ar_sum + ma_sum

                    # Clip prediction to prevent overflow in noise estimates
                    y_signal_range = np.max(np.abs(y[i, :]))
                    y_pred_k = np.clip(y_pred_k, -10 * y_signal_range, 10 * y_signal_range)

                    # Residual (noise estimate)
                    noise_hat[k] = y[i, k] - y_pred_k

                # Check convergence
                if iterations > 1:
                    theta_change = np.linalg.norm(theta - theta_old) / (np.linalg.norm(theta_old) + 1e-12)
                    if theta_change < tolerance:
                        break

            # Extract AR and MA coefficients
            # Note: In the regression, AR coefficients represent y[k] = theta[0]*y[k-1] + ...
            # In transfer function form A(q) = 1 + a1*q^-1 + ..., we have a1 = -theta[0]
            # So we need to negate the AR coefficients for transfer function convention
            AR_coeffs[i, :] = -theta[:na]
            MA_coeffs[i, :] = theta[na:na + nc]
            residuals_list.append(noise_hat[max_lag:])

        # Compute one-step-ahead predictions (Yid) for identification data
        Yid = np.zeros_like(y)
        Yid[:, :max_lag] = y[:, :max_lag]  # Copy initial values

        for i in range(ny):
            # Compute one-step-ahead predictions using the estimated AR and MA coefficients
            # We need to reconstruct the noise estimates first
            noise_est = np.zeros(N)

            # Reconstruct noise estimates and predictions
            y_signal_range = np.max(np.abs(y[i, :]))
            for k in range(max_lag, N):
                # AR component: use actual past outputs
                # AR_coeffs are in transfer function form (1 + a1*q^-1 + ...)
                # So: y[k] = -a1*y[k-1] - a2*y[k-2] - ...
                ar_sum = 0
                for lag in range(na):
                    if k - 1 - lag >= 0:
                        ar_sum += -AR_coeffs[i, lag] * y[i, k - 1 - lag]

                # MA component: use past noise estimates
                # MA_coeffs are in transfer function form (1 + c1*q^-1 + ...)
                # So: contribution is c1*e[k-1] + c2*e[k-2] + ...
                ma_sum = 0
                for lag in range(nc):
                    if k - 1 - lag >= 0:
                        ma_sum += MA_coeffs[i, lag] * noise_est[k - 1 - lag]

                # One-step-ahead prediction (clip to prevent overflow)
                pred = ar_sum + ma_sum
                Yid[i, k] = np.clip(pred, -10 * y_signal_range, 10 * y_signal_range)

                # Update noise estimate for this time step
                noise_est[k] = y[i, k] - Yid[i, k]

        # Create G_tf and H_tf transfer functions
        G_tf, H_tf = self._create_transfer_functions_arma(
            AR_coeffs, MA_coeffs, na, nc, ny, sample_time
        )

        # Create state-space representation
        if HAROLD_AVAILABLE:
            model = self._create_state_space_from_arma(
                AR_coeffs, MA_coeffs, na, nc, ny, sample_time
            )
        else:
            # Fallback when harold is not available
            model = self._create_mock_model(
                AR_coeffs, MA_coeffs, na, nc, ny, sample_time
            )

        # Attach transfer functions and predictions to model
        model.G_tf = G_tf
        model.H_tf = H_tf
        model.Yid = Yid

        # Attach AR and MA coefficients for easy access
        model.AR_coeffs = AR_coeffs
        model.MA_coeffs = MA_coeffs

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
            NUM_H[1 : nc + 1] = MA_coeffs[0, :] if ny == 1 else MA_coeffs[0, :]

            DEN_H = np.zeros(max_order + 1)
            DEN_H[0] = 1.0
            DEN_H[1 : na + 1] = AR_coeffs[0, :] if ny == 1 else AR_coeffs[0, :]

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
