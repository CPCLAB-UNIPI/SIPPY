"""
ARMAX algorithm mode handlers for different identification approaches.
"""

import warnings
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np
from scipy.optimize import minimize

from ..base import StateSpaceModel

# Import Harold if available
try:
    import harold

    HAROLD_AVAILABLE = True
except ImportError:
    HAROLD_AVAILABLE = False


class ARMAXModeHandler(ABC):
    """Abstract base class for ARMAX algorithm mode handlers."""

    @abstractmethod
    def identify(
        self,
        u: np.ndarray,
        y: np.ndarray,
        na: int,
        nb: int,
        nc: int,
        nk: int,
        max_iterations: int = 200,
        convergence_tolerance: float = 1e-6,
        **kwargs,
    ) -> Tuple[Optional[StateSpaceModel], dict]:
        """
        Perform ARMAX identification using the specific algorithm mode.

        Parameters:
        -----------
        u, y : ndarray
            Input and output data
        na, nb, nc : int
            Model orders
        nk : int
            Input delay
        max_iterations : int
            Maximum number of iterations
        convergence_tolerance : float
            Convergence tolerance for iterative methods
        **kwargs
            Mode-specific parameters

        Returns:
        --------
        model : StateSpaceModel or None
            Identified state-space model
        info : dict
            Additional information about the identification process
        """
        pass

    @abstractmethod
    def validate_parameters(self, **kwargs) -> bool:
        """Validate mode-specific parameters."""
        pass


class ILLSHandler(ARMAXModeHandler):
    """Iterative Least Squares ARMAX handler."""

    def validate_parameters(self, **kwargs) -> bool:
        """Validate ILLS-specific parameters."""
        # ILLS is relatively tolerant of parameters
        return True

    def identify(
        self,
        u: np.ndarray,
        y: np.ndarray,
        na: int,
        nb: int,
        nc: int,
        nk: int,
        max_iterations: int = 200,
        convergence_tolerance: float = 1e-6,
        **kwargs,
    ) -> Tuple[Optional[StateSpaceModel], dict]:
        """Identify ARMAX using Iterative Least Squares."""
        return self._identify_ills(
            u, y, na, nb, nc, nk, max_iterations, convergence_tolerance
        )

    def _identify_ills(
        self,
        u: np.ndarray,
        y: np.ndarray,
        na: int,
        nb: int,
        nc: int,
        nk: int,
        max_iterations: int,
        convergence_tolerance: float,
    ) -> Tuple[Optional[StateSpaceModel], dict]:
        """Identify ARMAX model using ILLS algorithm."""

        # Input validation
        if y.size != u.size:
            raise ValueError("Input and output must have same length")

        ny, N = 1, y.size  # Assume SISO for now

        # Check data length
        max_order = max(na, nb + nk, nc)
        if N <= max_order:
            return None, {"error": "Insufficient data length"}

        # Initialize variables from master branch algorithm
        sum_order = sum([na, nb, nc])
        N_eff = N - max_order

        # Initialize noise estimate
        noise_hat = np.zeros(N)

        # Build regression matrix
        Phi = np.zeros((N_eff, sum_order))

        Vn, Vn_old = np.inf, np.inf
        beta_hat = np.zeros(sum_order)
        I_beta = np.identity(beta_hat.size)
        iterations = 0
        max_reached = False

        # ILLS iterative loop from master branch
        while (Vn_old > Vn or iterations == 0) and iterations < max_iterations:
            beta_hat_old = beta_hat
            Vn_old = Vn
            iterations += 1

            # Update regression matrix with current noise estimate
            for i in range(N_eff):
                # AR part (lagged outputs)
                Phi[i, 0:na] = -y[i + max_order - 1 :: -1][0:na]
                # X part (lagged inputs)
                Phi[i, na : na + nb] = u[max_order + i - 1 :: -1][nk : nb + nk]
                # MA part (estimated noise terms)
                Phi[i, na + nb : na + nb + nc] = noise_hat[max_order + i - 1 :: -1][
                    0:nc
                ]

            # Least squares solution
            beta_hat = np.dot(np.linalg.pinv(Phi), y[max_order:N])
            Vn = np.mean((y[max_order:N] - np.dot(Phi, beta_hat)) ** 2)

            # Binary search fallback if solution not improving
            beta_hat_new = beta_hat
            interval_length = 0.5
            while Vn > Vn_old:
                beta_hat = np.dot(I_beta * interval_length, beta_hat_new) + np.dot(
                    I_beta * (1 - interval_length), beta_hat_old
                )
                Vn = np.mean((y[max_order:N] - np.dot(Phi, beta_hat)) ** 2)

                if interval_length < np.finfo(np.float32).eps:
                    beta_hat = beta_hat_old
                    Vn = Vn_old
                    break
                interval_length = interval_length / 2.0

            # Update noise estimate
            if iterations < max_iterations:
                predicted_output = np.dot(Phi, beta_hat)
                noise_hat[max_order:N] = y[max_order:N] - predicted_output

        if iterations >= max_iterations:
            warnings.warn("ARMAX ILLS: Reached maximum iterations.")
            max_reached = True

        # Compute one-step-ahead predictions for full data
        Yid = np.hstack((y[:max_order], np.dot(Phi, beta_hat)))

        # Create model from parameters
        try:
            model = self._create_state_space_model(beta_hat, na, nb, nc, nk, ny, 1, 1.0)
            if model is None:
                raise ValueError("Failed to create state-space model")

            # Attach one-step-ahead predictions to model
            model.Yid = Yid.reshape(1, -1)  # Shape: (1, N) for SISO

        except Exception as e:
            return None, {"error": f"Model creation failed: {str(e)}"}

        # Return results
        info = {
            "iterations": iterations,
            "max_reached": max_reached,
            "final_variance": Vn,
            "converged": not max_reached and abs(Vn_old - Vn) < convergence_tolerance,
            "predicted_output": Yid,
            "residuals": noise_hat,
        }

        return model, info

    def _create_state_space_model(
        self,
        beta_hat: np.ndarray,
        na: int,
        nb: int,
        nc: int,
        nk: int,
        ny: int,
        nu: int,
        Ts: float,
    ) -> Optional[StateSpaceModel]:
        """Create state-space model from ARMAX parameters."""
        try:
            # Extract coefficients
            A_coeffs = beta_hat[:na]
            B_coeffs = beta_hat[na : na + nb]
            C_coeffs = beta_hat[na + nb : na + nb + nc]

            # Create transfer functions using harold
            G_tf, H_tf = None, None
            if HAROLD_AVAILABLE:
                try:
                    # Determine maximum order for transfer function arrays
                    max_order = max(na, nb + nk, nc)

                    # G(q) = B / A - Deterministic transfer function
                    NUM_G = np.zeros(max_order)
                    NUM_G[nk : nk + nb] = B_coeffs  # B coefficients with delay

                    DEN_G = np.zeros(max_order + 1)
                    DEN_G[0] = 1.0
                    DEN_G[1 : na + 1] = A_coeffs  # A coefficients

                    G_tf = harold.Transfer(NUM_G, DEN_G, dt=Ts)

                    # H(q) = C / A - Noise transfer function
                    NUM_H = np.zeros(max_order + 1)
                    NUM_H[0] = 1.0
                    NUM_H[1 : nc + 1] = C_coeffs  # C coefficients

                    DEN_H = np.zeros(max_order + 1)
                    DEN_H[0] = 1.0
                    DEN_H[1 : na + 1] = A_coeffs  # A coefficients (same as G)

                    H_tf = harold.Transfer(NUM_H, DEN_H, dt=Ts)
                except Exception as e:
                    warnings.warn(
                        f"Failed to create ARMAX transfer functions with harold: {e}"
                    )
                    G_tf, H_tf = None, None

            # Create state-space representation
            n_states = na + nc

            # A matrix (companion form)
            A_mat = np.zeros((n_states, n_states))
            if na > 1:
                for i in range(na - 1):
                    A_mat[i, i + 1] = 1.0
            if na > 0:
                A_mat[na - 1, :na] = -A_coeffs

            # Add MA dynamics
            if nc > 0:
                A_mat[na:, na:] = np.eye(nc)
                A_mat[:na, na:] = np.zeros((na, nc))

            # B matrix
            B_mat = np.zeros((n_states, nu))
            if nu > 0:
                B_mat[:na, 0] = B_coeffs

            # C matrix
            C_mat = np.zeros((ny, n_states))
            if na > 0:
                C_mat[0, :na] = 1.0
            if nc > 0:
                C_mat[0, na:] = C_coeffs

            # D matrix
            D_mat = np.zeros((ny, nu))

            # Return state-space model with transfer functions
            if HAROLD_AVAILABLE:
                # Use Harold for consistent state-space creation
                try:
                    ss_model = harold.StateSpace(A_mat, B_mat, C_mat, D_mat, dt=Ts)
                    return StateSpaceModel(
                        A=ss_model.a,
                        B=ss_model.b,
                        C=ss_model.c,
                        D=ss_model.d,
                        K=np.zeros((ss_model.a.shape[0], ss_model.c.shape[0])),
                        Q=np.eye(ss_model.a.shape[0]) * 0.01,
                        R=np.eye(ss_model.c.shape[0]) * 0.01,
                        S=np.zeros((ss_model.a.shape[0], ss_model.c.shape[0])),
                        ts=Ts,
                        Vn=0.01,
                        G_tf=G_tf,
                        H_tf=H_tf,
                    )
                except Exception:
                    pass  # Fall back to manual creation

            # Manual state-space creation
            return StateSpaceModel(
                A=A_mat,
                B=B_mat,
                C=C_mat,
                D=D_mat,
                K=np.zeros((A_mat.shape[0], C_mat.shape[0])),
                Q=np.eye(A_mat.shape[0]) * 0.01,
                R=np.eye(C_mat.shape[0]) * 0.01,
                S=np.zeros((A_mat.shape[0], C_mat.shape[0])),
                ts=Ts,
                Vn=0.01,
                G_tf=G_tf,
                H_tf=H_tf,
            )

        except Exception:
            return None


class RLLSHandler(ARMAXModeHandler):
    """Recursive Least Squares ARMAX handler."""

    def validate_parameters(self, **kwargs) -> bool:
        """Validate RLLS-specific parameters."""
        forgetting_factor = kwargs.get("forgetting_factor", 1.0)
        if not 0.5 <= forgetting_factor <= 1.0:
            raise ValueError("Forgetting factor must be between 0.5 and 1.0")
        return True

    def identify(
        self,
        u: np.ndarray,
        y: np.ndarray,
        na: int,
        nb: int,
        nc: int,
        nk: int,
        max_iterations: int = 200,
        convergence_tolerance: float = 1e-6,
        **kwargs,
    ) -> Tuple[Optional[StateSpaceModel], dict]:
        """Identify ARMAX using Recursive Least Squares."""
        return self._identify_rlls(u, y, na, nb, nc, nk, max_iterations, **kwargs)

    def _identify_rlls(
        self,
        u: np.ndarray,
        y: np.ndarray,
        na: int,
        nb: int,
        nc: int,
        nk: int,
        max_iterations: int,
        convergence_tolerance: float = 1e-6,
        **kwargs,
    ) -> Tuple[Optional[StateSpaceModel], dict]:
        """Identify ARMAX model using RLLS algorithm from master branch."""

        # Input validation
        if y.size != u.size:
            raise ValueError("Input and output must have same length")

        # RLLS parameters from master branch
        forgetting_factor = kwargs.get("forgetting_factor", 1.0)
        ny = 1  # Assume SISO
        nu = 1
        N = y.size

        max_order = max(na, nb + nk, nc)
        nt = na + nb + nc + 1  # Total number of parameters

        if N <= max_order:
            return None, {"error": "Insufficient data for RLLS"}

        # Initialize RLLS variables from master branch
        Beta = 1e4  # Confidence parameter
        P_t = Beta * np.eye(nt - 1, nt - 1)  # Covariance matrix
        theta = np.zeros((nt - 1))  # Parameter vector
        eta = np.zeros(N)  # Bias term
        Yp = np.zeros(N)  # Predicted output
        E = np.zeros(N)  # Error

        # Recursive loop from master branch
        for k in range(N):
            if k > max_order:
                # Step 1: Build regressor vector
                vecY = y[k - na : k][::-1]  # Y vector
                vecU = u[k - nb - nk : k - nk][::-1]  # U vector
                vecE = E[k - nc : k][::-1]  # E vector

                # ARMAX regressor
                phi = np.hstack((-vecY, vecU, vecE))

                # Step 2: Gain update
                try:
                    K_t = np.dot(
                        np.dot(P_t, phi),
                        np.linalg.inv(
                            forgetting_factor + np.dot(np.dot(phi.T, P_t), phi)
                        ),
                    )
                except np.linalg.LinAlgError:
                    # Handle singular matrix
                    K_t = np.zeros(nt - 1)

                # Step 3: Parameter update
                theta = theta + np.dot(K_t, (y[k] - np.dot(phi.T, theta)))

                # Step 4: A posteriori prediction
                Yp[k] = np.dot(phi.T, theta) + eta[k]
                E[k] = y[k] - Yp[k]

                # Step 5: Covariance update
                try:
                    P_t = (1.0 / forgetting_factor) * (
                        np.dot(
                            np.eye(nt - 1)
                            - np.dot(K_t.reshape(-1, 1), phi.T.reshape(1, -1)),
                            P_t,
                        )
                    )
                except Exception:
                    # Keep P_t if update fails
                    pass

        # Calculate final variance
        Vn = (np.linalg.norm(y - Yp) ** 2) / (2 * (N - max_order))

        # Create model from final parameters
        try:
            model = self._create_state_space_model_rlls(
                theta, na, nb, nc, nk, ny, nu, 1.0
            )
            if model is None:
                raise ValueError("Failed to create RLLS model")

            # Attach one-step-ahead predictions to model
            model.Yid = Yp.reshape(1, -1)  # Shape: (1, N) for SISO

        except Exception as e:
            return None, {"error": f"RLLS model creation failed: {str(e)}"}

        info = {
            "final_variance": Vn,
            "predicted_output": Yp,
            "residuals": E,
            "final_parameters": theta,
        }

        return model, info

    def _create_state_space_model_rlls(
        self,
        theta: np.ndarray,
        na: int,
        nb: int,
        nc: int,
        nk: int,
        ny: int,
        nu: int,
        Ts: float,
    ) -> Optional[StateSpaceModel]:
        """Create state-space model from RLLS parameters."""
        try:
            # Extract coefficients from theta (same as ILLS)
            # theta contains [-a, b, c] parameters
            pos = 0
            A_coeffs = theta[pos : pos + na]
            pos += na
            B_coeffs = theta[pos : pos + nb]
            pos += nb
            C_coeffs = theta[pos : pos + nc]

            # Create transfer functions using harold
            G_tf, H_tf = None, None
            if HAROLD_AVAILABLE:
                try:
                    # Determine maximum order for transfer function arrays
                    max_order = max(na, nb + nk, nc)

                    # G(q) = B / A - Deterministic transfer function
                    NUM_G = np.zeros(max_order)
                    NUM_G[nk : nk + nb] = B_coeffs  # B coefficients with delay

                    DEN_G = np.zeros(max_order + 1)
                    DEN_G[0] = 1.0
                    DEN_G[1 : na + 1] = A_coeffs  # A coefficients

                    G_tf = harold.Transfer(NUM_G, DEN_G, dt=Ts)

                    # H(q) = C / A - Noise transfer function
                    NUM_H = np.zeros(max_order + 1)
                    NUM_H[0] = 1.0
                    NUM_H[1 : nc + 1] = C_coeffs  # C coefficients

                    DEN_H = np.zeros(max_order + 1)
                    DEN_H[0] = 1.0
                    DEN_H[1 : na + 1] = A_coeffs  # A coefficients (same as G)

                    H_tf = harold.Transfer(NUM_H, DEN_H, dt=Ts)
                except Exception as e:
                    warnings.warn(
                        f"Failed to create RLLS ARMAX transfer functions with harold: {e}"
                    )
                    G_tf, H_tf = None, None

            # Use same state-space creation as ILLS
            n_states = na + nc

            # A matrix (companion form)
            A_mat = np.zeros((n_states, n_states))
            if na > 1:
                for i in range(na - 1):
                    A_mat[i, i + 1] = 1.0
            if na > 0:
                A_mat[na - 1, :na] = -A_coeffs

            # Add MA dynamics
            if nc > 0:
                A_mat[na:, na:] = np.eye(nc)
                A_mat[:na, na:] = np.zeros((na, nc))

            # B matrix
            B_mat = np.zeros((n_states, nu))
            if nu > 0:
                B_mat[:na, 0] = B_coeffs

            # C matrix
            C_mat = np.zeros((ny, n_states))
            if na > 0:
                C_mat[0, :na] = 1.0
            if nc > 0:
                C_mat[0, na:] = C_coeffs

            # D matrix
            D_mat = np.zeros((ny, nu))

            # Return state-space model with transfer functions
            if HAROLD_AVAILABLE:
                try:
                    ss_model = harold.StateSpace(A_mat, B_mat, C_mat, D_mat, dt=Ts)
                    return StateSpaceModel(
                        A=ss_model.a,
                        B=ss_model.b,
                        C=ss_model.c,
                        D=ss_model.d,
                        K=np.zeros((ss_model.a.shape[0], ss_model.c.shape[0])),
                        Q=np.eye(ss_model.a.shape[0]) * 0.01,
                        R=np.eye(ss_model.c.shape[0]) * 0.01,
                        S=np.zeros((ss_model.a.shape[0], ss_model.c.shape[0])),
                        ts=Ts,
                        Vn=0.01,
                        G_tf=G_tf,
                        H_tf=H_tf,
                    )
                except Exception:
                    pass

            # Manual state-space creation
            return StateSpaceModel(
                A=A_mat,
                B=B_mat,
                C=C_mat,
                D=D_mat,
                K=np.zeros((A_mat.shape[0], C_mat.shape[0])),
                Q=np.eye(A_mat.shape[0]) * 0.01,
                R=np.eye(C_mat.shape[0]) * 0.01,
                S=np.zeros((A_mat.shape[0], C_mat.shape[0])),
                ts=Ts,
                Vn=0.01,
                G_tf=G_tf,
                H_tf=H_tf,
            )

        except Exception:
            return None


class OPTHandler(ARMAXModeHandler):
    """Optimization-based ARMAX handler."""

    def validate_parameters(self, **kwargs) -> bool:
        """Validate OPT-specific parameters."""
        opt_method = kwargs.get("optimization_method", "trust-constr")
        valid_methods = ["trust-constr", "SLSQP", "BFGS"]
        if opt_method not in valid_methods:
            raise ValueError(f"Optimization method must be one of {valid_methods}")
        return True

    def identify(
        self,
        u: np.ndarray,
        y: np.ndarray,
        na: int,
        nb: int,
        nc: int,
        nk: int,
        max_iterations: int = 200,
        convergence_tolerance: float = 1e-6,
        **kwargs,
    ) -> Tuple[Optional[StateSpaceModel], dict]:
        """Identify ARMAX using nonlinear optimization."""
        return self._identify_opt(u, y, na, nb, nc, nk, max_iterations, **kwargs)

    def _identify_opt(
        self,
        u: np.ndarray,
        y: np.ndarray,
        na: int,
        nb: int,
        nc: int,
        nk: int,
        max_iterations: int,
        convergence_tolerance: float = 1e-6,
        **kwargs,
    ) -> Tuple[Optional[StateSpaceModel], dict]:
        """Identify ARMAX model using optimization from master branch."""

        # Input validation
        if y.size != u.size:
            raise ValueError("Input and output must have same length")

        opt_method = kwargs.get("optimization_method", "trust-constr")
        ny = 1  # SISO
        nu = 1
        N = y.size
        n_params = na + nb + nc

        max_order = max(na, nb + nk, nc)
        if N <= max_order:
            return None, {"error": "Insufficient data for OPT"}

        # Cost function from master branch
        def cost_function(params):
            """Prediction error cost function."""
            try:
                # Extract parameters
                A_params = params[:na]
                B_params = params[na : na + nb]
                C_params = params[na + nb : na + nb + nc]

                # Simulate model and calculate errors
                predicted = np.zeros(N)
                for k in range(max_order, N):
                    # AR part
                    ar_part = np.sum(A_params * y[k - 1 : k - 1 - na : -1])

                    # X part (with delay)
                    start_idx = k - nk
                    if start_idx >= nb - 1:
                        x_part = np.sum(B_params * u[start_idx : start_idx - nb : -1])
                    else:
                        x_part = 0.0

                    # MA part (using residuals) - with numerical stability checks
                    residuals = y - predicted
                    if np.max(np.abs(residuals)) > 1e6:  # Check for residuals explosion
                        return np.inf

                    ma_part = np.sum(C_params * residuals[k - 1 : k - 1 - nc : -1])

                    # Check for overflow in predicted value
                    if not np.isfinite(-ar_part + x_part + ma_part):
                        return np.inf

                    predicted[k] = -ar_part + x_part + ma_part

                # Return prediction error with numerical stability
                error = y[max_order:] - predicted[max_order:]

                # Check for overflow before squaring
                if np.max(np.abs(error)) > 1e6:
                    return np.inf

                cost = np.sum(error**2)

                # Final check for finite cost
                if not np.isfinite(cost):
                    return np.inf

                return cost

            except Exception:
                return np.inf

        # Initial guess using least squares - more conservative approach
        try:
            # Get initial guess from simple least squares
            initial_guess = self._get_initial_guess(u, y, na, nb, nc, nk)
            if (
                initial_guess is None
                or np.any(np.isnan(initial_guess))
                or np.any(np.abs(initial_guess) > 2)
            ):
                # Use small random values if initial guess is problematic
                initial_guess = np.random.randn(n_params) * 0.01
        except Exception:
            # Use very small random values as fallback
            initial_guess = np.random.randn(n_params) * 0.01

        # Parameter bounds - more conservative to prevent overflow
        bounds = [(-5, 5)] * n_params  # Tighter bounds for numerical stability

        # Optimize
        try:
            # Test cost function with initial guess first
            try:
                test_cost = cost_function(initial_guess)
                if not np.isfinite(test_cost):
                    return None, {"error": f"Initial cost is infinite: {test_cost}"}
            except Exception as cost_e:
                return None, {
                    "error": f"Cost function evaluation failed: {str(cost_e)}"
                }

            # Try the requested optimization method first
            result = minimize(
                cost_function,
                x0=initial_guess,
                method=opt_method,
                bounds=bounds,
                options={"maxiter": max_iterations, "ftol": convergence_tolerance},
            )

        except Exception as e:
            # Try a simpler method if the first one fails
            try:
                result = minimize(
                    cost_function,
                    x0=initial_guess,
                    method="L-BFGS-B",  # Simpler, more robust method
                    bounds=bounds,
                    options={
                        "maxiter": max_iterations // 2,
                        "ftol": convergence_tolerance,
                    },
                )
            except Exception as e2:
                return None, {
                    "error": f"Both optimization methods failed: {str(e)}, {str(e2)}"
                }

        if not result.success:
            return None, {"error": f"Optimization failed: {result.message}"}

        # Create model from optimal parameters
        try:
            model = self._create_state_space_model_opt(
                result.x, na, nb, nc, nk, ny, nu, 1.0
            )
            if model is None:
                raise ValueError("Failed to create OPT model")
        except Exception as e:
            return None, {"error": f"OPT model creation failed: {str(e)}"}

        info = {
            "optimization_successful": result.success,
            "objective_value": result.fun,
            "iterations": result.nit,
            "final_parameters": result.x,
            "message": result.message,
        }

        return model, info

    def _get_initial_guess(
        self, u: np.ndarray, y: np.ndarray, na: int, nb: int, nc: int, nk: int
    ) -> Optional[np.ndarray]:
        """Get initial parameter guess using least squares."""
        try:
            max_order = max(na, nb + nk, nc)
            N = y.size

            if N <= max_order:
                return None

            N_eff = N - max_order
            sum_order = na + nb

            # Build regression matrix for ARX part (no MA)
            Phi = np.zeros((N_eff, sum_order))
            for i in range(N_eff):
                # AR part
                Phi[i, 0:na] = -y[i + max_order - 1 :: -1][0:na]
                # X part
                Phi[i, na : na + nb] = u[max_order + i - 1 :: -1][nk : nb + nk]

            # Least squares solution
            beta_hat = np.dot(np.linalg.pinv(Phi), y[max_order:N])

            # Return initial guess with zero MA parameters
            return np.concatenate([beta_hat[:na], beta_hat[na : na + nb], np.zeros(nc)])

        except Exception:
            return None

    def _create_state_space_model_opt(
        self,
        params: np.ndarray,
        na: int,
        nb: int,
        nc: int,
        nk: int,
        ny: int,
        nu: int,
        Ts: float,
    ) -> Optional[StateSpaceModel]:
        """Create state-space model from OPT parameters."""
        try:
            # Extract coefficients
            A_coeffs = params[:na]
            B_coeffs = params[na : na + nb]
            C_coeffs = params[na + nb : na + nb + nc]

            # Create transfer functions using harold
            G_tf, H_tf = None, None
            if HAROLD_AVAILABLE:
                try:
                    max_order = max(na, nb + nk, nc)
                    NUM_G = np.zeros(max_order)
                    NUM_G[nk : nk + nb] = B_coeffs
                    DEN_G = np.zeros(max_order + 1)
                    DEN_G[0] = 1.0
                    DEN_G[1 : na + 1] = A_coeffs
                    G_tf = harold.Transfer(NUM_G, DEN_G, dt=Ts)

                    NUM_H = np.zeros(max_order + 1)
                    NUM_H[0] = 1.0
                    NUM_H[1 : nc + 1] = C_coeffs
                    DEN_H = np.zeros(max_order + 1)
                    DEN_H[0] = 1.0
                    DEN_H[1 : na + 1] = A_coeffs
                    H_tf = harold.Transfer(NUM_H, DEN_H, dt=Ts)
                except Exception as e:
                    warnings.warn(
                        f"Failed to create OPT ARMAX transfer functions with harold: {e}"
                    )
                    G_tf, H_tf = None, None

            # Use same state-space creation as ILLS
            n_states = na + nc

            # A matrix (companion form)
            A_mat = np.zeros((n_states, n_states))
            if na > 1:
                for i in range(na - 1):
                    A_mat[i, i + 1] = 1.0
            if na > 0:
                A_mat[na - 1, :na] = -A_coeffs

            # Add MA dynamics
            if nc > 0:
                A_mat[na:, na:] = np.eye(nc)
                A_mat[:na, na:] = np.zeros((na, nc))

            # B matrix (account for delay)
            B_mat = np.zeros((n_states, nu))
            if nu > 0 and nb > 0:
                B_mat[na - 1, 0] = B_coeffs[0]  # Place first B coefficient

            # C matrix
            C_mat = np.zeros((ny, n_states))
            if na > 0:
                C_mat[0, :na] = 1.0
            if nc > 0:
                C_mat[0, na:] = C_coeffs

            # D matrix
            D_mat = np.zeros((ny, nu))

            # Return state-space model with transfer functions
            if HAROLD_AVAILABLE:
                try:
                    ss_model = harold.StateSpace(A_mat, B_mat, C_mat, D_mat, dt=Ts)
                    return StateSpaceModel(
                        A=ss_model.a,
                        B=ss_model.b,
                        C=ss_model.c,
                        D=ss_model.d,
                        K=np.zeros((ss_model.a.shape[0], ss_model.c.shape[0])),
                        Q=np.eye(ss_model.a.shape[0]) * 0.01,
                        R=np.eye(ss_model.c.shape[0]) * 0.01,
                        S=np.zeros((ss_model.a.shape[0], ss_model.c.shape[0])),
                        ts=Ts,
                        Vn=0.01,
                        G_tf=G_tf,
                        H_tf=H_tf,
                    )
                except Exception:
                    pass

            # Manual state-space creation
            return StateSpaceModel(
                A=A_mat,
                B=B_mat,
                C=C_mat,
                D=D_mat,
                K=np.zeros((A_mat.shape[0], C_mat.shape[0])),
                Q=np.eye(A_mat.shape[0]) * 0.01,
                R=np.eye(C_mat.shape[0]) * 0.01,
                S=np.zeros((A_mat.shape[0], C_mat.shape[0])),
                ts=Ts,
                Vn=0.01,
                G_tf=G_tf,
                H_tf=H_tf,
            )

        except Exception:
            return None


def get_armax_handler(mode: str) -> ARMAXModeHandler:
    """Get the appropriate ARMAX mode handler."""
    handlers = {
        "ILLS": ILLSHandler(),
        "OPT": OPTHandler(),
        "RLLS": RLLSHandler(),
        "ILS": ILLSHandler(),  # Alias for legacy compatibility
    }

    if mode not in handlers:
        raise ValueError(
            f"Unknown ARMAX mode: {mode}. Available modes: {list(handlers.keys())}"
        )

    return handlers[mode]
