"""
Base classes for system identification algorithms.
"""
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Optional, Union

import numpy as np

if TYPE_CHECKING:
    from .iddata import IDData


class IdentificationAlgorithm(ABC):
    """Abstract base class for system identification algorithms."""

    def __init__(self):
        self.name = self.__class__.__name__

    @abstractmethod
    def identify(self, y: Optional[np.ndarray] = None, u: Optional[np.ndarray] = None,
                 iddata: Optional['IDData'] = None, **kwargs) -> 'StateSpaceModel':
        """
        Perform system identification.

        Args:
            y: Output data (outputs x time_steps) - alternative to iddata
            u: Input data (inputs x time_steps) - alternative to iddata
            iddata: IDData object containing input and output data
            **kwargs: Algorithm-specific parameters

        Returns:
            StateSpaceModel: Identified model

        Note:
            Either (y, u) or iddata should be provided, but not both.
        """
        pass

    @abstractmethod
    def validate_parameters(self, **kwargs) -> bool:
        """Validate algorithm-specific parameters."""
        pass


class StateSpaceModel:
    """Enhanced state-space model container."""

    def __init__(self, A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray,
                 K: np.ndarray, Q: np.ndarray, R: np.ndarray, S: np.ndarray,
                 ts: float, Vn: Union[float, np.ndarray]):
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.K = K
        self.Q = Q
        self.R = R
        self.S = S
        self.ts = ts
        self.Vn = Vn
        self.n = A.shape[0]  # State dimension

        # Try to import harold for State object
        try:
            from harold import State
            self.G = State(A, B, C, D, ts)
        except ImportError:
            self.G = None

        self.x0 = np.zeros((self.n, 1))

        # Calculate observer matrices if possible
        try:
            self.A_K = A - np.dot(K, C)
            self.B_K = B - np.dot(K, D)
        except (ValueError, IndexError, TypeError):
            self.A_K = np.array([])
            self.B_K = np.array([])

    def is_stable(self) -> bool:
        """Check if the system matrix A is stable."""
        try:
            eigenvals = np.linalg.eigvals(self.A)
            return np.all(np.abs(eigenvals) < 1.0)
        except (ValueError, np.linalg.LinAlgError):
            return False

    def get_natural_frequencies(self) -> np.ndarray:
        """Get natural frequencies of the system."""
        try:
            eigenvals = np.linalg.eigvals(self.A)
            return np.abs(np.angle(eigenvals) / (2 * np.pi * self.ts))
        except (ValueError, np.linalg.LinAlgError, ZeroDivisionError):
            return np.array([])

    def get_damping_ratios(self) -> np.ndarray:
        """Get damping ratios of the system."""
        try:
            eigenvals = np.linalg.eigvals(self.A)
            return -np.real(eigenvals) / np.abs(eigenvals)
        except (ValueError, np.linalg.LinAlgError, ZeroDivisionError):
            return np.array([])

    def get_fir_coefficients(self, inputs: list, outputs: list,
                           sampling: float, tss: float) -> dict:
        """
        Get FIR coefficients for the model.

        Parameters:
        -----------
        inputs : list
            List of input variable names
        outputs : list
            List of output variable names
        sampling : float
            Sampling rate in seconds
        tss : float
            Time to steady state in minutes

        Returns:
        --------
        fir_model : dict
            Nested dictionary of FIR coefficients
        """
        from ..utils.simulation_utils import get_fir_coef
        return get_fir_coef(self, inputs, outputs, sampling, tss)

    def get_step_response(self, inputs: list, outputs: list) -> dict:
        """
        Get step response for the model.

        Parameters:
        -----------
        inputs : list
            List of input variable names
        outputs : list
            List of output variable names

        Returns:
        --------
        step_response : dict
            Nested dictionary of step responses
        """
        from ..utils.simulation_utils import get_step_response
        fir_model = self.get_fir_coefficients(inputs, outputs, 1.0, 60)
        return get_step_response(fir_model)

    def get_model_uncertainty(self, input_data: np.ndarray, output_data: np.ndarray,
                            input_name: str, output_name: str) -> tuple:
        """
        Get model uncertainty analysis.

        Parameters:
        -----------
        input_data : np.ndarray
            Input signal data
        output_data : np.ndarray
            Output signal data
        input_name : str
            Input variable name
        output_name : str
            Output variable name

        Returns:
        --------
        tuple : (freqs, model_bode_mag, ci95, ci68, snr)
            Frequency response and confidence intervals
        """
        from ..utils.simulation_utils import get_model_uncertainty

        # Get FIR coefficients for this input-output pair
        fir_model = self.get_fir_coefficients([input_name], [output_name], 1.0, 60)
        model = fir_model[output_name][input_name]

        return get_model_uncertainty(input_data, output_data, model)

    def simulate(self, u: np.ndarray, x0: np.ndarray = None) -> tuple:
        """
        Simulate the state-space model.

        Parameters:
        -----------
        u : np.ndarray
            Input signals (inputs x time_steps)
        x0 : np.ndarray, optional
            Initial state

        Returns:
        --------
        x : np.ndarray
            State trajectory
        y : np.ndarray
            Output signals
        """
        from ..utils.simulation_utils import simulate_ss_system
        return simulate_ss_system(self.A, self.B, self.C, self.D, u, x0)

    def supports_optimization_methods(self) -> bool:
        """
        Check if the model supports various optimization methods.

        Returns:
        --------
        bool : True if optimization methods are supported
        """
        return True  # Most modern identification algorithms support optimization


class SystemIdentificationConfig:
    """Configuration container for system identification."""

    def __init__(self,
                 method: str = 'N4SID',
                 centering: str = 'None',
                 ic: str = 'None',
                 tsample: float = 1.0,
                 ss_f: int = 20,
                 ss_threshold: float = 0.1,
                 ss_max_order: Optional[int] = None,
                 ss_fixed_order: Optional[int] = 1,  # Default to 1 to avoid issues
                 ss_orders: List[int] = [1, 10],
                 ss_d_required: bool = False,
                 ss_a_stability: bool = False):
        self.method = method
        self.centering = centering
        self.ic = ic
        self.tsample = tsample
        self.ss_f = ss_f
        self.ss_threshold = ss_threshold
        self.ss_max_order = ss_max_order
        self.ss_fixed_order = ss_fixed_order
        self.ss_orders = ss_orders
        self.ss_d_required = ss_d_required
        self.ss_a_stability = ss_a_stability
