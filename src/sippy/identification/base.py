"""
Base classes for system identification algorithms.
"""
from abc import ABC, abstractmethod
from typing import Optional, Union, List, Tuple
import numpy as np


class IdentificationAlgorithm(ABC):
    """Abstract base class for system identification algorithms."""
    
    def __init__(self):
        self.name = self.__class__.__name__
    
    @abstractmethod
    def identify(self, y: np.ndarray, u: np.ndarray, **kwargs) -> 'StateSpaceModel':
        """
        Perform system identification.
        
        Args:
            y: Output data (outputs x time_steps)
            u: Input data (inputs x time_steps)
            **kwargs: Algorithm-specific parameters
            
        Returns:
            StateSpaceModel: Identified model
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
        except:
            self.A_K = np.array([])
            self.B_K = np.array([])
    
    def is_stable(self) -> bool:
        """Check if the system matrix A is stable."""
        try:
            eigenvals = np.linalg.eigvals(self.A)
            return np.all(np.abs(eigenvals) < 1.0)
        except:
            return False
    
    def get_natural_frequencies(self) -> np.ndarray:
        """Get natural frequencies of the system."""
        try:
            eigenvals = np.linalg.eigvals(self.A)
            return np.abs(np.angle(eigenvals) / (2 * np.pi * self.ts))
        except:
            return np.array([])
    
    def get_damping_ratios(self) -> np.ndarray:
        """Get damping ratios of the system."""
        try:
            eigenvals = np.linalg.eigvals(self.A)
            return -np.real(eigenvals) / np.abs(eigenvals)
        except:
            return np.array([])


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
