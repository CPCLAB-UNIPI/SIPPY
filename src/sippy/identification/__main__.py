"""
Main system identification interface.
"""
import numpy as np
from typing import Union, Optional
from .base import StateSpaceModel, SystemIdentificationConfig
from .factory import create_algorithm
try:
    from sysidbox import functionset as fs
except ImportError:
    fs = None


class SystemIdentification:
    """Main class for system identification using factory pattern."""
    
    def __init__(self, config: Optional[SystemIdentificationConfig] = None):
        """
        Initialize system identification.
        
        Args:
            config: Configuration object. If None, default config is used.
        """
        self.config = config or SystemIdentificationConfig()
    
    def identify(self, y: np.ndarray, u: np.ndarray, **kwargs) -> StateSpaceModel:
        """
        Perform system identification.
        
        Args:
            y: Output data (outputs x time_steps)
            u: Input data (inputs x time_steps)
            **kwargs: Override config parameters
            
        Returns:
            StateSpaceModel: Identified model
        """
        # Merge config with kwargs
        config_dict = self.config.__dict__.copy()
        config_dict.update(kwargs)
        
        method = config_dict.get('method', 'N4SID')
        
        # Create algorithm instance
        algorithm = create_algorithm(method)
        
        # Apply data centering if specified
        y_centered, u_centered = self._apply_centering(y, u, config_dict.get('centering', 'None'))
        
        # Perform identification
        model = algorithm.identify(y_centered, u_centered, **config_dict)
        
        return model
    
    def _apply_centering(self, y: np.ndarray, u: np.ndarray, centering: str) -> tuple:
        """Apply data centering preprocessing."""
        y = 1.0 * np.atleast_2d(y)
        u = 1.0 * np.atleast_2d(u)
        
        [n1, n2] = y.shape
        ydim = min(n1, n2)
        ylength = max(n1, n2)
        if ylength == n1:
            y = y.T
        [n1, n2] = u.shape
        ulength = max(n1, n2)
        udim = min(n1, n2)
        if ulength == n1:
            u = u.T
            
        # Checking data consistency
        if ulength != ylength:
            print("Warning: y and u lengths are not the same. Using minimum length.")
            minlength = min(ulength, ylength)
            y = y[:, :minlength]
            u = u[:, :minlength]
        
        if centering == 'InitVal':
            y_rif = 1.0 * y[:, 0]
            u_init = 1.0 * u[:, 0]
            for i in range(ylength):
                y[:, i] = y[:, i] - y_rif
                u[:, i] = u[:, i] - u_init
        elif centering == 'MeanVal':
            y_rif = np.zeros(ydim)
            u_mean = np.zeros(udim)
            for i in range(ydim):
                y_rif[i] = np.mean(y[i, :])
            for i in range(udim):
                u_mean[i] = np.mean(u[i, :])
            for i in range(ylength):
                y[:, i] = y[:, i] - y_rif
                u[:, i] = u[:, i] - u_mean
        
        return y, u


# Convenience function for backward compatibility
def system_identification(y: np.ndarray, u: np.ndarray, id_method: str, **kwargs) -> StateSpaceModel:
    """
    Backward compatibility function that mimics the original API.
    
    This function provides the same interface as the original system_identification
    function but uses the new class-based architecture internally.
    """
    # Map old parameter names to new ones
    param_mapping = {
        'SS_fixed_order': 'ss_fixed_order',
        'SS_max_order': 'ss_max_order', 
        'SS_orders': 'ss_orders',
        'SS_threshold': 'ss_threshold',
        'SS_f': 'ss_f',
        'SS_D_required': 'ss_d_required',
        'SS_A_stability': 'ss_a_stability',
        'IC': 'ic'
    }
    
    # Convert parameter names
    mapped_kwargs = {}
    for key, value in kwargs.items():
        mapped_key = param_mapping.get(key, key)
        mapped_kwargs[mapped_key] = value
    
    config = SystemIdentificationConfig(method=id_method, **mapped_kwargs)
    identifier = SystemIdentification(config)
    return identifier.identify(y, u)
