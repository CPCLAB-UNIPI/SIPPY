"""
Difference filter implementation for signal differentiation.
"""

from typing import Optional, Any, Dict
import pandas as pd
from .base import FilterConfig, IFilter


class DifferenceFilter(IFilter):
    """
    Difference filter for computing discrete derivatives of time series.
    
    Supports first-order (simple difference) and second-order (double difference)
    filtering. Useful for removing trends and making time series stationary.
    """
    
    def __init__(self, config: Optional[FilterConfig] = None):
        """
        Initialize difference filter.
        
        Parameters:
        -----------
        config : FilterConfig, optional
            Filter configuration parameters
        """
        super().__init__(config)
        
        # Default to first-order difference
        self._order = 1
    
    def set_order(self, order: int) -> None:
        """
        Set the difference order.
        
        Parameters:
        -----------
        order : int
            Difference order (1 for first-order, 2 for second-order)
            
        Raises:
        ------
        ValueError
            If order is not 1 or 2
        """
        if order not in [1, 2]:
            raise ValueError(f"Difference order must be 1 or 2, got {order}")
        self._order = order
    
    def apply_filter(self, 
                    data: pd.DataFrame, 
                    tss: Optional[float] = None,
                    multiplier: Optional[float] = None,
                    slices: Optional[Dict[str, Any]] = None,
                    **kwargs) -> pd.DataFrame:
        """
        Apply difference filter to input data.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input time series data to differentiate
        tss : float, optional
            Time to steady state in seconds (not used in this filter)
        multiplier : float, optional
            Multiplier parameter (not used in this filter)
        slices : dict, optional
            Data slice definitions for bad data handling (overrides config)
        **kwargs
            Additional parameters
            
        Returns:
        --------
        pd.DataFrame
            Differenced data
            
        Raises:
        ------
        ValueError
            If data validation fails or difference order is invalid
        """
        # Validate input
        self._validate_input(data)
        
        # Process slices for bad data
        processed_data = self._process_slices(data, slices or self.config.slices)
        
        # Apply difference based on order
        try:
            if self._order == 1:
                differentiated = processed_data.diff().fillna(method='backfill')
            elif self._order == 2:
                differentiated = processed_data.diff().diff().fillna(method='backfill')
            else:
                raise ValueError(f"Difference order must be 1 or 2, got {self._order}")
        except Exception as e:
            raise ValueError(f"Failed to apply {self._order}-order difference: {e}")
        
        # Store results for backward compatibility
        self.data_manager.add_data("input", data, type="original")
        self.data_manager.add_data("trend", data.copy(), type="original_data") 
        self.data_manager.add_data("output", differentiated, type=f"order_{self._order}_difference")
        
        return differentiated
    
    def get_filter_info(self) -> dict:
        """
        Get information about this filter instance.
        
        Returns:
        --------
        dict
            Filter parameters and information
        """
        return {
            'type': 'DifferenceFilter',
            'order': self._order,
            'description': f'{self._order}-order discrete difference filter',
            'suitable_for': 'Removing trends and making time series stationary'
        }
