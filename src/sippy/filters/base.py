"""
Base classes and interfaces for signal processing filters.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


class FilterConfig:
    """
    Configuration container for filter parameters.
    
    Provides type-safe configuration with validation and sensible defaults.
    """
    
    def __init__(self, 
                 cutoff: Optional[float] = None,
                 order: Optional[int] = None,
                 tss: Optional[float] = None,
                 multiplier: float = 3.0,
                 slices: Optional[Dict[str, Any]] = None):
        """
        Initialize filter configuration.
        
        Parameters:
        -----------
        cutoff : float, optional
            Filter cutoff frequency in Hz
        order : int, optional
            Filter order (for FIR filters)
        tss : float, optional
            Time to steady state in seconds
        multiplier : float, default 3.0
            Multiplication factor for filter time to steady state
        slices : dict, optional
            Data slice definitions for processing
        """
        self.cutoff = cutoff
        self.order = order
        self.tss = tss
        self.multiplier = multiplier
        self.slices = slices or {}
        
        # Validate configuration
        if multiplier <= 0:
            raise ValueError("Multiplier must be positive")
        if tss is not None and tss <= 0:
            raise ValueError("Time to steady state must be positive")


class FilterDataManager:
    """
    Data manager for filter operations.
    
    Replaces the singleton pattern with a proper data management approach.
    Stores input data, processed data, and intermediate results.
    """
    
    def __init__(self):
        """Initialize data manager with empty storage."""
        self._data: Dict[str, Union[pd.DataFrame, np.ndarray]] = {}
        self._metadata: Dict[str, Any] = {}
    
    def add_data(self, key: str, data: Union[pd.DataFrame, np.ndarray], **metadata: Any) -> None:
        """
        Store data with optional metadata.
        
        Parameters:
        -----------
        key : str
            Storage key for the data
        data : pd.DataFrame or np.ndarray
            Data to store
        **metadata
            Additional metadata about the data
        """
        self._data[key] = data.copy() if isinstance(data, pd.DataFrame) else data
        self._metadata[key] = metadata
    
    def get_data(self, key: str) -> Optional[Union[pd.DataFrame, np.ndarray]]:
        """
        Retrieve stored data.
        
        Parameters:
        -----------
        key : str
            Storage key
            
        Returns:
        --------
        pd.DataFrame or np.ndarray or None
            Stored data if key exists, None otherwise
        """
        return self._data.get(key)
    
    def get_metadata(self, key: str) -> Dict[str, Any]:
        """
        Retrieve metadata for stored data.
        
        Parameters:
        -----------
        key : str
            Storage key
            
        Returns:
        --------
        dict
            Metadata dictionary (empty if key doesn't exist)
        """
        return self._metadata.get(key, {})
    
    def has_data(self, key: str) -> bool:
        """
        Check if data exists for the given key.
        
        Parameters:
        -----------
        key : str
            Storage key
            
        Returns:
        --------
        bool
            True if data exists, False otherwise
        """
        return key in self._data
    
    def clear(self) -> None:
        """Clear all stored data and metadata."""
        self._data.clear()
        self._metadata.clear()
        
    @property
    def data(self) -> Dict[str, Union[pd.DataFrame, np.ndarray]]:
        """Get all stored data (read-only property for backward compatibility)."""
        return self._data.copy()


class IFilter(ABC):
    """
    Abstract base class for signal processing filters.
    
    Defines the interface that all filter implementations must follow.
    """
    
    def __init__(self, config: Optional[FilterConfig] = None):
        """
        Initialize filter with optional configuration.
        
        Parameters:
        -----------
        config : FilterConfig, optional
            Filter configuration parameters
        """
        self.config = config or FilterConfig()
        self.data_manager = FilterDataManager()
    
    @abstractmethod
    def apply_filter(self, 
                    data: pd.DataFrame, 
                    tss: Optional[float] = None,
                    multiplier: Optional[float] = None,
                    slices: Optional[Dict[str, Any]] = None,
                    **kwargs) -> pd.DataFrame:
        """
        Apply the filter to input data.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input data to filter
        tss : float, optional
            Time to steady state in seconds (overrides config)
        multiplier : float, optional
            Filter timestep multiplier (overrides config)
        slices : dict, optional
            Data slice definitions (overrides config)
        **kwargs
            Additional filter-specific parameters
            
        Returns:
        --------
        pd.DataFrame
            Filtered data
            
        Raises:
        ------
        ValueError
            If input validation fails
        TypeError
            If input data type is unsupported
        """
        pass
    
    def _validate_input(self, data: pd.DataFrame) -> None:
        """
        Validate input data.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input data to validate
            
        Raises:
        ------
        TypeError
            If input is not a pandas DataFrame
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"Expected pandas.DataFrame, got {type(data)}")
        
        if data.empty:
            raise ValueError("Input DataFrame cannot be empty")
    
    def _process_slices(self, 
                       data: pd.DataFrame, 
                       slices: Dict[str, Any]) -> pd.DataFrame:
        """
        Process data slices for bad data interpolation.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input data
        slices : dict
            Slice definitions
            
        Returns:
        --------
        pd.DataFrame
            Processed data with slices applied
        """
        processed_data = data.copy(deep=True)
        
        for slice_info in slices.values():
            if 'tags' not in slice_info:
                continue
                
            # Check if any tags exist in the data
            valid_tags = [tag for tag in slice_info['tags'] if tag in processed_data.columns]
            if not valid_tags:
                continue
                
            start, end = slice_info['start'], slice_info['end']
            
            if slice_info['type'] == 'bad':
                if slice_info.get('isGlobal', False):
                    # Apply to all tags if global
                    for col in processed_data.columns:
                        processed_data.iloc[start:end, processed_data.columns.get_loc(col)] = np.nan
                else:
                    # Apply only to specified tags
                    for tag in valid_tags:
                        processed_data.iloc[start:end, processed_data.columns.get_loc(tag)] = np.nan
                # Forward fill
                processed_data = processed_data.ffill()
                
            elif slice_info['type'] == 'interpolate':
                for tag in valid_tags:
                    processed_data.iloc[start:end, processed_data.columns.get_loc(tag)] = np.nan
                    processed_data[tag] = processed_data[tag].interpolate(method='linear')
                    
        return processed_data
    
    def _calculate_sampling_time(self, data: pd.DataFrame) -> float:
        """
        Calculate sampling time from data index.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input data with time-based index
            
        Returns:
        --------
        float
            Sampling time in seconds
            
        Raises:
        ------
        ValueError
            If cannot determine sampling time
        """
        try:
            if len(data) < 2:
                raise ValueError("Cannot determine sampling time from single data point")
                
            ts = pd.Timedelta(data.index[1] - data.index[0]).total_seconds()
            if ts <= 0:
                raise ValueError(f"Invalid sampling time calculated: {ts}")
                
            return ts
        except Exception as e:
            raise ValueError(f"Failed to calculate sampling time: {e}")
    
    def get_name(self) -> str:
        """
        Get the filter name.
        
        Returns:
        --------
        str
            Human-readable filter name
        """
        return self.__class__.__name__
