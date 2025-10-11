"""
Filter factory for creating and managing filter instances.
"""

from typing import Any, Dict, Optional, Type
from .base import IFilter, FilterConfig
from .high_pass import HighPassFilter
from .difference import DifferenceFilter
from .zero_mean import ZeroMeanFilter
from .none_filter import NoneFilter


class FilterFactory:
    """
    Factory class for creating filter instances.
    
    Supports plugin-style registration of new filter types and provides
    a clean interface for filter creation with configuration validation.
    """
    
    # Registry of available filters
    _filters: Dict[str, Type[IFilter]] = {}
    
    @classmethod
    def register(cls, name: str, filter_class: Type[IFilter]) -> None:
        """
        Register a new filter type in the factory.
        
        Parameters:
        -----------
        name : str
            Filter identifier (case-insensitive)
        filter_class : Type[IFilter]
            Filter class that implements IFilter interface
            
        Raises:
        ------
        TypeError
            If filter_class doesn't implement IFilter
        ValueError
            If name is already registered
        """
        if not issubclass(filter_class, IFilter):
            raise TypeError("Filter class must implement IFilter interface")
            
        normalized_name = name.lower()
        if normalized_name in cls._filters:
            raise ValueError(f"Filter '{name}' is already registered")
            
        cls._filters[normalized_name] = filter_class
    
    @classmethod
    def unregister(cls, name: str) -> None:
        """
        Unregister a filter type.
        
        Parameters:
        -----------
        name : str
            Filter identifier to remove
        """
        normalized_name = name.lower()
        cls._filters.pop(normalized_name, None)
    
    @classmethod
    def list_filters(cls) -> list[str]:
        """
        Get list of available filter names.
        
        Returns:
        --------
        list[str]
            Available filter identifiers
        """
        return list(cls._filters.keys())
    
    @classmethod
    def is_available(cls, name: str) -> bool:
        """
        Check if a filter type is available.
        
        Parameters:
        -----------
        name : str
            Filter identifier
            
        Returns:
        --------
        bool
            True if filter is available, False otherwise
        """
        return name.lower() in cls._filters
    
    @classmethod
    def create(cls, name: str, config: Optional[FilterConfig] = None, **kwargs) -> IFilter:
        """
        Create a filter instance.
        
        Parameters:
        -----------
        name : str
            Filter identifier (case-insensitive)
        config : FilterConfig, optional
            Filter configuration
        **kwargs
            Additional configuration parameters (merged with config)
            
        Returns:
        --------
        IFilter
            Filter instance
            
        Raises:
        ------
        ValueError
            If filter type is not available or creation fails
        """
        normalized_name = name.lower()
        
        if normalized_name not in cls._filters:
            available = ', '.join(cls._filters.keys())
            raise ValueError(f"Unknown filter '{name}'. Available: {available}")
        
        try:
            filter_class = cls._filters[normalized_name]
            
            # Merge config with kwargs
            if config:
                # Update config with kwargs
                for key, value in kwargs.items():
                    setattr(config, key, value)
                return filter_class(config)
            else:
                # Create config from kwargs
                return (filter_class(FilterConfig(**kwargs)) if kwargs else filter_class())
                
        except Exception as e:
            raise ValueError(f"Failed to create filter '{name}': {e}")
    
    @classmethod
    def get_filter_info(cls, name: str) -> Dict[str, Any]:
        """
        Get information about a filter type.
        
        Parameters:
        -----------
        name : str
            Filter identifier
            
        Returns:
        --------
        dict
            Filter information including name, class, docstring
            
        Raises:
        ------
        ValueError
            If filter type is not available
        """
        normalized_name = name.lower()
        
        if normalized_name not in cls._filters:
            available = ', '.join(cls._filters.keys())
            raise ValueError(f"Unknown filter '{name}'. Available: {available}")
            
        filter_class = cls._filters[normalized_name]
        return {
            'name': name,
            'class': filter_class.__name__,
            'module': filter_class.__module__,
            'doc': filter_class.__doc__ or "No documentation available"
        }


# Register built-in filters
FilterFactory.register('highpass', HighPassFilter)
FilterFactory.register('high_pass', HighPassFilter)
FilterFactory.register('difference', DifferenceFilter)
FilterFactory.register('doubledifference', DifferenceFilter)
FilterFactory.register('diff', DifferenceFilter)
FilterFactory.register('zeromean', ZeroMeanFilter)
FilterFactory.register('zero_mean', ZeroMeanFilter)
FilterFactory.register('none', NoneFilter)
FilterFactory.register('passthrough', NoneFilter)


def get_filter(filter_type: str, **kwargs) -> IFilter:
    """
    Convenience function for creating filters (backward compatibility).
    
    Parameters:
    -----------
    filter_type : str
        Filter type identifier
    **kwargs
        Filter configuration parameters
        
    Returns:
    --------
    IFilter
        Filter instance
        
    Note:
    ----
    This function provides compatibility with the old DetrendingFilter.get_filter() interface.
    It will be deprecated in favor of FilterFactory.create().
    """
    import warnings
    
    warnings.warn(
        "get_filter() is deprecated, use FilterFactory.create() instead",
        DeprecationWarning,
        stacklevel=2
    )
    
    return FilterFactory.create(filter_type, **kwargs)
