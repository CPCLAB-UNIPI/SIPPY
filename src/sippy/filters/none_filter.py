"""
None/passthrough filter implementation.
"""

from typing import Any, Dict, Optional

import pandas as pd

from .base import FilterConfig, IFilter


class NoneFilter(IFilter):
    """
    Passthrough filter that performs no transformation on the data.

    Useful for testing purposes or when no filtering is desired
    but a consistent interface is still needed.
    """

    def __init__(self, config: Optional[FilterConfig] = None):
        """
        Initialize none filter.

        Parameters:
        -----------
        config : FilterConfig, optional
            Filter configuration parameters (not used)
        """
        super().__init__(config)

    def apply_filter(self,
                    data: pd.DataFrame,
                    tss: Optional[float] = None,
                    multiplier: Optional[float] = None,
                    slices: Optional[Dict[str, Any]] = None,
                    **kwargs) -> pd.DataFrame:
        """
        Apply passthrough filter (returns input data unchanged except for slice processing).

        Parameters:
        -----------
        data : pd.DataFrame
            Input time series data (returned unchanged)
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
            Original data (possibly with slice processing applied)

        Raises:
        ------
        ValueError
            If data validation fails
        """
        # Validate input
        self._validate_input(data)

        # Process slices for bad data if provided
        processed_data = self._process_slices(data, slices or self.config.slices)

        # Store results for backward compatibility
        self.data_manager.add_data("input", data, type="original")
        self.data_manager.add_data("trend", data.copy(), type="original_data")
        self.data_manager.add_data("output", processed_data, type="passthrough")

        return processed_data

    def get_filter_info(self) -> dict:
        """
        Get information about this filter instance.

        Returns:
        --------
        dict
            Filter information
        """
        return {
            'type': 'NoneFilter',
            'description': 'Passthrough filter that returns input data unchanged',
            'suitable_for': 'Testing or when no filtering is required',
            'effect': 'No transformation applied (except slice processing)'
        }
