"""
Zero-mean filter implementation for signal centering.
"""

from typing import Any, Dict, Optional

import pandas as pd

from .base import FilterConfig, IFilter


class ZeroMeanFilter(IFilter):
    """
    Zero-mean filter for removing DC offset from time series data.

    Subtracts the mean value from each signal column to center the data
    around zero. This is commonly used for signal preprocessing before
    further analysis or modeling.
    """

    def __init__(self, config: Optional[FilterConfig] = None):
        """
        Initialize zero-mean filter.

        Parameters:
        -----------
        config : FilterConfig, optional
            Filter configuration parameters
        """
        super().__init__(config)

    def apply_filter(self,
                    data: pd.DataFrame,
                    tss: Optional[float] = None,
                    multiplier: Optional[float] = None,
                    slices: Optional[Dict[str, Any]] = None,
                    **kwargs) -> pd.DataFrame:
        """
        Apply zero-mean filter to input data.

        Parameters:
        -----------
        data : pd.DataFrame
            Input time series data to center around zero
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
            Zero-mean centered data

        Raises:
        ------
        ValueError
            If data validation fails
        """
        # Validate input
        self._validate_input(data)

        # Process slices for bad data
        processed_data = self._process_slices(data, slices or self.config.slices)

        # Apply zero-mean filtering
        try:
            column_means = processed_data.mean()
            zero_mean_data = processed_data - column_means
        except Exception as e:
            raise ValueError(f"Failed to apply zero-mean filter: {e}")

        # Store results for backward compatibility
        self.data_manager.add_data("input", data, type="original")
        self.data_manager.add_data("trend", data.copy(), type="original_data")
        self.data_manager.add_data("output", zero_mean_data, type="zero_mean")
        self.data_manager.add_data("means", column_means, type="computed_means")

        return zero_mean_data

    def get_column_means(self) -> pd.Series:
        """
        Get the computed column means from the last filtering operation.

        Returns:
        --------
        pd.Series
            Mean values for each column (empty if no filtering performed yet)
        """
        means = self.data_manager.get_data("means")
        if means is not None:
            return means
        return pd.Series()

    def get_filter_info(self) -> dict:
        """
        Get information about this filter instance.

        Returns:
        --------
        dict
            Filter information
        """
        return {
            'type': 'ZeroMeanFilter',
            'description': 'Zero-mean filter that removes DC offset from signals',
            'suitable_for': 'Centering data around zero before analysis',
            'effect': 'Subtracts mean value from each column'
        }
