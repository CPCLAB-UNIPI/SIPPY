"""
High-pass filter implementation for signal preprocessing.
"""

from typing import Any, Dict, Optional

import pandas as pd
from scipy.signal import filtfilt, firwin, kaiserord

from .base import FilterConfig, IFilter


class HighPassFilter(IFilter):
    """
    High-pass filter for removing low-frequency trends from time series data.

    Uses FIR filter design with Kaiser window for optimal stopband attenuation.
    Suitable for removing slow drift from process data while preserving
    dynamic variations of interest.
    """

    def __init__(self, config: Optional[FilterConfig] = None):
        """
        Initialize high-pass filter.

        Parameters:
        -----------
        config : FilterConfig, optional
            Filter configuration parameters
        """
        super().__init__(config)

        # Filter-specific defaults
        self._ripple_db = 65  # Stopband attenuation in dB
        self._width_factor = 0.5  # Transition width factor relative to Nyquist

    def apply_filter(self,
                    data: pd.DataFrame,
                    tss: Optional[float] = None,
                    multiplier: Optional[float] = None,
                    slices: Optional[Dict[str, Any]] = None,
                    **kwargs) -> pd.DataFrame:
        """
        Apply high-pass filter to input data.

        Parameters:
        -----------
        data : pd.DataFrame
            Input time series data to filter
        tss : float, optional
            Time to steady state in seconds (overrides config)
        multiplier : float, optional
            Filter timestep multiplier for filter cutoff calculation
            (overrides config)
        slices : dict, optional
            Data slice definitions for bad data handling (overrides config)
        **kwargs
            Additional parameters (ignored for this filter)

        Returns:
        --------
        pd.DataFrame
            High-pass filtered data

        Raises:
        ------
        ValueError
            If data validation or filtering parameters are invalid
        """
        # Validate input
        self._validate_input(data)

        # Process slices for bad data
        processed_data = self._process_slices(data, slices or self.config.slices)

        # Get or sampling time
        if tss is not None:
            tss_seconds = tss * 60  # Convert minutes to seconds
        elif self.config.tss is not None:
            tss_seconds = self.config.tss * 60
        else:
            tss_seconds = 60  # Default 1 minute

        # Get multiplier
        if multiplier is not None:
            mult_factor = multiplier
        else:
            mult_factor = self.config.multiplier

        try:
            # Calculate sampling time from data
            sample_time = self._calculate_sampling_time(processed_data)
        except ValueError as e:
            raise ValueError(f"Cannot process data: {e}")

        # Calculate filter parameters
        filter_tss = tss_seconds * mult_factor
        cutoff = 1 / (2 * filter_tss)
        nyquist_rate = sample_time / 2.0
        width = self._width_factor / nyquist_rate

        # Validate filter parameters
        if cutoff <= 0 or cutoff >= nyquist_rate:
            raise ValueError(f"Invalid cutoff frequency {cutoff} Hz. Must be between 0 and {nyquist_rate:.2f} Hz")

        if width >= cutoff:
            raise ValueError(f"Filter width {width:.4f} Hz too large for cutoff {cutoff:.4f} Hz")

        try:
            # Design FIR filter
            num_taps, beta = kaiserord(self._ripple_db, width)
            window = ("kaiser", beta)
            coef = firwin(
                numtaps=num_taps,
                cutoff=cutoff,
                window=window,
                pass_zero=False,  # High-pass
                fs=nyquist_rate,
            )
        except Exception as e:
            raise ValueError(f"Failed to design high-pass filter: {e}")

        # Apply filter
        try:
            trend = processed_data.copy(deep=True)
            for column in processed_data.columns:
                trend[column] = filtfilt(coef, 1.0, processed_data[column].values)
        except Exception as e:
            raise ValueError(f"Failed to apply filter: {e}")

        # Handle slices for trend restoration
        if slices or self.config.slices:
            all_slices = slices or self.config.slices
            for slice_info in all_slices.values():
                if slice_info['type'] == 'bad':
                    if slice_info.get('isGlobal', False):
                        # Restore original data for bad slices (global)
                        start, end = slice_info['start'], slice_info['end']
                        trend.iloc[start:end, :] = processed_data.iloc[start:end, :]
                    else:
                        # Restore only specified tags
                        if 'tags' in slice_info:
                            start, end = slice_info['start'], slice_info['end']
                            valid_tags = [tag for tag in slice_info['tags'] if tag in trend.columns]
                            for tag in valid_tags:
                                col_idx = trend.columns.get_loc(tag)
                                trend.iloc[start:end, col_idx] = processed_data.iloc[start:end, col_idx]

        # Store results for backward compatibility
        self.data_manager.add_data("input", data, type="original")
        self.data_manager.add_data("trend", trend, type="filtered_trend")
        self.data_manager.add_data("output", processed_data - trend, type="highpass_output")

        return processed_data - trend

    def get_filter_info(self) -> dict:
        """
        Get information about this filter instance.

        Returns:
        --------
        dict
            Filter parameters and design information
        """
        return {
            'type': 'HighPassFilter',
            'ripple_db': self._ripple_db,
            'width_factor': self._width_factor,
            'description': 'High-pass FIR filter using Kaiser window design',
            'suitable_for': 'Removing low-frequency drift from process data'
        }
