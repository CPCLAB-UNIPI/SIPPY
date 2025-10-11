"""
Utility functions for filter operations and analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, Union


def calculate_sampling_frequency(data: pd.DataFrame) -> float:
    """
    Calculate the sampling frequency from time series data.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Time series data with time-based index
        
    Returns:
    --------
    float
        Sampling frequency in Hz
        
    Raises:
    ------
    ValueError
        If sampling frequency cannot be determined
    """
    if len(data) < 2:
        raise ValueError("Need at least 2 data points to calculate sampling frequency")
    
    try:
        # Assuming the index is time-based
        time_diff = pd.Timedelta(data.index[1] - data.index[0]).total_seconds()
        if time_diff <= 0:
            raise ValueError(f"Invalid time difference: {time_diff} seconds")
        
        return 1.0 / time_diff
    except Exception as e:
        raise ValueError(f"Failed to calculate sampling frequency: {e}")


def detect_outliers(data: pd.DataFrame, 
                    method: str = 'iqr',
                    threshold: float = 1.5) -> pd.DataFrame:
    """
    Detect outliers in time series data using various methods.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Time series data
    method : str, default 'iqr'
        Outlier detection method ('iqr', 'zscore', 'modified_zscore')
    threshold : float, default 1.5
        Threshold for outlier detection
        
    Returns:
    --------
    pd.DataFrame
        Boolean DataFrame indicating outlier locations
        
    Raises:
    ------
    ValueError
        If method is not supported
    """
    outliers = pd.DataFrame(index=data.index, columns=data.columns, dtype=bool)
    
    for column in data.columns:
        if method == 'iqr':
            outliers[column] = _detect_outliers_iqr(data[column], threshold)
        elif method == 'zscore':
            outliers[column] = _detect_outliers_zscore(data[column], threshold)
        elif method == 'modified_zscore':
            outliers[column] = _detect_outliers_modified_zscore(data[column], threshold)
        else:
            raise ValueError(f"Method '{method}' not supported. Use 'iqr', 'zscore', or 'modified_zscore'")
    
    return outliers


def _detect_outliers_iqr(series: pd.Series, threshold: float) -> pd.Series:
    """Detect outliers using the IQR method."""
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    return (series < lower_bound) | (series > upper_bound)


def _detect_outliers_zscore(series: pd.Series, threshold: float) -> pd.Series:
    """Detect outliers using the Z-score method."""
    z_scores = np.abs((series - series.mean()) / series.std())
    return z_scores > threshold


def _detect_outliers_modified_zscore(series: pd.Series, threshold: float) -> pd.Series:
    """Detect outliers using the Modified Z-score method."""
    median = series.median()
    mad = np.median(np.abs(series - median))
    modified_z_scores = 0.6745 * (series - median) / mad
    return np.abs(modified_z_scores) > threshold


def analyze_signal_properties(data: pd.DataFrame) -> Dict[str, Union[float, Dict[str, float]]]:
    """
    Analyze basic properties of time series signals.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Time series data to analyze
        
    Returns:
    --------
    dict
        Dictionary containing signal properties for each column
    """
    properties = {}
    
    for column in data.columns:
        series = data[column].dropna()
        
        if len(series) == 0:
            continue
            
        column_props = {
            'mean': float(series.mean()),
            'std': float(series.std()),
            'min': float(series.min()),
            'max': float(series.max()),
            'range': float(series.max() - series.min()),
            'signal_to_noise': float(series.std() / np.abs(series.mean())) if series.mean() != 0 else np.inf,
            'zero_crossings': count_zero_crossings(series),
            'autocorr_lag1': series.autocorr(lag=1) if len(series) > 1 else np.nan
        }
        
        properties[column] = column_props
    
    return properties


def count_zero_crossings(series: pd.Series, threshold: float = 0.0) -> int:
    """
    Count the number of zero crossings in a time series.
    
    Parameters:
    -----------
    series : pd.Series
        Time series data
    threshold : float, default 0.0
        Threshold level for zero crossing detection
        
    Returns:
    --------
    int
        Number of zero crossings
    """
    if len(series) < 2:
        return 0
    
    # Detect crosses where the signal crosses the threshold
    crosses = (series > threshold).astype(int).diff().abs()
    return int(crosses.sum())


def create_test_data(length: int = 1000,
                   freq: float = 1.0,
                   noise_level: float = 0.1,
                   trend_slope: float = 0.0) -> pd.DataFrame:
    """
    Create synthetic test data for filter testing.
    
    Parameters:
    -----------
    length : int, default 1000
        Length of time series
    freq : float, default 1.0
        Frequency of sine wave component
    noise_level : float, default 0.1
        Level of white noise
    trend_slope : float, default 0.0
        Linear trend slope
        
    Returns:
    --------
    pd.DataFrame
        Test data with 'signal' column and time index
    """
    np.random.seed(42)
    
    # Generate time index
    t = np.arange(length) / freq
    
    # Generate signal components
    signal = np.sin(2 * np.pi * 0.1 * t)  # Low frequency
    signal += 0.5 * np.sin(2 * np.pi * 0.3 * t)  # Mid frequency
    signal += 0.2 * np.sin(2 * np.pi * 0.7 * t)  # High frequency
    
    # Add trend
    signal += trend_slope * t
    
    # Add noise
    if noise_level > 0:
        signal += noise_level * np.random.randn(length)
    
    # Create DataFrame with time index
    index = pd.date_range(start='2023-01-01', periods=length, freq='1S')
    return pd.DataFrame({'signal': signal}, index=index)
