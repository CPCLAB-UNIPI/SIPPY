"""
Signal processing filters for SIPPY.

This module provides various signal preprocessing filters including
high-pass, difference, zero-mean, and pass-through filters.
"""

from .base import FilterConfig, FilterDataManager, IFilter
from .difference import DifferenceFilter
from .factory import FilterFactory
from .high_pass import HighPassFilter
from .none_filter import NoneFilter
from .zero_mean import ZeroMeanFilter

__all__ = [
    "FilterFactory",
    "IFilter",
    "FilterConfig",
    "FilterDataManager",
    "HighPassFilter",
    "DifferenceFilter",
    "ZeroMeanFilter",
    "NoneFilter",
]
