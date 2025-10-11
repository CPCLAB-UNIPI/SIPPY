"""
System Identification Module

Refactored system identification with class-based architecture and factory pattern.
"""
from ..utils import (
    GBN_seq,
    get_fir_coef,
    get_model_uncertainty,
    get_step_response,
    simulate_ss_system,
    white_noise_var,
)
from .__main__ import SystemIdentification, system_identification
from .base import SystemIdentificationConfig
from .iddata import IDData

__all__ = [
    'SystemIdentification', 'system_identification', 'SystemIdentificationConfig', 'IDData',
    'GBN_seq', 'white_noise_var', 'get_fir_coef', 'get_step_response',
    'get_model_uncertainty', 'simulate_ss_system'
]
