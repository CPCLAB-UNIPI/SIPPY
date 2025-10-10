"""
System Identification Module

Refactored system identification with class-based architecture and factory pattern.
"""
from .__main__ import SystemIdentification, system_identification
from .base import SystemIdentificationConfig
from ..utils import (
    GBN_seq, white_noise_var, get_fir_coef, get_step_response, 
    get_model_uncertainty, simulate_ss_system
)

__all__ = [
    'SystemIdentification', 'system_identification', 'SystemIdentificationConfig',
    'GBN_seq', 'white_noise_var', 'get_fir_coef', 'get_step_response', 
    'get_model_uncertainty', 'simulate_ss_system'
]
