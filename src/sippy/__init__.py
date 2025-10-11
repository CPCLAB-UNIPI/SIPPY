"""
SIPPY - Systems Identification Package for Python

New modular architecture with object-oriented identification algorithms.
"""

from .identification import (
    GBN_seq,
    SystemIdentification,
    SystemIdentificationConfig,
    get_fir_coef,
    get_model_uncertainty,
    get_step_response,
    simulate_ss_system,
    system_identification,
    white_noise_var,
)

__all__ = [
    "SystemIdentification",
    "system_identification",
    "SystemIdentificationConfig",
    "GBN_seq",
    "white_noise_var",
    "get_fir_coef",
    "get_step_response",
    "get_model_uncertainty",
    "simulate_ss_system",
]


def hello() -> str:
    return "Hello from sippy!"
