"""
System Identification Module

Refactored system identification with class-based architecture and factory pattern.
"""
from .__main__ import SystemIdentification, system_identification
from .base import SystemIdentificationConfig

__all__ = ['SystemIdentification', 'system_identification', 'SystemIdentificationConfig']
