"""
Factory pattern for system identification algorithms.
"""

from typing import Dict, Type

from .base import IdentificationAlgorithm


class AlgorithmFactory:
    """Factory for creating system identification algorithms."""

    _algorithms: Dict[str, Type[IdentificationAlgorithm]] = {}
    _initialized = False

    @classmethod
    def register(
        cls, name: str, algorithm_class: Type[IdentificationAlgorithm]
    ) -> None:
        """Register an algorithm class with the factory."""
        cls._algorithms[name.upper()] = algorithm_class

    @classmethod
    def _ensure_initialized(cls):
        """Ensure the factory is initialized with registered algorithms."""
        if not cls._initialized:
            try:
                # Import algorithms module to trigger registration
                import importlib

                importlib.import_module(
                    ".algorithms", package=__name__.rsplit(".", 1)[0]
                )
                cls._initialized = True
            except ImportError:
                cls._initialized = True  # Mark as initialized anyway

    @classmethod
    def create(cls, name: str, **kwargs) -> IdentificationAlgorithm:
        """Create an instance of the specified algorithm."""
        cls._ensure_initialized()
        name_upper = name.upper()
        if name_upper not in cls._algorithms:
            raise ValueError(
                f"Unknown algorithm: {name}. Available: {list(cls._algorithms.keys())}"
            )

        return cls._algorithms[name_upper](**kwargs)

    @classmethod
    def list_algorithms(cls) -> list:
        """List all registered algorithms."""
        cls._ensure_initialized()
        return list(cls._algorithms.keys())

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if an algorithm is registered."""
        cls._ensure_initialized()
        return name.upper() in cls._algorithms


def create_algorithm(method: str, **kwargs) -> IdentificationAlgorithm:
    """Convenience function to create an algorithm instance."""
    return AlgorithmFactory.create(method, **kwargs)
