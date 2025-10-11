"""
Concrete algorithm implementations.
"""
try:
    # Register all algorithms with the factory
    from ..factory import AlgorithmFactory
    from .cva import CVAAlgorithm
    from .moesp import MOESPAlgorithm
    from .n4sid import N4SIDAlgorithm

    AlgorithmFactory.register('N4SID', N4SIDAlgorithm)
    AlgorithmFactory.register('MOESP', MOESPAlgorithm)
    AlgorithmFactory.register('CVA', CVAAlgorithm)
except ImportError:
    # In case sysidbox is not available, we still have the base classes ready
    N4SIDAlgorithm = None
    MOESPAlgorithm = None
    CVAAlgorithm = None
