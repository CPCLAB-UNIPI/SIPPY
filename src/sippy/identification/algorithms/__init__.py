"""
Concrete algorithm implementations.
"""
try:
    from .n4sid import N4SIDAlgorithm
    from .moesp import MOESPAlgorithm
    from .cva import CVAAlgorithm
    
    # Register all algorithms with the factory
    from ..factory import AlgorithmFactory
    
    AlgorithmFactory.register('N4SID', N4SIDAlgorithm)
    AlgorithmFactory.register('MOESP', MOESPAlgorithm)
    AlgorithmFactory.register('CVA', CVAAlgorithm)
except ImportError:
    # In case sysidbox is not available, we still have the base classes ready
    N4SIDAlgorithm = None
    MOESPAlgorithm = None
    CVAAlgorithm = None
