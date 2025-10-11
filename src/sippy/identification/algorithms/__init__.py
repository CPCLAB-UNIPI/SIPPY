"""
Concrete algorithm implementations.
"""
try:
    # Register all algorithms with the factory
    from ..factory import AlgorithmFactory
    from .arx import ARXAlgorithm
    from .cva import CVAAlgorithm
    from .fir import FIRAlgorithm
    from .moesp import MOESPAlgorithm
    from .n4sid import N4SIDAlgorithm
    from .parsim_k import PARSIMKAlgorithm
    from .parsim_p import PARSIMPAlgorithm
    from .parsim_s import PARSIMSAlgorithm

    AlgorithmFactory.register('N4SID', N4SIDAlgorithm)
    AlgorithmFactory.register('MOESP', MOESPAlgorithm)
    AlgorithmFactory.register('CVA', CVAAlgorithm)
    AlgorithmFactory.register('PARSIM-K', PARSIMKAlgorithm)
    AlgorithmFactory.register('PARSIM-S', PARSIMSAlgorithm)
    AlgorithmFactory.register('PARSIM-P', PARSIMPAlgorithm)
    AlgorithmFactory.register('ARX', ARXAlgorithm)
    AlgorithmFactory.register('FIR', FIRAlgorithm)
except ImportError:
    # In case sysidbox is not available, we still have the base classes ready
    N4SIDAlgorithm = None
    MOESPAlgorithm = None
    CVAAlgorithm = None
    PARSIMKAlgorithm = None
    PARSIMSAlgorithm = None
    PARSIMPAlgorithm = None
    ARXAlgorithm = None
    FIRAlgorithm = None
