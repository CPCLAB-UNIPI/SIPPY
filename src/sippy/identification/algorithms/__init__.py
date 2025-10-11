"""
Concrete algorithm implementations.
"""

try:
    # Register all algorithms with the factory
    from ..factory import AlgorithmFactory
    from .arma import ARMAAlgorithm
    from .ararx import ARARXAlgorithm
    from .ararmax import ARARMAXAlgorithm
    from .armax import ARMAXAlgorithm
    from .arx import ARXAlgorithm
    from .bj import BJAlgorithm
    from .cva import CVAAlgorithm
    from .fir import FIRAlgorithm
    from .moesp import MOESPAlgorithm
    from .n4sid import N4SIDAlgorithm
    from .oe import OEAlgorithm
    from .parsim_k import PARSIMKAlgorithm
    from .parsim_p import PARSIMPAlgorithm
    from .parsim_s import PARSIMSAlgorithm

    AlgorithmFactory.register("N4SID", N4SIDAlgorithm)
    AlgorithmFactory.register("MOESP", MOESPAlgorithm)
    AlgorithmFactory.register("CVA", CVAAlgorithm)
    AlgorithmFactory.register("PARSIM-K", PARSIMKAlgorithm)
    AlgorithmFactory.register("PARSIM-S", PARSIMSAlgorithm)
    AlgorithmFactory.register("PARSIM-P", PARSIMPAlgorithm)
    AlgorithmFactory.register("ARX", ARXAlgorithm)
    AlgorithmFactory.register("ARARX", ARARXAlgorithm)
    AlgorithmFactory.register("ARARMAX", ARARMAXAlgorithm)
    AlgorithmFactory.register("FIR", FIRAlgorithm)
    AlgorithmFactory.register("ARMAX", ARMAXAlgorithm)
    AlgorithmFactory.register("OE", OEAlgorithm)
    AlgorithmFactory.register("ARMA", ARMAAlgorithm)
    AlgorithmFactory.register("BJ", BJAlgorithm)
except ImportError:
    # In case sysidbox is not available, we still have the base classes ready
    N4SIDAlgorithm = None
    MOESPAlgorithm = None
    CVAAlgorithm = None
    PARSIMKAlgorithm = None
    PARSIMSAlgorithm = None
    PARSIMPAlgorithm = None
    ARXAlgorithm = None
    ARARXAlgorithm = None
    ARARMAXAlgorithm = None
    ARMAXAlgorithm = None
    BJAlgorithm = None
    FIRAlgorithm = None
    OEAlgorithm = None
    ARMAAlgorithm = None
