import time

import control as ctrl
import numpy as np
from control.exception import slycot_check

from sippy_unipi.tf2ss import tf2ss


def benchmark_tf2ss(num, den):
    """Compare execution time and numerical accuracy."""
    if slycot_check():
        start_time = time.time()
        A_min, B_min, C_min, D_min = tf2ss(num, den)
        time_minreal = time.time() - start_time

        tf_mimo = [
            [
                ctrl.TransferFunction(num[i][j], den[i][j])
                for j in range(len(num[0]))
            ]
            for i in range(len(num))
        ]
        start_time = time.time()
        ss_slycot = ctrl.tf2ss(
            ctrl.append(*[ctrl.append(*row) for row in tf_mimo])
        )
        time_slycot = time.time() - start_time

        return {
            "time_minreal": time_minreal,
            "time_slycot": time_slycot,
            "accuracy_A": np.allclose(A_min, ss_slycot.A, atol=1e-6),
            "accuracy_B": np.allclose(B_min, ss_slycot.B, atol=1e-6),
            "accuracy_C": np.allclose(C_min, ss_slycot.C, atol=1e-6),
            "accuracy_D": np.allclose(D_min, ss_slycot.D, atol=1e-6),
        }
    else:
        return {
            "time_minreal": None,
            "time_slycot": None,
            "accuracy_A": None,
            "accuracy_B": None,
            "accuracy_C": None,
            "accuracy_D": None,
        }


# Example test case
num_test = [[[1, 1]]]
den_test = [[[1, 3, 2]]]
result = benchmark_tf2ss(num_test, den_test)
print("Benchmark Results:", result)
