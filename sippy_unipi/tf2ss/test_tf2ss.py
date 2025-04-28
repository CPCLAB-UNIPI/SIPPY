import control as ctrl
import numpy as np
import pytest
from control.exception import slycot_check

from sippy_unipi.tf2ss import _get_lcm_norm_coeffs, _pad_numerators, tf2ss

systems = [
    ([[[0.5]]], [[[32]]]),
    ([[[1]]], [[[1, 2]]]),  # H(s) = 1 / (s+2)
    ([[[1, 1]]], [[[1, 3, 2]]]),  # H(s) = (s+1) / (s^2 + 3s + 2)
    (
        [[[1], [0.5]], [[0], [1]]],
        [[[1, 2], [1, 2]], [[0, 1], [1, 2]]],
    ),  # MIMO case - different denominators
    (
        [
            [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
            [[3.0, -1.0, 1.0], [0.0, 1.0, 0.0]],
            [[0.0, 0.0, 1.0], [0.0, 2.0, 0.0]],
        ],
        [
            [[1.0, 0.4, 3.0], [1.0, 1.0, 0.0]],
            [[1.0, 0.4, 3.0], [1.0, 1.0, 0.0]],
            [[1.0, 0.4, 3.0], [1.0, 1.0, 0.0]],
        ],
    ),  # MIMO case - different denominators
    (
        [
            [[1.0, 6.0, 12.0, 7.0], [0.0, 1.0, 4.0, 3.0]],
            [[0.0, 0.0, 1.0, 1.0], [1.0, 8.0, 20.0, 15.0]],
        ],
        [
            [[1.0, 6.0, 11.0, 6.0], [1.0, 6.0, 11.0, 6.0]],
            [[1.0, 6.0, 11.0, 6.0], [1.0, 6.0, 11.0, 6.0]],
        ],
    ),
]


def compare_with_slycot(num_list, den_list):
    """Compute state-space using both methods and compare."""
    if not slycot_check():
        pytest.skip("Slycot not available, skipping test")
    A_min, B_min, C_min, D_min = tf2ss(num_list, den_list)

    # Convert to control.TransferFunction format
    n_outputs = len(num_list)
    n_inputs = len(num_list[0])
    tf_mimo = [
        [
            ctrl.TransferFunction(num_list[i][j], den_list[i][j])
            for j in range(n_inputs)
        ]
        for i in range(n_outputs)
    ]

    # Convert using control.tf2ss (uses slycot if available)
    ss_slycot = ctrl.tf2ss(
        ctrl.append(*[ctrl.append(*row) for row in tf_mimo])
    )
    if isinstance(ss_slycot, ctrl.StateSpace):
        A_slycot, B_slycot, C_slycot, D_slycot = (
            ss_slycot.A,
            ss_slycot.B,
            ss_slycot.C,
            ss_slycot.D,
        )
    else:
        raise ValueError("Failed to convert using control.tf2ss")

    return (A_min, B_min, C_min, D_min), (
        A_slycot,
        B_slycot,
        C_slycot,
        D_slycot,
    )


@pytest.mark.parametrize("num, den", systems)
def test_tf2ss_consistency(num, den):
    (A_min, B_min, C_min, D_min), (A_slycot, B_slycot, C_slycot, D_slycot) = (
        compare_with_slycot(num, den)
    )

    assert np.allclose(A_min, A_slycot, atol=1e-6), "Mismatch in A matrix"
    assert np.allclose(B_min, B_slycot, atol=1e-6), "Mismatch in B matrix"
    assert np.allclose(C_min, C_slycot, atol=1e-6), "Mismatch in C matrix"
    assert np.allclose(D_min, D_slycot, atol=1e-6), "Mismatch in D matrix"


@pytest.mark.parametrize("num, den", systems)
def test_tf2ss_runability(num, den):
    A, B, C, D = tf2ss(_pad_numerators(num), den)


@pytest.mark.parametrize(
    "den",
    [
        [
            [
                [1.0, 0.5],  # (s + 0.5)
                [1.0, 0.5, 0.25],  # (s^2 + 0.5s + 0.25)
            ]
        ],  # [1.0, 1.0, 0.75, 0.125]
        [[[1, 7, 6], [1, -5, -6]]],
        [[[3, -6, -9, 0], [7, 21, 14, 0, 0]]],
        [
            [[3, -6, -9, 0], [1, 0]],
            [[7, 21, 14, 0, 0], [1, 0]],
        ],
        [[[3, -6, -9, 0], [7, 21, 14, 0, 0]], [[1, 7, 6], [1, -5, -6]]],
    ],
)
def test_coeffs_retrieval(den):
    den_our = _get_lcm_norm_coeffs(den, mode="local")
    sys = ctrl.tf(
        [[[1] * len(den) for den in row] for row in den],
        den,
    )
    if isinstance(sys, ctrl.TransferFunction):
        _, den_ctrl, _ = sys._common_den()
        _, den_min, _ = sys.minreal()._common_den()
        assert np.allclose(den_our, den_ctrl, atol=1e-5)
        # TODO: create test evaluating minimum realization when implemented
        # assert np.allclose(den_our, den_min, atol=1e-5)
