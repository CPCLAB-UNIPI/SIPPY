import numpy as np
from control.matlab import lsim, tf

from .. import functionset as fset
from ._systems_generator import make_tf

# Numerator of input transfer function has 3 roots: nb = 3
NUM_TF_SISO = [1.5, -2.07, 1.3146]

# Common denominator between input and noise transfer functions has 4 roots: na = 4
DEN_TF_SISO = [
    1.0,
    -2.21,
    1.7494,
    -0.584256,
    0.0684029,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
]

# Numerator of noise transfer function has two roots: nc = 2
NUM_NOISE_TF_SISO = [
    1.0,
    0.3,
    0.2,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
]

INPUT_RANGE_SISO = (-1.0, 1.0)

NUM_TF_MIMO = [
    [
        [4, 3.3, 0.0, 0.0],
        [10, 0.0, 0.0],
        [7.0, 5.5, 2.2],
        [-0.9, -0.11, 0.0, 0.0],
    ],
    [
        [-85, -57.5, -27.7],
        [71, 12.3],
        [-0.1, 0.0, 0.0, 0.0],
        [0.994, 0.0, 0.0, 0.0],
    ],
    [
        [0.2, 0.0, 0.0, 0.0],
        [0.821, 0.432, 0.0],
        [0.1, 0.0, 0.0, 0.0],
        [0.891, 0.223],
    ],
]

DEN_TF_MIMO = [
    [1.0, -0.3, -0.25, -0.021, 0.0, 0.0],
    [1.0, -0.4, 0.0, 0.0, 0.0],
    [1.0, -0.1, -0.3, 0.0, 0.0],
]

NUM_NOISE_TF_MIMO = [
    [1.0, 0.85, 0.32, 0.0, 0.0, 0.0],
    [1.0, 0.4, 0.05, 0.0, 0.0],
    [1.0, 0.7, 0.485, 0.22, 0.0],
]

INPUT_RANGES_MIMO = [(-0.33, 0.1), (-1.0, 1.0), (2.3, 5.7), (8.0, 11.5)]


# Helper functions
def generate_inputs(
    n_samples: int,
    ranges: list[tuple[float, float]],
    switch_probability=0.03,
    seed=None,
):
    Usim = np.zeros((len(ranges), n_samples))
    for i, r in enumerate(ranges):
        Usim[i, :] = fset.GBN_seq(
            n_samples, switch_probability, scale=r, seed=seed
        )[0]

    return Usim


def add_noise(n_samples: int, var_list, tfs, time, seed=None):
    Uerr = fset.white_noise_var(n_samples, var_list, seed=seed)
    # TODO: The implementation seems to be wrong. Should match the one from compute_outputs() probably
    Yerr = np.array([lsim(H, Uerr[i, :], time)[0] for i, H in enumerate(tfs)])
    return Yerr, Uerr


def compute_outputs(n_samples: int, tfs, Usim, time):
    Yout = np.zeros((len(tfs), n_samples))
    for i, tfs_ in enumerate(tfs):
        if not isinstance(tfs_, list):
            tfs_ = [tfs_]
        if len(tfs_) != len(Usim):
            raise ValueError(
                f"The number of transfer functions in nesting level 1 must match the number of inputs. Got {len(tfs_)} transfer functions and {len(Usim)} inputs."
            )
        for j, tf_ in enumerate(tfs_):
            Yout[i, :] += lsim(tf_, Usim[j, :], time)[0]
    return Yout


def load_sample_input_tf(
    n_samples: int = 400,
    ts: float = 1.0,
    switch_probability: float = 0.08,  # [0..1]
    seed: int | None = None,
):
    end_time = int(n_samples * ts) - 1  # [s]
    time = np.linspace(0, end_time, n_samples)

    # Define Generalize Binary Sequence as input signal
    Usim = fset.GBN_seq(
        n_samples, switch_probability, scale=INPUT_RANGE_SISO, seed=seed
    )[0]

    # Define transfer functions
    sys = tf(NUM_TF_SISO, DEN_TF_SISO, ts)

    # ## time responses
    Y1, time, Xsim = lsim(sys, Usim, time)  # type: ignore

    return time, Y1, Usim, sys


def load_sample_noise_tf(
    n_samples: int = 400,
    ts: float = 1.0,
    switch_probability: float = 0.08,  # [0..1]
    noise_variance: float = 0.01,
    seed: int | None = None,
):
    end_time = int(n_samples * ts) - 1  # [s]
    time = np.linspace(0, end_time, n_samples)

    # Define Generalize Binary Sequence as input signal
    Usim = fset.GBN_seq(
        n_samples, switch_probability, scale=INPUT_RANGE_SISO, seed=seed
    )[0]

    # Define white noise as noise signal
    white_noise_variance = [0.01]
    e_t = fset.white_noise_var(Usim.size, white_noise_variance, seed=seed)[0]

    # Define transfer functions
    sys = tf(NUM_NOISE_TF_SISO, DEN_TF_SISO, ts)

    # ## time responses
    Y2, time, Xsim = lsim(sys, e_t, time)

    return time, Y2, e_t, sys


def load_sample_siso(
    n_samples: int = 400,
    ts: float = 1.0,
    switch_probability: float = 0.08,  # [0..1]
    seed: int | None = None,
):
    time, Ysim, Usim, g_sys = load_sample_input_tf(n_samples, ts, seed=seed)
    time, Yerr, Uerr, h_sys = load_sample_noise_tf(n_samples, ts, seed=seed)

    Y = Ysim + Yerr
    U = Usim + Uerr

    return time, Ysim, Usim, g_sys, Yerr, Uerr, h_sys, Y, U


def load_sample_mimo(
    n_samples: int = 400,
    ts: float = 1.0,
    input_ranges: list[tuple[float, float]] = INPUT_RANGES_MIMO,
    switch_probability: float = 0.08,  # [0..1]
    seed: int | None = None,
):
    end_time = int(n_samples * ts) - 1  # [s]
    time = np.linspace(0, end_time, n_samples)

    Usim = generate_inputs(n_samples, input_ranges, seed=seed)

    g_sys = make_tf(NUM_TF_MIMO, DEN_TF_MIMO, ts, random_state=seed)
    h_sys = make_tf(NUM_NOISE_TF_MIMO, DEN_TF_MIMO, ts, random_state=seed)

    Yerr, Uerr = add_noise(
        n_samples, [50.0, 100.0, 1.0], h_sys, time, seed=seed
    )

    Ysim = compute_outputs(n_samples, g_sys, Usim, time)
    Y = Ysim + Yerr
    U = Usim.copy()
    # TODO: currently the shape does not match. It should I guess, check TODO in add_noise()
    for i in range(min(Usim.shape[0], Uerr.shape[0])):
        U[i, :] += Uerr[i]

    return time, Ysim, Usim, g_sys, Yerr, Uerr, h_sys, Y, U
