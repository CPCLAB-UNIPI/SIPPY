"""
ARMAX Example

@author: Giuseppe Armenise, revised by RBdC
"""

# Checking path to access other files
import control.matlab as cnt
import matplotlib.pyplot as plt
import numpy as np
from utils import (
    W_V,
    create_output_dir,
    plot_bode,
    plot_response,
    plot_responses,
)

from sippy import functionset as fset
from sippy import system_identification
from sippy.typing import IOMethods

np.random.seed(0)

modes = ["FIXED", "IC"]
ylegends = ["System", "ARMAX", "ARX", "OE"]

output_dir = create_output_dir(__file__, subdirs=modes)
# TEST RECURSIVE IDENTIFICATION METHODS

# Define sampling time and Time vector
sampling_time = 1.0  # [s]
end_time = 400  # [s]
npts = int(end_time // sampling_time) + 1
Time = np.linspace(0, end_time, npts)

# Define Generalize Binary Sequence as input signal
switch_probability = 0.08  # [0..1]
[Usim, _, _] = fset.GBN_seq(npts, switch_probability, Range=[-1, 1])

# Define white noise as noise signal
white_noise_variance = [0.01]
e_t = fset.white_noise_var(Usim.size, white_noise_variance)[0]

# ## Define the system (ARMAX model)

# ### Numerator of noise transfer function has two roots: nc = 2

NUM_H = [
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

# ### Common denominator between input and noise transfer functions has 4 roots: na = 4

DEN = [
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

# ### Numerator of input transfer function has 3 roots: nb = 3

NUM = [1.5, -2.07, 1.3146]

# ### Define transfer functions

g_sample = cnt.tf(NUM, DEN, sampling_time)
h_sample = cnt.tf(NUM_H, DEN, sampling_time)

# ## Time responses
Y1, Time, Xsim = cnt.lsim(g_sample, Usim, Time)  # type: ignore
Y2, Time, Xsim = cnt.lsim(h_sample, e_t, Time)
Ytot = np.array(Y1) + np.array(Y2)
Utot: np.ndarray = Usim + e_t

fig = plot_responses(
    Time,
    [Usim, e_t, Utot],
    [Y1, Y2, Ytot],
    ["u", "e", ["u", "e"]],
)

fig.savefig(output_dir + "/responses.png")


# SYSTEM IDENTIFICATION from collected data


for mode in modes:
    if mode == "IC":
        # use Information criterion
        na_ord = (4, 4)
        nb_ord = (3, 3)
        nc_ord = (2, 2)
        nd_ord = (3, 3)
        nf_ord = (4, 4)
        theta = (11, 11)

    elif mode == "FIXED":
        # use fixed model orders
        na_ord = [4]
        nb_ord = [[3]]
        nc_ord = [2]
        nf_ord = [4]
        theta = [[11]]

    identification_params: dict[
        IOMethods,
        tuple[tuple[list[int] | list[list[int]] | tuple[int, int], ...], dict],
    ] = {
        "ARMAX": (
            (na_ord, nb_ord, nc_ord, theta),
            {"IC": "BIC", "id_mode": "RLLS"},
        ),
        "ARX": ((na_ord, nb_ord, theta), {"IC": "BIC", "id_mode": "RLLS"}),
        "OE": ((nb_ord, nf_ord, theta), {"IC": "BIC", "id_mode": "RLLS"}),
    }

    syss = []
    for method, orders_params in identification_params.items():
        orders, params = orders_params
        sys_id = system_identification(
            Ytot, Usim, method, *orders, max_iter=300, **params
        )
        syss.append(sys_id)

    ys = [Ytot] + [getattr(sys, "Yid").T for sys in syss]

    # ## Check consistency of the identified system
    fig = plot_response(
        Time,
        Usim,
        ys,
        legends=[["U"], ylegends],
        titles=[
            "Input, identification data (Switch probability=0.08)",
            "Output (identification data)",
        ],
    )
    fig.savefig(output_dir + f"/{mode}/system_consistency.png")

    # VALIDATION of the identified system:
    # ## Generate new time series for input and noise

    switch_probability = 0.07  # [0..1]
    input_range = [0.5, 1.5]
    [U_valid, _, _] = fset.GBN_seq(npts, switch_probability, Range=input_range)
    white_noise_variance = [0.01]
    e_valid = fset.white_noise_var(U_valid.size, white_noise_variance)[0]
    #
    # Compute time responses for true system with new inputs

    Yvalid1, Time, Xsim = cnt.lsim(g_sample, U_valid, Time)  # type: ignore
    Yvalid2, Time, Xsim = cnt.lsim(h_sample, e_valid, Time)
    Ytotvalid = Yvalid1 + Yvalid2

    # ## Compute time responses for identified system with new inputs

    # ARMA - ARARX - ARARMAX
    ys = [Ytotvalid] + [
        fset.validation(sys, U_valid, Ytotvalid, Time) for sys in syss
    ]

    # Plot
    fig = plot_response(
        Time,
        Usim,
        ys,
        legends=[["U"], ylegends],
        titles=[
            "Input, identification data (Switch probability=0.07)",
            "Output (identification data)",
        ],
    )
    fig.savefig(output_dir + f"/{mode}/system_validation.png")

    for y, sys in zip(ys[1:], syss):
        yv = y.T
        rmse = np.sqrt(np.mean((Ytotvalid - yv) ** 2))
        R2 = 1 - np.sum((Ytotvalid - yv) ** 2) / np.sum(
            (Ytotvalid - np.mean(Ytotvalid)) ** 2
        )
        print(f"RMSE = {rmse:.2f}")
        print(f"R2 = {R2:.02f}")

    # Step tests
    u = np.ones_like(Time)
    u[0] = 0

    for tf in ["G", "H"]:
        syss_tfs = [
            locals()[f"{tf.lower()}_sample"],
            *[getattr(sys, tf) for sys in syss],
        ]
        mags, fis, oms = zip(*[cnt.bode(sys, W_V) for sys in syss_tfs])

        fig = plot_bode(
            oms[0],
            mags,
            fis,
            ylegends,
        )
        fig.savefig(output_dir + f"/bode_{tf}.png")

        ys, _ = zip(*[cnt.step(sys, Time) for sys in syss_tfs])

        fig = plot_response(
            Time,
            u,
            ys,
            legends=[["U"], ylegends],
            titles=["Step Response G(z)", None],
        )
        fig.savefig(output_dir + f"/step_{tf}.png")

    plt.close("all")
