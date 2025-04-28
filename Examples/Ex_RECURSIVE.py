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

from sippy_unipi import functionset as fset
from sippy_unipi import system_identification
from sippy_unipi._typing import IOMethods
from sippy_unipi.datasets import load_sample_siso

np.random.seed(0)

modes = ["FIXED", "IC"]
ylegends = ["System", "ARMAX", "ARX", "OE"]

output_dir = create_output_dir(__file__, subdirs=modes)
# TEST RECURSIVE IDENTIFICATION METHODS
n_samples = 401
ts = 1.0
time, Ysim, Usim, g_sys, Yerr, Uerr, h_sys, Ytot, Utot = load_sample_siso(
    n_samples, ts, seed=0
)

fig = plot_responses(
    time,
    [Usim, Uerr, Utot],
    [Ysim, Yerr, Ytot],
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
        time,
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
    [U_valid, _, _] = fset.GBN_seq(
        n_samples, switch_probability, scale=input_range
    )
    white_noise_variance = [0.01]
    e_valid = fset.white_noise_var(U_valid.size, white_noise_variance)[0]
    #
    # Compute time responses for true system with new inputs

    Yvalid1, time, Xsim = cnt.lsim(g_sys, U_valid, time)  # type: ignore
    Yvalid2, time, Xsim = cnt.lsim(h_sys, e_valid, time)
    Ytotvalid = Yvalid1 + Yvalid2

    # ## Compute time responses for identified system with new inputs

    # ARMA - ARARX - ARARMAX
    ys = [Ytotvalid] + [
        fset.validation(sys, U_valid, Ytotvalid, time) for sys in syss
    ]

    # Plot
    fig = plot_response(
        time,
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
    u = np.ones_like(time)
    u[0] = 0

    for tf in ["G", "H"]:
        syss_tfs = [
            locals()[f"{tf.lower()}_sys"],
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

        ys, _ = zip(*[cnt.step(sys, time) for sys in syss_tfs])

        fig = plot_response(
            time,
            u,
            ys,
            legends=[["U"], ylegends],
            titles=["Step Response G(z)", None],
        )
        fig.savefig(output_dir + f"/step_{tf}.png")

    plt.close("all")
