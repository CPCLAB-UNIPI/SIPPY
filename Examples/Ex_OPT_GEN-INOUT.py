"""ARMAX Example

@author: Giuseppe Armenise, revised by RBdC
"""

import control.matlab as cnt
import matplotlib.pyplot as plt
import numpy as np
from tf2ss import lsim

from sippy_unipi import functionset as fset
from sippy_unipi import system_identification

## TEST OPTIMIZATION-BASED IDENTIFICATION METHODS for GENERAL INPUT-OUTPUT MODEL

# Define sampling time and Time vector
sampling_time = 1.0  # [s]
end_time = 400  # [s]
npts = int(end_time / sampling_time) + 1
Time = np.linspace(0, end_time, npts)

# Define Generalize Binary Sequence as input signal
switch_probability = 0.08  # [0..1]
[Usim, _, _] = fset.GBN_seq(npts, switch_probability, Range=[-1, 1])

# Define white noise as noise signal
white_noise_variance = [0.01]
e_t = fset.white_noise_var(Usim.size, white_noise_variance)[0]

# ## Define the system (BOX-JENKINS model)

# ### Numerator of noise transfer function has two roots: nc = 2

NUM_H = [1.0, 0.3, 0.2]

# ### Denominator of noise transfer function has three roots: nd = 3

DEN_H = [1.0, -1.2, 0.5, -0.03]

# ### Denominator between input and noise transfer functions has 4 roots: nf = 4

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
h_sample = cnt.tf(NUM_H, DEN_H, sampling_time)

# ## Time responses

# ### Input reponse

Y1, Time, Xsim = lsim(g_sample, Usim, Time)
plt.figure(0)
plt.plot(Time, Usim)
plt.plot(Time, Y1)
plt.xlabel("Time")
plt.title(r"Time response y$_k$(u) = g$\cdot$u$_k$")
plt.legend(["u(t)", "y(t)"])
plt.grid()
plt.show(block=False)

# ### Noise response

Y2, Time, Xsim = lsim(h_sample, e_t, Time)
plt.figure(1)
plt.plot(Time, e_t)
plt.plot(Time, Y2)
plt.xlabel("Time")
plt.title(r"Time response y$_k$(e) = h$\cdot$e$_k$")
plt.legend(["e(t)", "y(t)"])
plt.grid()
plt.show(block=False)

# ## Total output
# $$Y_t = Y_1 + Y_2 = G.u + H.e$$

Ytot = Y1 + Y2
Utot = Usim + e_t
plt.figure(2)
plt.plot(Time, Utot)
plt.plot(Time, Ytot)
plt.xlabel("Time")
plt.title(r"Time response y$_k$ = g$\cdot$u$_k$ + h$\cdot$e$_k$")
plt.legend(["u(t) + e(t)", "y_t(t)"])
plt.grid()


##### SYSTEM IDENTIFICATION from collected data

# choose identification mode
mode = "FIXED"

if mode == "IC":
    # use Information criterion

    # ARMA - ARARX - ARARMAX
    Id_ARMA = system_identification(
        Ytot,
        Usim,
        "ARMA",
        IC="BIC",
        na_ord=[2, 2],
        nc_ord=[2, 2],
        delays=[11, 11],
        max_iterations=300,
    )

    Id_ARARX = system_identification(
        Ytot,
        Usim,
        "ARARX",
        IC="BIC",
        na_ord=[4, 4],
        nb_ord=[3, 3],
        nd_ord=[3, 3],
        delays=[11, 11],
        max_iterations=300,
    )

    Id_ARARMAX = system_identification(
        Ytot,
        Usim,
        "ARARMAX",
        IC="BIC",
        na_ord=[4, 4],
        nb_ord=[3, 3],
        nc_ord=[2, 2],
        nd_ord=[3, 3],
        delays=[11, 11],
        max_iterations=300,
    )

    # OE - BJ - GEN
    Id_OE = system_identification(
        Ytot,
        Usim,
        "OE",
        IC="BIC",
        nb_ord=[3, 3],
        nf_ord=[4, 4],
        delays=[11, 11],
        max_iterations=300,
    )
    #
    Id_BJ = system_identification(
        Ytot,
        Usim,
        "BJ",
        IC="BIC",
        nb_ord=[3, 3],
        nc_ord=[2, 2],
        nd_ord=[3, 3],
        nf_ord=[4, 4],
        delays=[11, 11],
        max_iterations=300,
    )
    # #
    Id_GEN = system_identification(
        Ytot,
        Usim,
        "GEN",
        IC="BIC",
        na_ord=[2, 2],
        nb_ord=[3, 3],
        nc_ord=[2, 2],
        nd_ord=[3, 3],
        nf_ord=[4, 4],
        delays=[11, 11],
        max_iterations=300,
    )


elif mode == "FIXED":
    # use fixed model orders

    na_ord = [2]
    nb_ord = [[3]]
    nc_ord = [2]
    nd_ord = [3]
    nf_ord = [4]
    theta = [[11]]

    # ARMA - ARARX - ARARMAX
    Id_ARMA = system_identification(
        Ytot, Usim, "ARMA", ARMA_orders=[na_ord, nc_ord, theta]
    )
    # #
    Id_ARARX = system_identification(
        Ytot,
        Usim,
        "ARARX",
        ARARX_orders=[na_ord, nb_ord, nd_ord, theta],
        max_iterations=300,
    )
    # #
    Id_ARARMAX = system_identification(
        Ytot,
        Usim,
        "ARARMAX",
        ARARMAX_orders=[na_ord, nb_ord, nc_ord, nd_ord, theta],
        max_iterations=300,
    )

    # OE - BJ - GEN
    Id_OE = system_identification(
        Ytot, Usim, "OE", OE_orders=[nb_ord, nf_ord, theta], max_iterations=300
    )
    #
    Id_BJ = system_identification(
        Ytot,
        Usim,
        "BJ",
        BJ_orders=[nb_ord, nc_ord, nd_ord, nf_ord, theta],
        max_iterations=300,
    )
    # #
    Id_GEN = system_identification(
        Ytot,
        Usim,
        "GEN",
        GEN_orders=[na_ord, nb_ord, nc_ord, nd_ord, nf_ord, theta],
        max_iterations=300,
    )
#
Y_arma = Id_ARMA.Yid.T
Y_ararx = Id_ARARX.Yid.T
Y_ararmax = Id_ARARMAX.Yid.T
#
Y_oe = Id_OE.Yid.T
Y_bj = Id_BJ.Yid.T
Y_gen = Id_GEN.Yid.T


# ## Check consistency of the identified system

plt.figure(3)
plt.plot(Time, Usim)
plt.ylabel("Input GBN")
plt.xlabel("Time")
plt.title("Input, identification data (Switch probability=0.08)")
plt.grid()
plt.show(block=False)

plt.figure(4)
plt.plot(Time, Ytot)
# plt.plot(Time, Y_arma)
plt.plot(Time, Y_ararx)
plt.plot(Time, Y_ararmax)
plt.plot(Time, Y_oe)
plt.plot(Time, Y_bj)
plt.plot(Time, Y_gen)
plt.grid()
plt.xlabel("Time")
plt.ylabel("y(t)")
plt.title("Output, (identification data)")
plt.legend(["System", "ARMA", "ARARX", "ARARMAX", "OE", "BJ", "GEN"])
plt.show(block=False)


##### VALIDATION of the identified system:
# ## Generate new time series for input and noise

switch_probability = 0.07  # [0..1]
input_range = [0.5, 1.5]
[U_valid, _, _] = fset.GBN_seq(npts, switch_probability, Range=input_range)
white_noise_variance = [0.01]
e_valid = fset.white_noise_var(U_valid.size, white_noise_variance)[0]
#
## Compute time responses for true system with new inputs

Yvalid1, Time, Xsim = lsim(g_sample, U_valid, Time)
Yvalid2, Time, Xsim = lsim(h_sample, e_valid, Time)
Ytotvalid = Yvalid1 + Yvalid2

# ## Compute time responses for identified system with new inputs


# ARMA - ARARX - ARARMAX
# Yv_arma = fset.validation(Id_ARMA,U_valid,Ytotvalid,Time)
#
Yv_ararx = fset.validation(Id_ARARX, U_valid, Ytotvalid, Time)
#
Yv_ararmax = fset.validation(Id_ARARMAX, U_valid, Ytotvalid, Time)
#
# OE - BJ
Yv_oe = fset.validation(Id_OE, U_valid, Ytotvalid, Time)
#
Yv_bj = fset.validation(Id_BJ, U_valid, Ytotvalid, Time)
#
Yv_gen = fset.validation(Id_GEN, U_valid, Ytotvalid, Time)

# Plot
plt.figure(7)
plt.plot(Time, U_valid)
plt.ylabel("Input GBN")
plt.xlabel("Time")
plt.title("Input, validation data (Switch probability=0.07)")
plt.grid()
plt.show(block=False)

plt.figure(8)
plt.plot(Time, Ytotvalid)
# plt.plot(Time, Yv_arma)
plt.plot(Time, Yv_ararx.T)
plt.plot(Time, Yv_ararmax.T)
plt.plot(Time, Yv_oe.T)
plt.plot(Time, Yv_bj.T)
plt.plot(Time, Yv_gen.T)
plt.xlabel("Time")
plt.ylabel("y_tot")
plt.legend(["System", "ARMA", "ARARX", "ARARMAX", "OE", "BJ", "GEN"])
plt.grid()
plt.show(block=False)

# rmse = np.round(np.sqrt(np.mean((Ytotvalid - Yv_armaxi.T) ** 2)), 2)
EV = 100.0 * (
    np.round((1.0 - np.mean((Ytotvalid - Yv_bj) ** 2) / np.std(Ytotvalid)), 2)
)
# plt.title("Validation: | RMSE ARMAX_i = {}".format(rmse))
plt.title(f"Validation: | Explained Variance BJ = {EV}%")


## Bode Plots
w_v = np.logspace(-3, 4, num=701)
plt.figure(11)
mag, fi, om = cnt.frequency_response(g_sample, w_v)
mag1, fi1, om = cnt.frequency_response(Id_ARMA.G, w_v)
mag2, fi2, om = cnt.frequency_response(Id_ARARX.G, w_v)
mag3, fi3, om = cnt.frequency_response(Id_ARARMAX.G, w_v)
mag4, fi4, om = cnt.frequency_response(Id_OE.G, w_v)
mag5, fi5, om = cnt.frequency_response(Id_BJ.G, w_v)
mag6, fi6, om = cnt.frequency_response(Id_GEN.G, w_v)
(
    plt.subplot(2, 1, 1),
    plt.loglog(om, mag),
    plt.grid(),
)
(
    plt.loglog(om, mag1),
    plt.loglog(om, mag2),
    plt.loglog(om, mag3),
    plt.loglog(om, mag4),
    plt.loglog(om, mag5),
    plt.loglog(om, mag6),
)
plt.xlabel("w"), plt.ylabel("Amplitude Ratio"), plt.title("Bode Plot G(iw)")
plt.subplot(2, 1, 2), plt.semilogx(om, fi), plt.grid()
(
    plt.semilogx(om, fi1),
    plt.semilogx(om, fi2),
    plt.semilogx(om, fi3),
    plt.semilogx(om, fi4),
    plt.semilogx(om, fi5),
    plt.semilogx(om, fi6),
)
plt.xlabel("w"), plt.ylabel("phase")
plt.legend(["System", "ARMA", "ARARX", "ARARMAX", "OE", "BJ", "GEN"])

plt.figure(12)
mag, fi, om = cnt.frequency_response(h_sample, w_v)
mag, fi, om = cnt.frequency_response(g_sample, w_v)
mag1, fi1, om = cnt.frequency_response(Id_ARMA.H, w_v)
mag2, fi2, om = cnt.frequency_response(Id_ARARX.H, w_v)
mag3, fi3, om = cnt.frequency_response(Id_ARARMAX.H, w_v)
mag4, fi4, om = cnt.frequency_response(Id_OE.H, w_v)
mag5, fi5, om = cnt.frequency_response(Id_BJ.H, w_v)
mag6, fi6, om = cnt.frequency_response(Id_GEN.H, w_v)
(
    plt.subplot(2, 1, 1),
    plt.loglog(om, mag),
    plt.grid(),
)
(
    plt.loglog(om, mag1),
    plt.loglog(om, mag2),
    plt.loglog(om, mag3),
    plt.loglog(om, mag4),
    plt.loglog(om, mag5),
    plt.loglog(om, mag6),
)
plt.xlabel("w"), plt.ylabel("Amplitude Ratio"), plt.title("Bode Plot H(iw)")
plt.subplot(2, 1, 2), plt.semilogx(om, fi), plt.grid()
(
    plt.semilogx(om, fi1),
    plt.semilogx(om, fi2),
    plt.semilogx(om, fi3),
    plt.semilogx(om, fi4),
    plt.semilogx(om, fi5),
    plt.semilogx(om, fi6),
)
plt.xlabel("w"), plt.ylabel("phase")
plt.legend(["System", "ARMA", "ARARX", "ARARMAX", "OE", "BJ", "GEN"])


## Step test
# G(z)
plt.figure(13)
yg1 = cnt.step(g_sample, Time)
yg2 = cnt.step(Id_ARMA.G, Time)
yg3 = cnt.step(Id_ARARX.G, Time)
yg4 = cnt.step(Id_ARARMAX.G, Time)
yg5 = cnt.step(Id_OE.G, Time)
yg6 = cnt.step(Id_BJ.G, Time)
yg7 = cnt.step(Id_GEN.G, Time)
plt.plot(Time, yg1[0].T)
plt.plot(Time, yg2[0].T)
plt.plot(Time, yg3[0].T)
plt.plot(Time, yg4[0].T)
plt.plot(Time, yg5[0].T)
plt.plot(Time, yg6[0].T)
plt.plot(Time, yg7[0].T)
plt.title("Step Response G(z)")
(
    plt.xlabel("time"),
    plt.ylabel("y(t)"),
    plt.grid(),
)
plt.legend(["System", "ARMA", "ARARX", "ARARMAX", "OE", "BJ", "GEN"])
# H(z)
plt.figure(14)
yh1 = cnt.step(h_sample, Time)
yh2 = cnt.step(Id_ARMA.H, Time)
yh3 = cnt.step(Id_ARARX.H, Time)
yh4 = cnt.step(Id_ARARMAX.H, Time)
yh5 = cnt.step(Id_OE.H, Time)
yh6 = cnt.step(Id_BJ.H, Time)
yh7 = cnt.step(Id_GEN.H, Time)
plt.plot(Time, yh1[0].T)
plt.plot(Time, yh2[0].T)
plt.plot(Time, yh3[0].T)
plt.plot(Time, yh4[0].T)
plt.plot(Time, yh5[0].T)
plt.plot(Time, yh6[0].T)
plt.plot(Time, yh7[0].T)
plt.title("Step Response H(z)")
(
    plt.xlabel("time"),
    plt.ylabel("y(t)"),
    plt.grid(),
)
plt.legend(["System", "ARMA", "ARARX", "ARARMAX", "OE", "BJ", "GEN"])
