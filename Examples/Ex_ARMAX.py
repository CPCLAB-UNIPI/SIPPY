"""ARMAX Example

@author: Giuseppe Armenise, revised by RBdC
"""

import control.matlab as cnt
import matplotlib.pyplot as plt
import numpy as np
from tf2ss import lsim

import sys

local_root = r"C:\Users\frasc\Desktop\Ricerca\Progetti\Daedalos\SIPPY-1.0.1\SIPPY-1.0.1"

if local_root in sys.path:
    sys.path.remove(local_root)
sys.path.insert(0, local_root)

if "sippy_unipi" in sys.modules:
    del sys.modules["sippy_unipi"]

import sippy_unipi
print("sippy loaded from:", sippy_unipi.__file__)

from sippy_unipi import *
from sippy_unipi import functionset as fset

## TEST IDENTIFICATION METHODS for ARMAX model

# Define sampling time and Time vector
sampling_time = 1.0  # [s]
end_time = 400  # [s]
npts = int(end_time / sampling_time) + 1
Time = np.linspace(0, end_time, npts)

# Define Generalized Binary Noise (GBN) sequence as input signal
switch_probability = 0.08  # [0..1]
Usim, _, _,_ = fset.GBN_seq(npts, switch_probability, Range=[-1, 1])

# Define white noise as noise signal
white_noise_variance = [0.01]
e_t = fset.white_noise_var(Usim.size, white_noise_variance)[0]

#%% Define the system to be identified (as an ARMAX model)

# Numerator of noise transfer function 
# N_H(z) = 1.0 z^14 + 0.3 z^13 + 0.2 z^12

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

# Common denominator between input and noise transfer functions  
# D(z) = 1.0 z^14 - 2.21 z^13 + 1.7494 z^12 - 0.584256 z^11 + 0.0684029 z^10

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

# Numerator of input transfer function has 3 roots: nb = 3 !!!
# N(z) = 1.5z^2 - 2.07z + 1.3146

NUM = [1.5,
       -2.07,
       1.3146
]

# Define G and H transfer functions

g_sample = cnt.tf(NUM, DEN, sampling_time)
h_sample = cnt.tf(NUM_H, DEN, sampling_time)

#%% Input output Identification time responses

# Compute Input reponse
Y1, Time, Xsim = lsim(g_sample, Usim, Time)

plt.figure(figsize=(10, 6))

# Subplot Input
plt.subplot(2, 1, 1)
plt.plot(Time, Usim, color='red', linewidth=2)
plt.ylabel("u(t)", fontsize=12)
plt.title("Input signal", fontsize=14)
plt.grid(True)
plt.tick_params(labelsize=11)

# Subplot Output
plt.subplot(2, 1, 2)
plt.plot(Time, Y1, color='green', linewidth=2)
plt.xlabel("Time [s]", fontsize=12)
plt.ylabel("y(t)", fontsize=12)
plt.title(r"Output response $y_k = g \cdot u_k$", fontsize=14)
plt.grid(True)
plt.tick_params(labelsize=11)

plt.tight_layout()
plt.show(block=False)


#%% Noise output Identification time responses

Y2, Time, Xsim = lsim(h_sample, e_t, Time)

plt.figure(figsize=(10, 6))

# Subplot Noise Input
plt.subplot(2, 1, 1)
plt.plot(Time, e_t, color='red', linewidth=2)
plt.ylabel("e(t)", fontsize=12)
plt.title("Input signal", fontsize=14)
plt.grid(True)
plt.tick_params(labelsize=11)

#  Subplot Noise Output
plt.subplot(2, 1, 2)
plt.plot(Time, Y2, color='green', linewidth=2)
plt.xlabel("Time [s]", fontsize=12)
plt.ylabel("y(t)", fontsize=12)
plt.title(r"Output response $y_k = h \cdot e_k$", fontsize=14)
plt.grid(True)
plt.tick_params(labelsize=11)

plt.tight_layout()
plt.show(block=False)


#%% Total output
# $$Y_t = Y_1 + Y_2 = G.u + H.e$$

Ytot = Y1 + Y2
Utot = Usim + e_t

plt.figure(figsize=(10, 6))

#  Subplot Input (u + e) 
plt.subplot(2, 1, 1)
plt.plot(Time, Utot, color='red', linewidth=2)
plt.ylabel(r"$u(t)+e(t)$", fontsize=12)
plt.title("Total input signal", fontsize=14)
plt.grid(True)
plt.tick_params(labelsize=11)

#  Subplot Output
plt.subplot(2, 1, 2)
plt.plot(Time, Ytot, color='green', linewidth=2)
plt.xlabel("Time [s]", fontsize=12)
plt.ylabel(r"$y(t)$", fontsize=12)
plt.title(r"Output response $y_k = g\cdot u_k + h\cdot e_k$", fontsize=14)
plt.grid(True)
plt.tick_params(labelsize=11)

plt.tight_layout()
plt.show(block=False)



#%% SYSTEM IDENTIFICATION from collected data

# Choose order selection mode
mode = "FIXED"

if mode == "IC":
    # use Information criterion

    Id_ARMAXi = system_identification(
        Ytot,
        Usim,
        "ARMAX",
        IC="AIC",
        na_ord=[1, 5],
        nb_ord=[1, 5],
        nc_ord=[1, 5],
        delays=[11, 11],
        max_iterations=300,
        ARMAX_mod="ILLS",
    )

    Id_ARMAXo = system_identification(
        Ytot,
        Usim,
        "ARMAX",
        IC="AICc",
        na_ord=[1, 5],
        nb_ord=[1, 5],
        nc_ord=[1, 5],
        delays=[11, 11],
        max_iterations=300,
        ARMAX_mod="OPT",
    )

    Id_ARMAXr = system_identification(
        Ytot,
        Usim,
        "ARMAX",
        IC="BIC",
        na_ord=[1, 5],
        nb_ord=[1, 5],
        nc_ord=[1, 5],
        delays=[11, 11],
        max_iterations=300,
        ARMAX_mod="RLLS",
    )


elif mode == "FIXED":
    # use fixed model orders

    na_ord = [4]
    nb_ord = [[3]]
    nc_ord = [2]
    theta = [[11]]

    # ITERATIVE ARMAX
    Id_ARMAXi = system_identification(
        Ytot,
        Usim,
        "ARMAX",
        ARMAX_orders=[na_ord, nb_ord, nc_ord, theta],
        max_iterations=300,
        ARMAX_mod="ILLS",
    )

    # OPTIMIZATION-BASED ARMAX
    Id_ARMAXo = system_identification(
        Ytot,
        Usim,
        "ARMAX",
        ARMAX_orders=[na_ord, nb_ord, nc_ord, theta],
        max_iterations=300,
        ARMAX_mod="OPT",
    )

    # RECURSIVE ARMAX
    Id_ARMAXr = system_identification(
        Ytot,
        Usim,
        "ARMAX",
        ARMAX_orders=[na_ord, nb_ord, nc_ord, theta],
        max_iterations=300,
        ARMAX_mod="RLLS",
    )

Y_armaxi = Id_ARMAXi.Yid.T
Y_armaxo = Id_ARMAXo.Yid.T
Y_armaxr = Id_ARMAXr.Yid.T


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
plt.plot(Time, Y_armaxi)
plt.plot(Time, Y_armaxo)
plt.plot(Time, Y_armaxr)
plt.grid()
plt.xlabel("Time")
plt.ylabel("y(t)")
plt.title("Output, (identification data)")
plt.legend(["System", "ARMAX-I", "ARMAX-0", "ARMAX-R"])
plt.show(block=False)


##### VALIDATION of the identified system:
# ## Generate new time series for input and noise

switch_probability = 0.07  # [0..1]
input_range = [0.5, 1.5]
U_valid, _, _,_ = fset.GBN_seq(npts, switch_probability, Range=input_range)
white_noise_variance = [0.01]
e_valid = fset.white_noise_var(U_valid.size, white_noise_variance)[0]
#
## Compute time responses for true system with new inputs

Yvalid1, Time, Xsim = lsim(g_sample, U_valid, Time)
Yvalid2, Time, Xsim = lsim(h_sample, e_valid, Time)
Ytotvalid = Yvalid1 + Yvalid2

# ## Compute time responses for identified systems with new inputs


# ARMAX - ILLS
Yv_armaxi = fset.validation(Id_ARMAXi, U_valid, Ytotvalid, Time, k = 10)

# ARMAX - OPT
Yv_armaxo = fset.validation(Id_ARMAXo, U_valid, Ytotvalid, Time, k = 2)

# ARMAX - RLLS
Yv_armaxr = fset.validation(Id_ARMAXr, U_valid, Ytotvalid, Time)

# Plot
plt.figure(5)
plt.plot(Time, U_valid)
plt.ylabel("Input GBN")
plt.xlabel("Time")
plt.title("Input, validation data (Switch probability=0.07)")
plt.grid()
plt.show(block=False)

plt.figure(6)
plt.plot(Time, Ytotvalid)
plt.plot(Time, Yv_armaxi.T)
plt.plot(Time, Yv_armaxo.T)
plt.plot(Time, Yv_armaxr.T)
plt.xlabel("Time")
plt.ylabel("y_tot")
plt.legend(["System", "ARMAX-I", "ARMAX-0", "ARMAX-R"])
plt.grid()
plt.show(block=False)

EV = 100 * (1.0 - np.var(Ytotvalid - Yv_armaxi) / np.var(Ytotvalid))

# plt.title("Validation: | RMSE ARMAX_i = {}".format(rmse))
plt.title(f"Validation: | Explained Variance ARMAX_i = {EV}%")


## Bode Plots
w_v = np.logspace(-3, 4, num=701)
plt.figure(7)
mag, fi, om = cnt.frequency_response(g_sample, w_v)
mag1, fi1, om = cnt.frequency_response(Id_ARMAXi.G, w_v)
mag2, fi2, om = cnt.frequency_response(Id_ARMAXo.G, w_v)
mag3, fi3, om = cnt.frequency_response(Id_ARMAXr.G, w_v)
(
    plt.subplot(2, 1, 1),
    plt.loglog(om, mag),
    plt.grid(),
)
plt.loglog(om, mag1), plt.loglog(om, mag2), plt.loglog(om, mag3)
plt.xlabel("w"), plt.ylabel("Amplitude Ratio"), plt.title("Bode Plot G(iw)")
plt.subplot(2, 1, 2), plt.semilogx(om, fi), plt.grid()
(
    plt.semilogx(om, fi1),
    plt.semilogx(om, fi2),
    plt.semilogx(om, fi3),
)
plt.xlabel("w"), plt.ylabel("phase")
plt.legend(["System", "ARMAX-I", "ARMAX-0", "ARMAX-R"])

plt.figure(8)
mag, fi, om = cnt.frequency_response(h_sample, w_v)
mag1, fi1, om = cnt.frequency_response(Id_ARMAXi.H, w_v)
mag2, fi2, om = cnt.frequency_response(Id_ARMAXo.H, w_v)
mag3, fi3, om = cnt.frequency_response(Id_ARMAXr.H, w_v)
(
    plt.subplot(2, 1, 1),
    plt.loglog(om, mag),
    plt.grid(),
)
plt.loglog(om, mag1), plt.loglog(om, mag2), plt.loglog(om, mag3)
plt.xlabel("w"), plt.ylabel("Amplitude Ratio"), plt.title("Bode Plot H(iw)")
plt.subplot(2, 1, 2), plt.semilogx(om, fi), plt.grid()
(
    plt.semilogx(om, fi1),
    plt.semilogx(om, fi2),
    plt.semilogx(om, fi3),
)
plt.xlabel("w"), plt.ylabel("phase")
plt.legend(["System", "ARMAX-I", "ARMAX-0", "ARMAX-R"])


## Step test
# G(z)
plt.figure(9)
yg1 = cnt.step(g_sample, Time)
yg2 = cnt.step(Id_ARMAXi.G, Time)
yg3 = cnt.step(Id_ARMAXo.G, Time)
yg4 = cnt.step(Id_ARMAXr.G, Time)
plt.plot(Time, yg1[0].T)
plt.plot(Time, yg2[0].T)
plt.plot(Time, yg3[0].T)
plt.plot(Time, yg4[0].T)
plt.title("Step Response G(z)")
(
    plt.xlabel("time"),
    plt.ylabel("y(t)"),
    plt.grid(),
)
plt.legend(["System", "ARMAX-I", "ARMAX-0", "ARMAX-R"])
# H(z)
plt.figure(10)
yh1 = cnt.step(h_sample, Time)
yh2 = cnt.step(Id_ARMAXi.H, Time)
yh3 = cnt.step(Id_ARMAXo.H, Time)
yh4 = cnt.step(Id_ARMAXr.H, Time)
plt.plot(Time, yh1[0].T)
plt.plot(Time, yh2[0].T)
plt.plot(Time, yh3[0].T)
plt.plot(Time, yh4[0].T)
plt.title("Step Response H(z)")
(
    plt.xlabel("time"),
    plt.ylabel("y(t)"),
    plt.grid(),
)
plt.legend(["System", "ARMAX-I", "ARMAX-0", "ARMAX-R"])
