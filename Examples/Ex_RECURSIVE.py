"""ARMAX Example

@author: Giuseppe Armenise, revised by RBdC
"""

import control.matlab as cnt
import matplotlib.pyplot as plt
import numpy as np
import warnings

from sippy.identification import SystemIdentification, IDData
import pandas as pd

# Suppress the harmless control library warning about return_x
warnings.filterwarnings(action='ignore', message='return_x specified for a transfer function system')
from sippy.utils.signal_utils import GBN_seq, white_noise_var

# Suppress the harmless control library warning about return_x
warnings.filterwarnings(action='ignore', message='return_x specified for a transfer function system')

## TEST RECURSIVE IDENTIFICATION METHODS

# Define sampling time and Time vector
sampling_time = 1.0  # [s]
end_time = 400  # [s]
npts = int(end_time / sampling_time) + 1
Time = np.linspace(0, end_time, npts)

# Define Generalize Binary Sequence as input signal
switch_probability = 0.08  # [0..1]
[Usim, _, _] = GBN_seq(npts, switch_probability, Range=[-1, 1])

# Define white noise as noise signal
white_noise_variance = [0.01]
e_t = white_noise_var(Usim.size, white_noise_variance)[0]

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

# ### Input reponse

Y1, Time, Xsim = cnt.lsim(g_sample, Usim, Time)
plt.figure(0)
plt.plot(Time, Usim)
plt.plot(Time, Y1)
plt.xlabel("Time")
plt.title(r"Time response y$_k$(u) = g$\cdot$u$_k$")
plt.legend(["u(t)", "y(t)"])
plt.grid()
plt.show(block=False)

# ### Noise response

Y2, Time, Xsim = cnt.lsim(h_sample, e_t, Time)
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


##### SYSTEM IDENTIFICATION from collected data using new API

# choose RECURSIVE identification mode: ARMAX - ARX - OE
mode = "FIXED"

# Create IDData object - need to convert from numpy arrays to DataFrame
import pandas as pd
time_index = pd.date_range("2023-01-01", periods=npts, freq=f"{int(sampling_time*1000)}ms")
data_dict = {}
data_dict["u"] = Usim
data_dict["y"] = Ytot

data_df = pd.DataFrame(data_dict, index=time_index)
inputs = ["u"]
outputs = ["y"]
data = IDData(data=data_df, inputs=inputs, outputs=outputs, tsample=sampling_time)

if mode == "IC":
    # use Information criterion

    sys_id_armax = SystemIdentification()
    Id_ARMAX = sys_id_armax.identify(
        y=data.get_output_array(), u=data.get_input_array(), 
        id_method="ARMAX", na=[4], nb=[3], nc=[2], theta=[11],
        max_iterations=300, ARMAX_mod="RLLS"
    )

    sys_id_arx = SystemIdentification()
    Id_ARX = sys_id_arx.identify(
        y=data.get_output_array(), u=data.get_input_array(),
        id_method="ARX", na=[4], nb=[3], theta=[11],
        max_iterations=300
    )

    sys_id_oe = SystemIdentification()
    Id_OE = sys_id_oe.identify(
        y=data.get_output_array(), u=data.get_input_array(),
        id_method="OE", nb=[3], nf=[4], theta=[11],
        max_iterations=300
    )


elif mode == "FIXED":
    # use fixed model orders

    na_ord = [4]
    nb_ord = [[3]]
    nc_ord = [2]
    nf_ord = [4]
    theta = [[11]]

    sys_id_armax = SystemIdentification()
    Id_ARMAX = sys_id_armax.identify(
        y=data.get_output_array(), u=data.get_input_array(), 
        id_method="ARMAX", na=na_ord, nb=nb_ord, nc=nc_ord, theta=theta,
        max_iterations=300, ARMAX_mod="RLLS"
    )

    sys_id_arx = SystemIdentification()
    Id_ARX = sys_id_arx.identify(
        y=data.get_output_array(), u=data.get_input_array(),
        id_method="ARX", na=na_ord, nb=nb_ord, theta=theta, max_iterations=300
    )

    sys_id_oe = SystemIdentification()
    Id_OE = sys_id_oe.identify(
        y=data.get_output_array(), u=data.get_input_array(),
        id_method="OE", nb=nb_ord, nf=nf_ord, theta=theta, max_iterations=300
    )

# Use one-step-ahead predictions (Yid) for identification data
Y_armax = Id_ARMAX.Yid.squeeze() if Id_ARMAX.Yid is not None else Id_ARMAX.simulate(Usim.reshape(1, -1))[1].squeeze()
Y_arx = Id_ARX.Yid.squeeze() if Id_ARX.Yid is not None else Id_ARX.simulate(Usim.reshape(1, -1))[1].squeeze()
Y_oe = Id_OE.Yid.squeeze() if Id_OE.Yid is not None else Id_OE.simulate(Usim.reshape(1, -1))[1].squeeze()


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
plt.plot(Time, Y_armax)
plt.plot(Time, Y_arx)
plt.plot(Time, Y_oe)
plt.grid()
plt.xlabel("Time")
plt.ylabel("y(t)")
plt.title("Output, (identification data)")
plt.legend(["System", "ARMAX", "ARX", "OE"])
plt.show(block=False)


##### VALIDATION of the identified system:
# ## Generate new time series for input and noise

switch_probability = 0.07  # [0..1]
input_range = [0.5, 1.5]
[U_valid, _, _] = GBN_seq(npts, switch_probability, Range=input_range)
white_noise_variance = [0.01]
e_valid = white_noise_var(U_valid.size, white_noise_variance)[0]
#
## Compute time responses for true system with new inputs

Yvalid1, Time, Xsim = cnt.lsim(g_sample, U_valid, Time)
Yvalid2, Time, Xsim = cnt.lsim(h_sample, e_valid, Time)
Ytotvalid = Yvalid1 + Yvalid2

# ## Compute time responses for identified system with new inputs

# ARMAX
_, Yv_armax = Id_ARMAX.simulate(U_valid.reshape(1, -1))

# ARX
_, Yv_arx = Id_ARX.simulate(U_valid.reshape(1, -1))

# OE
_, Yv_oe = Id_OE.simulate(U_valid.reshape(1, -1))

# Squeeze outputs to 1D for plotting
Yv_armax = Yv_armax.squeeze()
Yv_arx = Yv_arx.squeeze()
Yv_oe = Yv_oe.squeeze()

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
plt.plot(Time, Yv_armax)
plt.plot(Time, Yv_arx)
plt.plot(Time, Yv_oe)
plt.xlabel("Time")
plt.ylabel("y_tot")
plt.legend(["System", "ARMAX", "ARX", "OE"])
plt.grid()
plt.show(block=False)

# rmse = np.round(np.sqrt(np.mean((Ytotvalid - Yv_armaxi.T) ** 2)), 2)
EV = 100 * (
    np.round(
        (1.0 - np.mean((Ytotvalid - Yv_armax) ** 2) / np.std(Ytotvalid)), 2
    )
)
# plt.title("Validation: | RMSE ARMAX_i = {}".format(rmse))
plt.title(f"Validation: | Explained Variance ARMAX = {EV}%")


## Bode Plots using transfer functions
w_v = np.logspace(-3, 4, num=701)
plt.figure(7)

# Get frequency responses for G(z) - deterministic transfer function
mag, fi, om = cnt.frequency_response(g_sample, w_v)
mag1, fi1, om = cnt.frequency_response(Id_ARMAX.G_tf, w_v) if Id_ARMAX.G_tf else cnt.freqresp(cnt.ss(Id_ARMAX.A, Id_ARMAX.B, Id_ARMAX.C, Id_ARMAX.D, sampling_time), w_v)
mag2, fi2, om = cnt.frequency_response(Id_ARX.G_tf, w_v) if Id_ARX.G_tf else cnt.freqresp(cnt.ss(Id_ARX.A, Id_ARX.B, Id_ARX.C, Id_ARX.D, sampling_time), w_v)
mag3, fi3, om = cnt.frequency_response(Id_OE.G_tf, w_v) if Id_OE.G_tf else cnt.freqresp(cnt.ss(Id_OE.A, Id_OE.B, Id_OE.C, Id_OE.D, sampling_time), w_v)

plt.subplot(2, 1, 1)
plt.loglog(om, mag)
plt.grid()
plt.loglog(om, mag1.squeeze() if hasattr(mag1, 'squeeze') else mag1)
plt.loglog(om, mag2.squeeze() if hasattr(mag2, 'squeeze') else mag2)
plt.loglog(om, mag3.squeeze() if hasattr(mag3, 'squeeze') else mag3)
plt.xlabel("w")
plt.ylabel("Amplitude Ratio")
plt.title("Bode Plot G(iw)")
plt.subplot(2, 1, 2)
plt.semilogx(om, fi)
plt.grid()
plt.semilogx(om, fi1.squeeze() if hasattr(fi1, 'squeeze') else fi1)
plt.semilogx(om, fi2.squeeze() if hasattr(fi2, 'squeeze') else fi2)
plt.semilogx(om, fi3.squeeze() if hasattr(fi3, 'squeeze') else fi3)
plt.xlabel("w")
plt.ylabel("phase")
plt.legend(["System", "ARMAX", "ARX", "OE"])

plt.figure(8)
# Get frequency responses for H(z) - noise transfer function
mag, fi, om = cnt.frequency_response(h_sample, w_v)
mag1, fi1, om = cnt.frequency_response(Id_ARMAX.H_tf, w_v) if Id_ARMAX.H_tf else cnt.freqresp(cnt.ss(Id_ARMAX.A, Id_ARMAX.B, Id_ARMAX.C, Id_ARMAX.D, sampling_time), w_v)
mag2, fi2, om = cnt.frequency_response(Id_ARX.H_tf, w_v) if Id_ARX.H_tf else cnt.freqresp(cnt.ss(Id_ARX.A, Id_ARX.B, Id_ARX.C, Id_ARX.D, sampling_time), w_v)
mag3, fi3, om = cnt.frequency_response(Id_OE.H_tf, w_v) if Id_OE.H_tf else cnt.freqresp(cnt.ss(Id_OE.A, Id_OE.B, Id_OE.C, Id_OE.D, sampling_time), w_v)

plt.subplot(2, 1, 1)
plt.loglog(om, mag)
plt.grid()
plt.loglog(om, mag1.squeeze() if hasattr(mag1, 'squeeze') else mag1)
plt.loglog(om, mag2.squeeze() if hasattr(mag2, 'squeeze') else mag2)
plt.loglog(om, mag3.squeeze() if hasattr(mag3, 'squeeze') else mag3)
plt.xlabel("w")
plt.ylabel("Amplitude Ratio")
plt.title("Bode Plot H(iw)")
plt.subplot(2, 1, 2)
plt.semilogx(om, fi)
plt.grid()
plt.semilogx(om, fi1.squeeze() if hasattr(fi1, 'squeeze') else fi1)
plt.semilogx(om, fi2.squeeze() if hasattr(fi2, 'squeeze') else fi2)
plt.semilogx(om, fi3.squeeze() if hasattr(fi3, 'squeeze') else fi3)
plt.xlabel("w")
plt.ylabel("phase")
plt.legend(["System", "ARMAX", "ARX", "OE"])


## Step test using transfer functions
# G(z) - step response of deterministic transfer function
plt.figure(9)
# True system step response
step_input = np.ones((1, len(Time)))
yg1, _, _ = cnt.lsim(g_sample, step_input[0, :], Time)

# Identified models step responses using G_tf
yg2, _, _ = cnt.lsim(Id_ARMAX.G_tf, step_input[0, :], Time) if Id_ARMAX.G_tf else (Id_ARMAX.simulate(step_input)[1].squeeze(), None, None)
yg3, _, _ = cnt.lsim(Id_ARX.G_tf, step_input[0, :], Time) if Id_ARX.G_tf else (Id_ARX.simulate(step_input)[1].squeeze(), None, None)
yg4, _, _ = cnt.lsim(Id_OE.G_tf, step_input[0, :], Time) if Id_OE.G_tf else (Id_OE.simulate(step_input)[1].squeeze(), None, None)

plt.plot(Time, yg1)
plt.plot(Time, yg2)
plt.plot(Time, yg3)
plt.plot(Time, yg4)
plt.title("Step Response G(z)")
plt.xlabel("time")
plt.ylabel("y(t)")
plt.grid()
plt.legend(["System", "ARMAX", "ARX", "OE"])

# H(z) - step response of noise transfer function
plt.figure(10)
# True system step response
yh1, _, _ = cnt.lsim(h_sample, step_input[0, :], Time)

# Identified models step responses using H_tf (now available!)
yh2, _, _ = cnt.lsim(Id_ARMAX.H_tf, step_input[0, :], Time) if Id_ARMAX.H_tf else (yg2, None, None)
yh3, _, _ = cnt.lsim(Id_ARX.H_tf, step_input[0, :], Time) if Id_ARX.H_tf else (yg3, None, None)
yh4, _, _ = cnt.lsim(Id_OE.H_tf, step_input[0, :], Time) if Id_OE.H_tf else (yg4, None, None)

plt.plot(Time, yh1)
plt.plot(Time, yh2)
plt.plot(Time, yh3)
plt.plot(Time, yh4)
plt.title("Step Response H(z)")
plt.xlabel("time")
plt.ylabel("y(t)")
plt.grid()
plt.legend(["System", "ARMAX", "ARX", "OE"])

plt.show()
