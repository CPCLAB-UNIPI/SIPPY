"""ARMAX Example

@author: Giuseppe Armenise, revised by RBdC
"""

import control.matlab as cnt
import matplotlib.pyplot as plt
import numpy as np

from sippy.identification import SystemIdentification, IDData
from sippy.utils.signal_utils import GBN_seq, white_noise_var

## TEST OPTIMIZATION-BASED IDENTIFICATION METHODS for GENERAL INPUT-OUTPUT MODEL

# Define sampling time and Time vector
sampling_time = 1.0  # [s]
end_time = 400  # [s]
npts = int(end_time / sampling_time) + 1
Time = np.linspace(0, end_time, npts)

# Define Generalize Binary Sequence as input signal
switch_probability = 0.08  # [0..1]
[Usim, _, _] = GBN_seq(npts, switch_probability, Range=[-1, 1])
# Reshape to inputs x time_steps format that SIPPY expects
Usim = Usim.reshape(1, -1)

# Define white noise as noise signal
white_noise_variance = [0.01]
e_t = white_noise_var(Usim.size, white_noise_variance)[0]

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

Y1, Time, Xsim = cnt.lsim(g_sample, Usim.T, Time)
plt.figure(0)
plt.plot(Time, Usim.squeeze())
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
Utot = Usim.squeeze() + e_t
plt.figure(2)
plt.plot(Time, Utot)
plt.plot(Time, Ytot)
plt.xlabel("Time")
plt.title(r"Time response y$_k$ = g$\cdot$u$_k$ + h$\cdot$e$_k$")
plt.legend(["u(t) + e(t)", "y_t(t)"])
plt.grid()


##### SYSTEM IDENTIFICATION from collected data using new API

# choose identification mode
mode = "FIXED"

# Create IDData object
import pandas as pd
time_index = pd.date_range("2023-01-01", periods=npts, freq=f"{int(sampling_time*1000)}ms")
data_df = pd.DataFrame({"u": Usim[0, :], "y": Ytot.flatten()}, index=time_index)
data = IDData(data=data_df, inputs=["u"], outputs=["y"], tsample=sampling_time)

if mode == "IC":
    # use Information criterion

    # ARMA - ARARX - ARARMAX
    sys_id_arma = SystemIdentification()
    Id_ARMA = sys_id_arma.identify(y=data.get_output_array(), u=data.get_input_array(),
                                   id_method="ARMA", criterion="BIC", na=[2, 2], nc=[2, 2], delays=[11, 11], 
                                   max_iterations=300)

    sys_id_ararx = SystemIdentification()
    Id_ARARX = sys_id_ararx.identify(y=data.get_output_array(), u=data.get_input_array(),
                                     id_method="ARARX", criterion="BIC", na=[4, 4], nb=[3, 3], nd=[3, 3], 
                                     delays=[11, 11], max_iterations=300)

    sys_id_ararmax = SystemIdentification()
    Id_ARARMAX = sys_id_ararmax.identify(y=data.get_output_array(), u=data.get_input_array(),
                                         id_method="ARARMAX", criterion="BIC", na=[4, 4], nb=[3, 3], nc=[2, 2], nd=[3, 3],
                                         delays=[11, 11], max_iterations=300)

    # OE - BJ - GEN
    sys_id_oe = SystemIdentification()
    Id_OE = sys_id_oe.identify(y=data.get_output_array(), u=data.get_input_array(),
                               id_method="OE", criterion="BIC", nb=[3, 3], nf=[4, 4], delays=[11, 11],
                               max_iterations=300)
    #
    sys_id_bj = SystemIdentification()
    Id_BJ = sys_id_bj.identify(y=data.get_output_array(), u=data.get_input_array(),
                              id_method="BJ", criterion="BIC", nb=[3, 3], nc=[2, 2], nd=[3, 3], nf=[4, 4],
                              delays=[11, 11], max_iterations=300)
    # #
    sys_id_gen = SystemIdentification()
    Id_GEN = sys_id_gen.identify(y=data.get_output_array(), u=data.get_input_array(),
                                 id_method="GEN", criterion="BIC", na=[2, 2], nb=[3, 3], nc=[2, 2], nd=[3, 3], 
                                 nf=[4, 4], delays=[11, 11], max_iterations=300)


elif mode == "FIXED":
    # use fixed model orders

    na_ord = [2]
    nb_ord = [[3]]
    nc_ord = [2]
    nd_ord = [3]
    nf_ord = [4]
    theta = [[11]]

    # ARMA - ARARX - ARARMAX
    sys_id_arma = SystemIdentification()
    Id_ARMA = sys_id_arma.identify(y=data.get_output_array(), u=data.get_input_array(),
                                   id_method="ARMA", na=na_ord, nc=nc_ord, theta=theta)
    # #
    sys_id_ararx = SystemIdentification()
    Id_ARARX = sys_id_ararx.identify(y=data.get_output_array(), u=data.get_input_array(),
                                     id_method="ARARX", na=na_ord, nb=nb_ord, nd=nd_ord, theta=theta, max_iterations=300)
    # #
    sys_id_ararmax = SystemIdentification()
    Id_ARARMAX = sys_id_ararmax.identify(y=data.get_output_array(), u=data.get_input_array(),
                                         id_method="ARARMAX", na=na_ord, nb=nb_ord, nc=nc_ord, nd=nd_ord, theta=theta,
                                         max_iterations=300)

    # OE - BJ - GEN
    sys_id_oe = SystemIdentification()
    Id_OE = sys_id_oe.identify(y=data.get_output_array(), u=data.get_input_array(),
                               id_method="OE", nb=nb_ord, nf=nf_ord, theta=theta, max_iterations=300)
    #
    sys_id_bj = SystemIdentification()
    Id_BJ = sys_id_bj.identify(y=data.get_output_array(), u=data.get_input_array(),
                              id_method="BJ", nb=nb_ord, nc=nc_ord, nd=nd_ord, nf=nf_ord, theta=theta,
                              max_iterations=300)
    # #
    sys_id_gen = SystemIdentification()
    Id_GEN = sys_id_gen.identify(y=data.get_output_array(), u=data.get_input_array(),
                                 id_method="GEN", na=na_ord, nb=nb_ord, nc=nc_ord, nd=nd_ord, nf=nf_ord, theta=theta,
                                 max_iterations=300)
#
_, Y_arma = Id_ARMA.simulate(Usim)
_, Y_ararx = Id_ARARX.simulate(Usim)
_, Y_ararmax = Id_ARARMAX.simulate(Usim)
#
_, Y_oe = Id_OE.simulate(Usim)
_, Y_bj = Id_BJ.simulate(Usim)
_, Y_gen = Id_GEN.simulate(Usim)
# Extract outputs for plotting - squeeze to remove single-dimension axes
Y_arma = Y_arma.squeeze() if Y_arma.ndim > 1 else Y_arma
Y_ararx = Y_ararx.squeeze() if Y_ararx.ndim > 1 else Y_ararx
Y_ararmax = Y_ararmax.squeeze() if Y_ararmax.ndim > 1 else Y_ararmax
Y_oe = Y_oe.squeeze() if Y_oe.ndim > 1 else Y_oe
Y_bj = Y_bj.squeeze() if Y_bj.ndim > 1 else Y_bj
Y_gen = Y_gen.squeeze() if Y_gen.ndim > 1 else Y_gen


# ## Check consistency of the identified system

plt.figure(3)
plt.plot(Time, Usim.squeeze())
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
[U_valid, _, _] = GBN_seq(npts, switch_probability, Range=input_range)
U_valid = U_valid.reshape(1, -1)
white_noise_variance = [0.01]
e_valid = white_noise_var(U_valid.size, white_noise_variance)[0]
#
## Compute time responses for true system with new inputs

Yvalid1, Time, Xsim = cnt.lsim(g_sample, U_valid.T, Time)
Yvalid2, Time, Xsim = cnt.lsim(h_sample, e_valid, Time)
Ytotvalid = Yvalid1 + Yvalid2

# ## Compute time responses for identified system with new inputs

# ARMA - ARARX - ARARMAX
# Yv_arma = Id_ARMA.simulate(U_valid)
#
_, Yv_ararx = Id_ARARX.simulate(U_valid)
#
_, Yv_ararmax = Id_ARARMAX.simulate(U_valid)
#
# OE - BJ
_, Yv_oe = Id_OE.simulate(U_valid)
#
_, Yv_bj = Id_BJ.simulate(U_valid)
#
_, Yv_gen = Id_GEN.simulate(U_valid)
# Extract outputs for plotting - squeeze to remove single-dimension axes
Yv_ararx = Yv_ararx.squeeze() if Yv_ararx.ndim > 1 else Yv_ararx
Yv_ararmax = Yv_ararmax.squeeze() if Yv_ararmax.ndim > 1 else Yv_ararmax
Yv_oe = Yv_oe.squeeze() if Yv_oe.ndim > 1 else Yv_oe
Yv_bj = Yv_bj.squeeze() if Yv_bj.ndim > 1 else Yv_bj
Yv_gen = Yv_gen.squeeze() if Yv_gen.ndim > 1 else Yv_gen

# Plot
plt.figure(7)
plt.plot(Time, U_valid.squeeze())
plt.ylabel("Input GBN")
plt.xlabel("Time")
plt.title("Input, validation data (Switch probability=0.07)")
plt.grid()
plt.show(block=False)

plt.figure(8)
plt.plot(Time, Ytotvalid)
# plt.plot(Time, Yv_arma)
plt.plot(Time, Yv_ararx)
plt.plot(Time, Yv_ararmax)
plt.plot(Time, Yv_oe)
plt.plot(Time, Yv_bj)
plt.plot(Time, Yv_gen)
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


## Note: Advanced plotting (Bode plots, Step responses) would need TF extraction from Harold State objects
## This requires additional implementation to work with control library
print("\n🔍 Advanced analysis features (Bode plots, step responses) require TF extraction from Harold objects")
