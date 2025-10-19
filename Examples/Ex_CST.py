"""Created on Mon May 28 13:03:03 2018

@author: Riccardo Bacci di Capaci

CST example

A Continuous Stirred Tank to be identified from input-output data

"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sippy.identification import SystemIdentification, IDData
from sippy.utils.signal_utils import GBN_seq, white_noise_var

# sampling time
ts = 1.0  # [min]

# time settings (t final, samples number, samples vector)
tfin = 1000
npts = int(tfin / ts) + 1
Time = np.linspace(0, tfin, npts)

# Data
V = 10.0  # tank volume [m^3]         --> assumed to be constant
ro = 1100.0  # solution density [kg/m^3] --> assumed to be constant
cp = 4.180  # specific heat [kJ/kg*K]    --> assumed to be constant
Lam = 2272.0  # latent heat   [kJ/kg]     --> assumed to be constant (Tvap = 100°C, Pvap = 1atm)
# initial conditions
# Ca_0
# Tin_0


# VARIABLES

# 4 Inputs
# - as v. manipulated
# Input Flow rate Fin           [m^3/min]
# Steam Flow rate W             [kg/min]
# - as disturbances
# Input Concentration Ca_in     [kg salt/m^3 solution]
# Input Temperature T_in        [°C]
# U = [F, W, Ca_in, T_in]
m = 4

# 2 Outputs
# Output Concentration Ca       [kg salt/m^3 solution]  (Ca = Ca_out)
# Output Temperature T          [°C]                    (T = T_out)
# X = [Ca, T]
p = 2


# Function with Nonlinear System Dynamics
def Fdyn(X, U):
    # Balances

    # V is constant ---> perfect Level Control
    # ro*F_in = ro*F_out = ro*F --> F = F_in = F_out at each instant

    # Mass Balance on A
    # Ca_in*F - Ca*F = V*dCA/dt
    #
    dx_0 = (U[2] * U[0] - X[0] * U[0]) / V

    # Energy Balance
    # ro*cp*F*T_in - ro*cp*F*T + W*Lam = (V*ro*cp)*dT/dt
    #
    dx_1 = (ro * cp * U[0] * U[3] - ro * cp * U[0] * X[1] + U[1] * Lam) / (
        V * ro * cp
    )

    fx = np.append(dx_0, dx_1)

    return fx


# Build input sequences
U = np.zeros((m, npts))

# manipulated inputs as GBN
# Input Flow rate Fin = F = U[0]    [m^3/min]
prob_switch_1 = 0.05
F_min = 0.4
F_max = 0.6
Range_GBN_1 = [F_min, F_max]
[U[0, :], _, _] = GBN_seq(npts, prob_switch_1, Range=Range_GBN_1)
# Steam Flow rate W = U[1]          [kg/min]
prob_switch_2 = 0.05
W_min = 20
W_max = 40
Range_GBN_2 = [W_min, W_max]
[U[1, :], _, _] = GBN_seq(npts, prob_switch_2, Range=Range_GBN_2)

# disturbance inputs as RW (random-walk)

# Input Concentration Ca_in = U[2]  [kg salt/m^3 solution]
Ca_0 = 10.0  # initial condition
sigma_Ca = 0.01  # variation
U[2, :] = GBN_seq(npts, prob_switch_1, Range=[Ca_0 - 3*sigma_Ca, Ca_0 + 3*sigma_Ca])[0]
# Input Temperature T_in            [°C]
Tin_0 = 25.0  # initial condition
sigma_T = 0.01  # variation
U[3, :] = GBN_seq(npts, prob_switch_1, Range=[Tin_0 - 3*sigma_T, Tin_0 + 3*sigma_T])[0]


##### COLLECT DATA

# Output Initial conditions
Caout_0 = Ca_0
Tout_0 = (ro * cp * U[0, 0] * Tin_0 + U[1, 0] * Lam) / (ro * cp * U[0, 0])
Xo1 = Caout_0 * np.ones((1, npts))
Xo2 = Tout_0 * np.ones((1, npts))
X = np.vstack((Xo1, Xo2))

# Run Simulation
for j in range(npts - 1):
    # Explicit Runge-Kutta 4 (TC dynamics is integrateed by hand)
    Mx = 5  # Number of elements in each time step
    dt = ts / Mx  # integration step
    # Output & Input
    X0k = X[:, j]
    Uk = U[:, j]
    # Integrate the model
    for i in range(Mx):
        k1 = Fdyn(X0k, Uk)
        k2 = Fdyn(X0k + dt / 2.0 * k1, Uk)
        k3 = Fdyn(X0k + dt / 2.0 * k2, Uk)
        k4 = Fdyn(X0k + dt * k3, Uk)
        Xk_1 = X0k + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    X[:, j + 1] = Xk_1

# Add noise (with assigned variances)
var = [0.001, 0.001]
noise = white_noise_var(npts, var)

# Build Output
Y = X + noise


#### IDENTIFICATION STAGE (Linear Models) using new API

# Orders
na_ords = [2, 2]
nb_ords = [[1, 1, 1, 1], [1, 1, 1, 1]]
nc_ords = [1, 1]
nd_ords = [1, 1]
nf_ords = [2, 2]
theta = [[1, 1, 1, 1], [1, 1, 1, 1]]
# Number of iterations
n_iter = 300

# Create IDData object - need to convert from numpy arrays to DataFrame
time_index = pd.date_range("2023-01-01", periods=npts, freq=f"{int(ts*1000)}ms")
data_dict = {}
for i in range(4):
    data_dict[f"u{i+1}"] = U[i, :]
for i in range(2):
    data_dict[f"y{i+1}"] = Y[i, :]

data_df = pd.DataFrame(data_dict, index=time_index)
inputs = [f"u{i+1}" for i in range(4)]
outputs = [f"y{i+1}" for i in range(2)]
data = IDData(data=data_df, inputs=inputs, outputs=outputs, tsample=ts)

# IN-OUT Models: ARX - ARMAX - OE - BJ - GEN

sys_id_arx = SystemIdentification()
Id_ARX = sys_id_arx.identify(y=data.get_output_array(), u=data.get_input_array(), 
                             id_method="ARX", na=na_ords, nb=nb_ords, theta=theta)

sys_id_armax = SystemIdentification()
Id_ARMAX = sys_id_armax.identify(y=data.get_output_array(), u=data.get_input_array(),
                                 id_method="ARMAX", na=na_ords, nb=nb_ords, nc=nc_ords, theta=theta,
                                 max_iterations=n_iter)

sys_id_oe = SystemIdentification()
Id_OE = sys_id_oe.identify(y=data.get_output_array(), u=data.get_input_array(),
                           id_method="OE", nb=nb_ords, nf=nf_ords, theta=theta, 
                           max_iterations=n_iter)

sys_id_bj = SystemIdentification()
Id_BJ = sys_id_bj.identify(y=data.get_output_array(), u=data.get_input_array(),
                          id_method="BJ", nb=nb_ords, nc=nc_ords, nd=nd_ords, nf=nf_ords, theta=theta,
                          max_iterations=n_iter, stability_constraint=True)

sys_id_gen = SystemIdentification()
Id_GEN = sys_id_gen.identify(y=data.get_output_array(), u=data.get_input_array(),
                             id_method="GEN", na=na_ords, nb=nb_ords, nc=nc_ords, nd=nd_ords, nf=nf_ords, theta=theta,
                             max_iterations=n_iter, stability_constraint=True, stability_margin=0.98)

# SS - mimo
# choose method
method = "N4SID"
SS_ord = 2
sys_id_ss = SystemIdentification()
Id_SS = sys_id_ss.identify(y=data.get_output_array(), u=data.get_input_array(),
                          id_method=method, ss_fixed_order=SS_ord)

# GETTING RESULTS (Y_id)
# IN-OUT
_, Y_arx = Id_ARX.simulate(U)
_, Y_armax = Id_ARMAX.simulate(U)
_, Y_oe = Id_OE.simulate(U)
_, Y_bj = Id_BJ.simulate(U)
_, Y_gen = Id_GEN.simulate(U)
# SS
x_ss, Y_ss = Id_SS.simulate(U)


##### PLOTS

# Input
plt.close("all")
plt.figure(1)

str_input = [
    "F [m$^3$/min]",
    "W [kg/min]",
    "Ca$_{in}$ [kg/m$^3$]",
    "T$_{in}$ [$^o$C]",
]
for i in range(m):
    plt.subplot(m, 1, i + 1)
    plt.plot(Time, U[i, :])
    plt.ylabel("Input " + str(i + 1))
    plt.ylabel(str_input[i])
    plt.grid()
    plt.xlabel("Time")
    plt.axis([0, tfin, 0.95 * np.amin(U[i, :]), 1.05 * np.amax(U[i, :])])  # type: ignore
    if i == 0:
        plt.title("identification")

# Output
plt.figure(2)
str_output = ["Ca [kg/m$^3$]", "T [$^o$C]"]
for i in range(p):
    plt.subplot(p, 1, i + 1)
    plt.plot(Time, Y[i, :])

    # Plot model outputs - handle single-output models (only plot first output)
    if i < Y_arx.shape[0]:
        plt.plot(Time, Y_arx[i, :])
    if i < Y_armax.shape[0]:
        plt.plot(Time, Y_armax[i, :])
    if i < Y_oe.shape[0]:
        plt.plot(Time, Y_oe[i, :])
    if i < Y_bj.shape[0]:
        plt.plot(Time, Y_bj[i, :])
    if i < Y_gen.shape[0]:
        plt.plot(Time, Y_gen[i, :])
    if i < Y_ss.shape[0]:
        plt.plot(Time, Y_ss[i, :])

    plt.ylabel("Output " + str(i + 1))
    plt.ylabel(str_output[i])
    plt.legend(["Data", "ARX", "ARMAX", "OE", "BJ", "GEN", "SS"])
    plt.grid()
    plt.xlabel("Time")
    if i == 0:
        plt.title("identification")


#### VALIDATION STAGE

# Build new input sequences
U_val = np.zeros((m, npts))
# U_val = U.copy()

# manipulated inputs as GBN
# Input Flow rate Fin = F = U[0]    [m^3/min]
prob_switch_1 = 0.05
F_min = 0.4
F_max = 0.6
Range_GBN_1 = [F_min, F_max]
[U_val[0, :], _, _] = GBN_seq(npts, prob_switch_1, Range=Range_GBN_1)
# Steam Flow rate W = U[1]          [kg/min]
prob_switch_2 = 0.05
W_min = 20
W_max = 40
Range_GBN_2 = [W_min, W_max]
[U_val[1, :], _, _] = GBN_seq(npts, prob_switch_2, Range=Range_GBN_2)

# disturbance inputs as RW (random-walk)
# Input Concentration Ca_in = U[2]  [kg salt/m^3 solution]
Ca_0 = 10.0  # initial condition
sigma_Ca = 0.02  # variation
U_val[2, :] = GBN_seq(npts, prob_switch_1, Range=[Ca_0 - 3*sigma_Ca, Ca_0 + 3*sigma_Ca])[0]
# Input Temperature T_in            [°C]
Tin_0 = 25.0  # initial condition
sigma_T = 0.1  # variation
U_val[3, :] = GBN_seq(npts, prob_switch_1, Range=[Tin_0 - 3*sigma_T, Tin_0 + 3*sigma_T])[0]

#### COLLECT DATA

# Output Initial conditions
Caout_0 = Ca_0
Tout_0 = (ro * cp * U[0, 0] * Tin_0 + U[1, 0] * Lam) / (ro * cp * U[0, 0])
Xo1 = Caout_0 * np.ones((1, npts))
Xo2 = Tout_0 * np.ones((1, npts))
X_val = np.vstack((Xo1, Xo2))

# Run Simulation
for j in range(npts - 1):
    # Explicit Runge-Kutta 4 (TC dynamics is integrateed by hand)
    Mx = 5  # Number of elements in each time step
    dt = ts / Mx  # integration step
    # Output & Input
    X0k = X_val[:, j]
    Uk = U_val[:, j]
    # Integrate the model
    for i in range(Mx):
        k1 = Fdyn(X0k, Uk)
        k2 = Fdyn(X0k + dt / 2.0 * k1, Uk)
        k3 = Fdyn(X0k + dt / 2.0 * k2, Uk)
        k4 = Fdyn(X0k + dt * k3, Uk)
        Xk_1 = X0k + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    X_val[:, j + 1] = Xk_1

# Add noise (with assigned variances)
var = [0.01, 0.05]
noise_val = white_noise_var(npts, var)

# Build Output
Y_val = X_val + noise_val


# MODEL VALIDATION

# IN-OUT Models: ARX - ARMAX - OE - BJ
_, Yv_arx = Id_ARX.simulate(U_val)
_, Yv_armax = Id_ARMAX.simulate(U_val)
_, Yv_oe = Id_OE.simulate(U_val)
_, Yv_bj = Id_BJ.simulate(U_val)
_, Yv_gen = Id_GEN.simulate(U_val)
# SS
x_ss_val, Yv_ss = Id_SS.simulate(U_val)


##### PLOTS

# Input
plt.figure(3)
str_input = [
    "F [m$^3$/min]",
    "W [kg/min]",
    "Ca$_{in}$ [kg/m$^3$]",
    "T$_{in}$ [$^o$C]",
]
for i in range(m):
    plt.subplot(m, 1, i + 1)
    plt.plot(Time, U_val[i, :])
    # plt.ylabel("Input " + str(i+1))
    plt.ylabel(str_input[i])
    plt.grid()
    plt.xlabel("Time")
    plt.axis(
        [0, tfin, 0.95 * np.amin(U_val[i, :]), 1.05 * np.amax(U_val[i, :])]  # type: ignore
    )
    if i == 0:
        plt.title("validation")

# Output
plt.figure(4)
str_output = ["Ca [kg/m$^3$]", "T [$^o$C]"]
for i in range(p):
    plt.subplot(p, 1, i + 1)
    plt.plot(Time, Y_val[i, :])

    # Plot validation outputs - handle single-output models (only plot first output)
    if i < Yv_arx.shape[0]:
        plt.plot(Time, Yv_arx[i, :])
    if i < Yv_armax.shape[0]:
        plt.plot(Time, Yv_armax[i, :])
    if i < Yv_oe.shape[0]:
        plt.plot(Time, Yv_oe[i, :])
    if i < Yv_bj.shape[0]:
        plt.plot(Time, Yv_bj[i, :])
    if i < Yv_gen.shape[0]:
        plt.plot(Time, Yv_gen[i, :])
    if i < Yv_ss.shape[0]:
        plt.plot(Time, Yv_ss[i, :])

    # plt.ylabel("Output " + str(i+1))
    plt.ylabel(str_output[i])
    plt.legend(["Data", "ARX", "ARMAX", "OE", "BJ", "GEN", "SS"])
    # plt.legend(['Data','ARX','ARMAX','GEN','SS'])
    # plt.legend(['Data','ARMAX'])
    plt.grid()
    plt.xlabel("Time")
    if i == 0:
        plt.title("validation")

plt.show()
