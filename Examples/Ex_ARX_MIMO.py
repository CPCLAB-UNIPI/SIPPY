"""
Created

@author: Giuseppe Armenise, revised by RBdC
example armax mimo
case 3 outputs x 4 inputs

"""

# Checking path to access other files
import control.matlab as cnt
import matplotlib.pyplot as plt
import numpy as np
from utils import create_output_dir, plot_comparison

from sippy import functionset as fset
from sippy import system_identification

np.random.seed(0)
output_dir = create_output_dir(__file__)


# Helper functions
def generate_inputs(npts, ranges):
    Usim = np.zeros((4, npts))
    for i, r in enumerate(ranges):
        Usim[i, :], _, _ = fset.GBN_seq(npts, 0.03, Range=r)
    return Usim


def add_noise(npts, var_list, H_samples):
    err_inputH = fset.white_noise_var(npts, var_list)
    err_outputH = [
        cnt.lsim(H, err_inputH[i, :], Time)[0]  # type: ignore
        for i, H in enumerate(H_samples)
    ]
    return err_outputH


def compute_outputs(g_samples, Usim, Time):
    Yout = np.zeros((3, npts))
    for i in range(3):
        for j in range(4):
            Yout[i, :] += cnt.lsim(g_samples[i][j], Usim[j, :], Time)[0]
    return Yout


# 4*3 MIMO system
# generating transfer functions in z-operator
var_list = [50.0, 100.0, 1.0]
ts = 1.0

NUM = [
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

DEN = [
    [1.0, -0.3, -0.25, -0.021, 0.0, 0.0],
    [1.0, -0.4, 0.0, 0.0, 0.0],
    [1.0, -0.1, -0.3, 0.0, 0.0],
]

H = [
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 0.0, 0.0],
]

na = [3, 1, 2]
nb = [[2, 1, 3, 2], [3, 2, 1, 1], [1, 2, 1, 2]]
th = [[1, 2, 2, 1], [1, 2, 0, 0], [0, 1, 0, 2]]

# SISO transfer functions (G and H)
g_samples = [
    [cnt.tf(NUM[i][j], DEN[i], ts) for j in range(4)] for i in range(3)
]
H_samples = [cnt.tf(H[i], DEN[i], ts) for i in range(3)]

# time
tfin = 400
npts = int(tfin // ts) + 1
Time = np.linspace(0, tfin, npts)

# INPUT
Usim = generate_inputs(npts, [[-0.33, 0.1], [-1, 1], [2.3, 5.7], [8.0, 11.5]])

# Adding noise
err_outputH = add_noise(npts, var_list, H_samples)

# OUTPUTS
Ytot = compute_outputs(g_samples, Usim, Time)
for i in range(3):
    Ytot[i, :] += err_outputH[i]

# identification parameters
ordersna = na
ordersnb = nb
theta_list = th

# IDENTIFICATION STAGE

# ARX
Id_ARX = system_identification(
    Ytot,
    Usim,
    "ARX",
    *(ordersna, ordersnb, theta_list),
    id_mode="LLS",
)

# FIR
Id_FIR = system_identification(
    Ytot, Usim, "FIR", *([0, 0, 0], ordersnb, theta_list), id_mode="LLS"
)

# output of the identified model
Yout_ARX = Id_ARX.Yid
Yout_FIR = Id_FIR.Yid

# plot

fig = plot_comparison(
    Time,
    Usim,
    ["Input 1 - GBN", "Input 2 - GBN", "Input 3 - GBN", "Input 4 - GBN"],
    title="Input (Switch probability=0.03) (validation data)",
)
fig.savefig(output_dir + "/input_validation.png")

fig = plot_comparison(
    Time,
    [Ytot, Yout_ARX, Yout_FIR],
    ylabels=[f"$y_{i}$" for i in range(3)],
    legend=["System", "ARX", "FIR"],
    title="Output (validation data)",
)
fig.savefig(output_dir + "/output_validation.png")

plt.close("all")
