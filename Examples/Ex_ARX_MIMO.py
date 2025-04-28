"""
Created

@author: Giuseppe Armenise, revised by RBdC
example armax mimo
case 3 outputs x 4 inputs

"""

# Checking path to access other files
import matplotlib.pyplot as plt
import numpy as np
from utils import create_output_dir, plot_comparison

from sippy_unipi import system_identification
from sippy_unipi.datasets import load_sample_mimo

seed = 0
np.random.seed(0)
output_dir = create_output_dir(__file__)


na = [3, 1, 2]
nb = [[2, 1, 3, 2], [3, 2, 1, 1], [1, 2, 1, 2]]
th = [[1, 2, 2, 1], [1, 2, 0, 0], [0, 1, 0, 2]]

n_samples = 401
ts = 1.0
time, Ysim, Usim, g_sys, Yerr, Uerr, h_sys, Ytot, Utot = load_sample_mimo(
    n_samples, ts, seed=seed
)

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
    time,
    Usim,
    ["Input 1 - GBN", "Input 2 - GBN", "Input 3 - GBN", "Input 4 - GBN"],
    title="Input (Switch probability=0.03) (validation data)",
)
fig.savefig(output_dir + "/input_validation.png")

fig = plot_comparison(
    time,
    [Ytot, Yout_ARX, Yout_FIR],
    ylabels=[f"$y_{i}$" for i in range(3)],
    legend=["System", "ARX", "FIR"],
    title="Output (validation data)",
)
fig.savefig(output_dir + "/output_validation.png")

plt.close("all")
