"""
Created

@author: Giuseppe Armenise
example armax mimo
case 3 outputs x 4 inputs

"""

# Checking path to access other files
import control.matlab as cnt
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
        cnt.lsim(H, err_inputH[i, :], Time)[0] for i, H in enumerate(H_samples)
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
    [1.0, 0.85, 0.32, 0.0, 0.0, 0.0],
    [1.0, 0.4, 0.05, 0.0, 0.0],
    [1.0, 0.7, 0.485, 0.22, 0.0],
]

na = [3, 1, 2]
nb = [[2, 1, 3, 2], [3, 2, 1, 1], [1, 2, 1, 2]]
th = [[1, 2, 2, 1], [1, 2, 0, 0], [0, 1, 0, 2]]
nc = [2, 2, 3]

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
Usim = generate_inputs(
    npts, [[-0.33, 0.1], [-1.0, 1.0], [2.3, 5.7], [8.0, 11.5]]
)

# Adding noise
err_outputH = add_noise(npts, var_list, H_samples)

# OUTPUTS
Ytot = compute_outputs(g_samples, Usim, Time)
for i in range(3):
    Ytot[i, :] += err_outputH[i]

# identification parameters
ordersna = na
ordersnb = nb
ordersnc = nc
theta_list = th

# IDENTIFICATION STAGE
# TESTING ARMAX models
identification_params = {
    "ARMAX-I": {
        "ARMAX_orders": [ordersna, ordersnb, ordersnc, theta_list],
        "max_iter": 20,
        "centering": "MeanVal",
    },
    "ARMAX-O": {
        "ARMAX_orders": [ordersna, ordersnb, ordersnc, theta_list],
        "ARMAX_mod": "OPT",
        "max_iter": 20,
        "centering": None,
    },
    "ARMAX-R": {
        "ARMAX_orders": [ordersna, ordersnb, ordersnc, theta_list],
        "ARMAX_mod": "RLLS",
        "max_iter": 20,
        "centering": "InitVal",
    },
}

syss = []
for method, params in identification_params.items():
    sys_id = system_identification(Ytot, Usim, "ARMAX", **params)
    syss.append(sys_id)

Youts = [getattr(sys, "Yid") for sys in syss]

# plots
fig = plot_comparison(
    Time,
    Usim,
    ["Input 1 - GBN", "Input 2 - GBN", "Input 3 - GBN", "Input 4 - GBN"],
    title="Input (Switch probability=0.03) (identification data)",
)
fig.savefig(output_dir + "/input_identification.png")

fig = plot_comparison(
    Time,
    [Ytot] + Youts,
    ylabels=[f"$y_{i}$" for i in range(3)],
    legend=["System", "ARMAX-I", "ARMAX-O", "ARMAX-R"],
    title="Output (identification data)",
)
fig.savefig(output_dir + "/output_identification.png")

# VALIDATION STAGE

# time
tfin = 400
npts = int(tfin // ts) + 1
Time = np.linspace(0, tfin, npts)

# (NEW) INPUTS
U_valid = generate_inputs(
    npts, [[0.33, 0.7], [-2.0, -1.0], [1.3, 2.7], [1.0, 5.2]]
)

# Adding noise
err_outputH = add_noise(npts, var_list, H_samples)

# OUTPUTS
Ytot_v = compute_outputs(g_samples, U_valid, Time)
for i in range(3):
    Ytot_v[i, :] += err_outputH[i]

# ## Compute time responses for identified systems with new inputs
Yv_armaxi = fset.validation(
    syss[0], U_valid, Ytot_v, Time, centering="MeanVal"
)
Yv_armaxo = fset.validation(syss[1], U_valid, Ytot_v, Time)
Yv_armaxr = fset.validation(
    syss[2], U_valid, Ytot_v, Time, centering="InitVal"
)

# plots
fig = plot_comparison(
    Time,
    U_valid,
    ["Input 1 - GBN", "Input 2 - GBN", "Input 3 - GBN", "Input 4 - GBN"],
    title="Input (Switch probability=0.03) (validation data)",
)
fig.savefig(output_dir + "/input_validation.png")

fig = plot_comparison(
    Time,
    [Ytot_v, Yv_armaxi, Yv_armaxo, Yv_armaxr],
    ylabels=[f"$y_{i}$" for i in range(3)],
    legend=["System", "ARMAX-I", "ARMAX-O", "ARMAX-R"],
    title="Output (validation data)",
)
fig.savefig(output_dir + "/output_validation.png")
