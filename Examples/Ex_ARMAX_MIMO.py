"""
Created

@author: Giuseppe Armenise
example armax mimo
case 3 outputs x 4 inputs

"""

# Checking path to access other files
import matplotlib.pyplot as plt
import numpy as np
from utils import create_output_dir, plot_comparison

from sippy_unipi import functionset as fset
from sippy_unipi import system_identification
from sippy_unipi.datasets import load_sample_mimo

seed = 0
np.random.seed(seed)
output_dir = create_output_dir(__file__)


na = [3, 1, 2]
nb = [[2, 1, 3, 2], [3, 2, 1, 1], [1, 2, 1, 2]]
th = [[1, 2, 2, 1], [1, 2, 0, 0], [0, 1, 0, 2]]
nc = [2, 2, 3]

n_samples = 401
ts = 1.0
time, Ysim, Usim, g_sys, Yerr, Uerr, h_sys, Ytot, Utot = load_sample_mimo(
    n_samples, ts, seed=seed
)

# identification parameters
ordersna = na
ordersnb = nb
ordersnc = nc
theta_list = th

# IDENTIFICATION STAGE
# TESTING ARMAX models
orders = [ordersna, ordersnb, ordersnc, theta_list]
identification_params = {
    "ARMAX-I": {
        "id_mode": "ILLS",
        "max_iter": 20,
        "centering": "MeanVal",
    },
    "ARMAX-O": {
        "id_mode": "OPT",
        "max_iter": 20,
        "centering": "MeanVal",
    },
    "ARMAX-R": {
        "id_mode": "RLLS",
        "max_iter": 20,
        "centering": "MeanVal",
    },
}

syss = []
for method, params in identification_params.items():
    sys_id = system_identification(Ytot, Usim, "ARMAX", *orders, **params)
    syss.append(sys_id)

Youts = [getattr(sys, "Yid") for sys in syss]

# plots
fig = plot_comparison(
    time,
    Usim,
    ["Input 1 - GBN", "Input 2 - GBN", "Input 3 - GBN", "Input 4 - GBN"],
    title="Input (Switch probability=0.03) (identification data)",
)
fig.savefig(output_dir + "/input_identification.png")

fig = plot_comparison(
    time,
    [Ytot] + Youts,
    ylabels=[f"$y_{i}$" for i in range(3)],
    legend=["System", "ARMAX-I", "ARMAX-O", "ARMAX-R"],
    title="Output (identification data)",
)
fig.savefig(output_dir + "/output_identification.png")

# VALIDATION STAGE
time, Ysim, Usim_v, g_sys, Yerr, Uerr, h_sys, Ytot_v, Utot_v = (
    load_sample_mimo(
        n_samples,
        ts,
        input_ranges=[(0.33, 0.7), (-2.0, -1.0), (1.3, 2.7), (1.0, 5.2)],
        seed=seed,
    )
)

# ## Compute time responses for identified systems with new inputs
Yv_armaxi = fset.validation(syss[0], Usim_v, Ytot_v, time, centering="MeanVal")
Yv_armaxo = fset.validation(syss[1], Usim_v, Ytot_v, time)
Yv_armaxr = fset.validation(syss[2], Usim_v, Ytot_v, time, centering="InitVal")

# plots
fig = plot_comparison(
    time,
    Usim_v,
    ["Input 1 - GBN", "Input 2 - GBN", "Input 3 - GBN", "Input 4 - GBN"],
    title="Input (Switch probability=0.03) (validation data)",
)
fig.savefig(output_dir + "/input_validation.png")

fig = plot_comparison(
    time,
    [Ytot_v, Yv_armaxi, Yv_armaxo, Yv_armaxr],
    ylabels=[f"$y_{i}$" for i in range(3)],
    legend=["System", "ARMAX-I", "ARMAX-O", "ARMAX-R"],
    title="Output (validation data)",
)
fig.savefig(output_dir + "/output_validation.png")
plt.close("all")
