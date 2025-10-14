"""Created on Fri Jan 19 2018

@author: Giuseppe Armenise, revised by RBdC

In this test, no error occurs.
Using method='N4SID','MOESP' or 'CVA', if the message
"Kalman filter cannot be calculated" is shown, it means
that the package slycot is not well-installed.

"""

# Checking path to access other files
try:
    from sippy_unipi import system_identification
except ImportError:
    import os
    import sys

    sys.path.append(os.pardir)
    from sippy_unipi import system_identification

import matplotlib.pyplot as plt
import numpy as np

from sippy_unipi import functionset as fset
from sippy_unipi import functionsetSIM as fsetSIM

# Example to test SS-methods

# sample time
ts = 1.0

# SISO SS system (n = 2)
A = np.array([[0.89, 0.0], [0.0, 0.45]])
B = np.array([[0.3], [2.5]])
C = np.array([[0.7, 1.0]])
D = np.array([[0.0]])

tfin = 500
npts = int(tfin / ts) + 1
Time = np.linspace(0, tfin, npts)

# Input sequence
U = np.zeros((1, npts))
[U[0], _, _] = fset.GBN_seq(npts, 0.05)

##Output
x, yout = fsetSIM.SS_lsim_process_form(A, B, C, D, U)

# measurement noise
noise = fset.white_noise_var(npts, [0.15])

# Output with noise
y_tot = yout + noise

#
plt.close("all")
plt.figure(0)
plt.plot(Time, U[0])
plt.ylabel("input")
plt.grid()
plt.xlabel("Time")
#
plt.figure(1)
plt.plot(Time, y_tot[0])
plt.ylabel("y_tot")
plt.grid()
plt.xlabel("Time")
plt.title("Ytot")

##System identification
METHOD = ["N4SID", "CVA", "MOESP", "PARSIM-S", "PARSIM-P", "PARSIM-K"]
lege = ["System"]
for i in range(len(METHOD)):
    method = METHOD[i]
    sys_id = system_identification(y_tot, U, method, SS_fixed_order=2)
    xid, yid = fsetSIM.SS_lsim_process_form(
        sys_id.A, sys_id.B, sys_id.C, sys_id.D, U, sys_id.x0
    )
    #
    plt.plot(Time, yid[0])
    plt.show(block=False)
    lege.append(method)
plt.legend(lege)
