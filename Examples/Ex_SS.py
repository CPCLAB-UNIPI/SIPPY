# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 2018

@author: Giuseppe Armenise, revised by RBdC

In this test, no error occurs.
Using method='N4SID','MOESP' or 'CVA', if the message
"Kalman filter cannot be calculated" is shown, it means
that the package slycot is not well-installed.

"""
# Checking path to access other files
try:
    from sippy.identification import SystemIdentification, IDData
except ImportError:
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(os.getcwd()), 'src'))
    from sippy.identification import SystemIdentification, IDData

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sippy.utils.signal_utils import GBN_seq, white_noise_var
from sippy.utils.simulation_utils import simulate_ss_system


# Create compatibility aliases
class fset:
    @staticmethod
    def GBN_seq(*args, **kwargs):
        return GBN_seq(*args, **kwargs)

    @staticmethod
    def white_noise_var(*args, **kwargs):
        return white_noise_var(*args, **kwargs)

class fsetSIM:
    @staticmethod
    def SS_lsim_process_form(*args, **kwargs):
        return simulate_ss_system(*args, **kwargs)


# Example to test SS-methods

# sample time
ts = 1.0

# SISO SS system (n = 2)
A = np.array([[0.89, 0.], [0., 0.45]])
B = np.array([[0.3], [2.5]])
C = np.array([[0.7, 1.]])
D = np.array([[0.0]])

tfin = 500
npts = int(tfin / ts) + 1
Time = np.linspace(0, tfin, npts)

# Input sequence
U = np.zeros((1, npts))
[U[0],_,_] = fset.GBN_seq(npts, 0.05)

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

##System identification using new API
# Create IDData object
time_index = pd.date_range("2023-01-01", periods=npts, freq=f"{int(ts*1000)}ms")
data_df = pd.DataFrame({"u": U[0], "y": y_tot[0]}, index=time_index)
data = IDData(data=data_df, inputs=["u"], outputs=["y"], tsample=ts)

METHOD = ['N4SID', 'CVA', 'MOESP']
lege = ['System']
for i in range(len(METHOD)):
    method = METHOD[i]
    try:
        # Create a new SystemIdentification instance for each method
        identifier = SystemIdentification()
        sys_id = identifier.identify(y=y_tot, u=U, id_method=method, ss_fixed_order=2)
        xid, yid = simulate_ss_system(sys_id.A, sys_id.B, sys_id.C, sys_id.D, U, sys_id.x0)
        plt.plot(Time, yid[0])
        lege.append(method)
    except Exception as e:
        print(f"Method {method} failed: {e}")
        lege.append(f"{method} (failed)")

plt.legend(lege)
plt.show()
