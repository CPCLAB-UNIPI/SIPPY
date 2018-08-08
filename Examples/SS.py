# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 2018

@author: Giuseppe Armenise

In this test, no error occurs. 
Using method='N4SID','MOESP' or 'CVA', if the message
"Kalman filter cannot be calculated" is shown, it means
that the package slycot is not well-installed.

"""
from __future__ import division
from past.utils import old_div
#Checking path to access other files
try:
    from SIPPY import *
except ImportError:
    import sys, os
    sys.path.append(os.pardir)
    from SIPPY import *

import numpy as np
from SIPPY import functionset as fset
from SIPPY import functionsetSIM as fsetSIM

ts=1.0

A = np.array([[0.89, 0.],[0., 0.45]])
B = np.array([[0.3],[2.5]])
C = np.array([[0.7,1.]])
D = np.array([[0.0]])


tfin = 500
npts = int(old_div(tfin,ts)) + 1
Time = np.linspace(0, tfin, npts)

#Input sequence
U=np.zeros((1,npts))
U[0]=fset.PRBS_seq(npts,0.05)

##Output
x,yout = fsetSIM.SS_lsim_process_form(A,B,C,D,U)

#measurement noise
noise=fset.white_noise_var(npts,[0.15])

#Output with noise
y_tot=yout+noise


##System identification
method='N4SID'
sys_id=system_identification(y_tot,U,method,SS_fixed_order=2)
xid,yid=fsetSIM.SS_lsim_process_form(sys_id.A,sys_id.B,sys_id.C,sys_id.D,U,sys_id.x0)

import os
if 'DISPLAY' in os.environ:
	import matplotlib.pyplot as plt

	plt.close("all")
	plt.figure(1)
	plt.plot(Time,y_tot[0])
	plt.plot(Time,yid[0])
	plt.ylabel("y_tot")
	plt.grid()
	plt.xlabel("Time")
	plt.title("Ytot")
	plt.legend(['Original system','Identified system, '+method])

	plt.figure(2)
	plt.plot(Time,U[0])
	plt.ylabel("input")
	plt.grid()
	plt.xlabel("Time")

	plt.show()
