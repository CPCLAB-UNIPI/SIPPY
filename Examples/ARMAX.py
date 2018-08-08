 # -*- coding: utf-8 -*-
"""
ARMAX Example

@author: Giuseppe Armenise
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
import control.matlab as cnt
#tsampling
ts=1.

# time 
tfin = 400
npts = int(old_div(tfin,ts)) + 1
Time = np.linspace(0, tfin, npts)
# input sequence
Usim =fset.PRBS_seq(npts,0.08)

#Defining the system
NUM_H=[1.,0.3,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]#nc=1
DEN=[1.,-2.21,1.7494,-0.584256,0.0684029,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]#na=4
NUM=[1.,-2.07,1.3146]#nb=3
nc=1
na=4
nb=3
theta=11

#Transfer functions
g_sample=cnt.tf(NUM,DEN,ts)
h_sample=cnt.tf(NUM_H,DEN,ts)
#Input responses
Y1, Time, Xsim = cnt.lsim(g_sample, Usim, Time) 
e_t=fset.white_noise_var(Y1[0].size,[0.005])
e_t=e_t[0]
Y2, Time, Xsim = cnt.lsim(h_sample, e_t, Time) 
Ytot=Y1+Y2


#System identification from collected data
Id_sys=system_identification(Ytot,Usim,'ARMAX',IC='BIC',na_ord=[2,5],\
                             nb_ord=[1,5],nc_ord=[0,2],delays=[10,13],\
                             ARMAX_max_iterations=300)

#Output of the identified system
Y_id1, Time, Xsim = cnt.lsim(Id_sys.G, Usim, Time)
Y_hid1, Time, Xsim = cnt.lsim(Id_sys.H, e_t, Time)
Y_idTot=Y_id1+Y_hid1

#Validation of the identified system
U_valid = fset.PRBS_seq(npts,0.07,[0.5,1.5])

e_valid=fset.white_noise_var(U_valid.size,[0.01])
e_valid=e_valid[0]

#System responses
Yvalid1, Time, Xsim = cnt.lsim(g_sample, U_valid, Time)
Yvalid2, Time, Xsim = cnt.lsim(h_sample, e_valid, Time)
Ytotvalid=Yvalid1+Yvalid2

#Identified system responses
Yidvalid1, Time, Xsim = cnt.lsim(Id_sys.G, U_valid, Time)
Yidvalid2, Time, Xsim = cnt.lsim(Id_sys.H, e_valid, Time)
Yidtotvalid=Yidvalid1+Yidvalid2

import os
if 'DISPLAY' in os.environ:
	import matplotlib.pyplot as plt

	plt.close('all')
	plt.figure(0)
	plt.plot(Time,Ytot[0])
	plt.plot(Time,Y_idTot[0])
	plt.grid()
	plt.xlabel("Time")
	plt.ylabel("y_tot")
	plt.title("Gu+He (identification data)")
	plt.legend(['Original system','Identified system'])

	plt.figure(1)
	plt.plot(Time,Y1[0])
	plt.plot(Time,Y_id1[0])
	plt.ylabel("y_out")
	plt.grid()
	plt.xlabel("Time")
	plt.title("Gu (identification data)")
	plt.legend(['Original system','Identified system'])

	##    #
	plt.figure(2)
	plt.plot(Time,Y2[0])
	plt.plot(Time,Y_hid1[0])
	plt.ylabel("y_err")
	plt.grid()
	plt.xlabel("Time")
	plt.title("He (identification data)")
	plt.legend(['Original system','Identified system'])


	plt.figure(3)
	plt.plot(Time,Usim)
	plt.ylabel("Input PRBS")
	plt.xlabel("Time")
	plt.title("Input, identification data (Switch probability=0.08)")
	plt.grid()
	#    
	#    
	plt.figure(4)
	plt.plot(Time,Ytotvalid[0])
	plt.plot(Time,Yidtotvalid[0])
	plt.xlabel("Time")
	plt.ylabel("y_tot")
	plt.title("Gu+He (Validation)")
	plt.legend(['Original system','Identified system'])
	plt.grid()


	plt.figure(5)
	plt.plot(Time,Yvalid1[0])
	plt.plot(Time,Yidvalid1[0])
	plt.grid()
	plt.xlabel("Time")
	plt.ylabel("y_out")
	plt.title("Gu (Validation)")
	plt.legend(['Original system','Identified system'])

	plt.figure(6)
	plt.plot(Time,Yvalid2[0])
	plt.plot(Time,Yidvalid2[0])
	plt.grid()
	plt.xlabel("Time")
	plt.ylabel("y_err")
	plt.title("He (Validation)")
	plt.legend(['Original system','Identified system'])


	plt.figure(7)
	plt.plot(Time,U_valid)
	plt.ylabel("Input PRBS")
	plt.xlabel("Time")
	plt.title("Input, validation data (Switch probability=0.07)")
	plt.grid()

	plt.show()
