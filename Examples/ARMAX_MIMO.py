# -*- coding: utf-8 -*-
"""
Created

@author: Giuseppe Armenise
example armax mimo
case 3 outputs x 4 inputs

"""
from sys_identification import *
import control as cnt
from lib import functionset as fset
import matplotlib.pyplot as plt
#generating transfer functions in z-transf.
var_list=[50.,100.,1.]
ts=1.

NUM11=[4,3.3,0.,0.]
NUM12=[10,0.,0.]
NUM13=[7.0,5.5,2.2]
NUM14=[-0.9,-0.11,0.,0.]
DEN1=[1.,-0.3,-0.25,-0.021,0.,0.]#
H1=[1.,0.85,0.32,0.,0.,0.]
na1=3
nb11=2
nb12=1
nb13=3
nb14=2
th11=1
th12=2
th13=2
th14=1
nc1=2
#
DEN2=[1.,-0.4,0.,0.,0.]
NUM21=[-85,-57.5,-27.7]
NUM22=[71,12.3]
NUM23=[-0.1,0.,0.,0.]
NUM24=[0.994,0.,0.,0.]
H2=[1.,0.4,0.05,0.,0.]

na2=1
nb21=3
nb22=2
nb23=1
nb24=1

th21=1
th22=2
th23=0
th24=0
nc2=2

DEN3=[1.,-0.1,-0.3,0.,0.]
NUM31=[0.2,0.,0.,0.]
NUM32=[0.821,0.432,0.]
NUM33=[0.1,0.,0.,0.]
NUM34=[0.891,0.223]
H3=[1.,0.7,0.485,0.22,0.]

na3=2
nb31=1
nb32=2
nb33=1
nb34=2
th31=0
th32=1
th33=0
th34=2
nc3=3

# transfer function G, H
g_sample11=cnt.tf(NUM11,DEN1,ts)
g_sample12=cnt.tf(NUM12,DEN1,ts)
g_sample13=cnt.tf(NUM13,DEN1,ts)
g_sample14=cnt.tf(NUM14,DEN1,ts)

g_sample22=cnt.tf(NUM22,DEN2,ts)
g_sample21=cnt.tf(NUM21,DEN2,ts)
g_sample23=cnt.tf(NUM23,DEN2,ts)
g_sample24=cnt.tf(NUM24,DEN2,ts)

g_sample31=cnt.tf(NUM31,DEN3,ts)
g_sample32=cnt.tf(NUM32,DEN3,ts)
g_sample33=cnt.tf(NUM33,DEN3,ts)
g_sample34=cnt.tf(NUM34,DEN3,ts)

H_sample1=cnt.tf(H1,DEN1,ts)
H_sample2=cnt.tf(H2,DEN2,ts)
H_sample3=cnt.tf(H3,DEN3,ts)


# 
tfin = 400
npts = int(tfin/ts) + 1
Time = np.linspace(0, tfin, npts)
#                                   #INPUT#
Usim=np.zeros((4,npts))
Usim_noise=np.zeros((4,npts))
Usim[0,:] = fset.PRBS_seq(npts,0.03,[-0.33,0.1])
Usim[1,:] = fset.PRBS_seq(npts,0.03)
Usim[2,:] = fset.PRBS_seq(npts,0.03,[2.3,5.7])
Usim[3,:] = fset.PRBS_seq(npts,0.03,[8.,11.5])

#Adding noise
err_inputH=np.zeros((4,npts))

err_inputH=fset.white_noise_var(npts,var_list)

err_outputH=np.ones((3,npts))
err_outputH[0,:], Time, Xsim = cnt.lsim(H_sample1, err_inputH[0,:], Time)
err_outputH[1,:], Time, Xsim = cnt.lsim(H_sample2, err_inputH[1,:], Time)
err_outputH[2,:], Time, Xsim = cnt.lsim(H_sample3, err_inputH[2,:], Time)

#OUTPUTS
Yout=np.zeros((3,npts))

Yout11, Time, Xsim = cnt.lsim(g_sample11, Usim[0,:], Time)
Yout12, Time, Xsim = cnt.lsim(g_sample12, Usim[1,:], Time)
Yout13,Time,Xsim = cnt.lsim(g_sample13, Usim[2,:], Time)
Yout14,Time,Xsim = cnt.lsim(g_sample14, Usim[3,:], Time)
Yout21, Time, Xsim = cnt.lsim(g_sample21, Usim[0,:], Time)
Yout22, Time, Xsim = cnt.lsim(g_sample22, Usim[1,:], Time)
Yout23,Time,Xsim = cnt.lsim(g_sample23, Usim[2,:], Time)
Yout24,Time,Xsim = cnt.lsim(g_sample24, Usim[3,:], Time)
Yout31, Time, Xsim = cnt.lsim(g_sample31, Usim[0,:], Time)
Yout32, Time, Xsim = cnt.lsim(g_sample32, Usim[1,:], Time)
Yout33,Time,Xsim = cnt.lsim(g_sample33, Usim[2,:], Time)
Yout34,Time,Xsim = cnt.lsim(g_sample34, Usim[3,:], Time)


Ytot1=Yout11+Yout12+Yout13+Yout14
Ytot2=Yout21+Yout22+Yout23+Yout24
Ytot3=Yout31+Yout32+Yout33+Yout34

Ytot=np.zeros((3,npts))

Ytot[0,:]=Ytot1+err_outputH[0,:]
Ytot[1,:]=Ytot2+err_outputH[1,:]
Ytot[2,:]=Ytot3+err_outputH[2,:]


##identification parameters
ordersna=[na1,na2,na3]
ordersnb=[[nb11,nb12,nb13,nb14],[nb21,nb22,nb23,nb24],[nb31,nb32,nb33,nb34]]
ordersnc=[nc1,nc2,nc3]
theta_list=[[th11,th12,th13,th14],[th21,th22,th23,th24],[th31,th32,th33,th34]]

#IDENTIFICATION
Id_sys=system_identification(Ytot,Usim,'ARMAX',ARMAX_orders=[ordersna,ordersnb,ordersnc,theta_list],ARMAX_max_iterations=20) #

#Generating transfer functions from NUMERATOR and DENOMINATOR
#you can use directly Id_sys.G for the G*u term, see the arx_MIMO example.
gid11=cnt.tf(Id_sys.NUMERATOR[0][0],Id_sys.DENOMINATOR[0][0],1.)
gid12=cnt.tf(Id_sys.NUMERATOR[0][1],Id_sys.DENOMINATOR[0][1],1.)
gid13=cnt.tf(Id_sys.NUMERATOR[0][2],Id_sys.DENOMINATOR[0][2],1.)
gid21=cnt.tf(Id_sys.NUMERATOR[1][0],Id_sys.DENOMINATOR[1][0],1.)
gid22=cnt.tf(Id_sys.NUMERATOR[1][1],Id_sys.DENOMINATOR[1][1],1.)
gid23=cnt.tf(Id_sys.NUMERATOR[1][2],Id_sys.DENOMINATOR[1][2],1.)
gid14=cnt.tf(Id_sys.NUMERATOR[0][3],Id_sys.DENOMINATOR[0][3],1.)
gid24=cnt.tf(Id_sys.NUMERATOR[1][3],Id_sys.DENOMINATOR[1][3],1.)
gid31=cnt.tf(Id_sys.NUMERATOR[2][0],Id_sys.DENOMINATOR[2][0],1.)
gid32=cnt.tf(Id_sys.NUMERATOR[2][1],Id_sys.DENOMINATOR[2][1],1.)
gid33=cnt.tf(Id_sys.NUMERATOR[2][2],Id_sys.DENOMINATOR[2][2],1.)
gid34=cnt.tf(Id_sys.NUMERATOR[2][3],Id_sys.DENOMINATOR[2][3],1.)

hid1=cnt.tf(Id_sys.NUMERATOR_H[0][0],Id_sys.DENOMINATOR_H[0][0],1.)
hid2=cnt.tf(Id_sys.NUMERATOR_H[1][0],Id_sys.DENOMINATOR_H[1][0],1.)
hid3=cnt.tf(Id_sys.NUMERATOR_H[2][0],Id_sys.DENOMINATOR_H[2][0],1.)

#output of the identified model
Yout_id11, Time, Xsim = cnt.lsim(gid11, Usim[0,:], Time)
Yout_id12, Time, Xsim = cnt.lsim(gid12, Usim[1,:], Time)
Yout_id13, Time, Xsim = cnt.lsim(gid13, Usim[2,:], Time)
Yout_id14, Time, Xsim = cnt.lsim(gid14, Usim[3,:], Time)
Yout_id21, Time, Xsim = cnt.lsim(gid21, Usim[0,:], Time)
Yout_id22, Time, Xsim = cnt.lsim(gid22, Usim[1,:], Time)
Yout_id23, Time, Xsim = cnt.lsim(gid23, Usim[2,:], Time)
Yout_id24, Time, Xsim = cnt.lsim(gid24, Usim[3,:], Time)
Yout_id31, Time, Xsim = cnt.lsim(gid31, Usim[0,:], Time)
Yout_id32, Time, Xsim = cnt.lsim(gid32, Usim[1,:], Time)
Yout_id33, Time, Xsim = cnt.lsim(gid33, Usim[2,:], Time)
Yout_id34, Time, Xsim = cnt.lsim(gid34, Usim[3,:], Time)
Yerr1, Time, Xsim = cnt.lsim(hid1, err_inputH[0,:], Time)
Yerr2, Time, Xsim = cnt.lsim(hid2, err_inputH[1,:], Time)
Yerr3, Time, Xsim = cnt.lsim(hid3, err_inputH[2,:], Time)

######plot
#    
plt.close('all')
plt.figure(0)
plt.subplot(4,1,1)
plt.plot(Time,Usim[0,:])
plt.grid()
plt.ylabel("Input 1 PRBS")
plt.xlabel("Time")
plt.title("Input (Switch probability=0.03)")

plt.subplot(4,1,2)
plt.plot(Time,Usim[1,:])
plt.grid()
plt.ylabel("Input 2 PRBS")
plt.xlabel("Time")

plt.subplot(4,1,3)
plt.plot(Time,Usim[2,:])
plt.ylabel("Input 3 PRBS")
plt.xlabel("Time")
plt.grid()

plt.subplot(4,1,4)
plt.plot(Time,Usim[3,:])
plt.ylabel("Input 4 PRBS")
plt.xlabel("Time")
plt.grid()

plt.figure(3)
plt.subplot(3,1,1)
plt.plot(Time,Ytot1[0,:])
plt.plot(Time,Yout_id11[0,:]+Yout_id12[0,:]+Yout_id13[0,:]+Yout_id14[0,:])
plt.ylabel("y_1,out")
plt.grid()
plt.xlabel("Time")
plt.title("Gu (identification data)")
plt.legend(['Original system','Identified system'])

plt.subplot(3,1,2)
plt.plot(Time,Ytot2[0,:])
plt.plot(Time,Yout_id21[0,:]+Yout_id22[0,:]+Yout_id23[0,:]+Yout_id24[0,:])
plt.ylabel("y_2,out")
plt.grid()
plt.xlabel("Time")
plt.legend(['Original system','Identified system'])

plt.subplot(3,1,3)
plt.plot(Time,Ytot3[0,:])
plt.plot(Time,Yout_id31[0,:]+Yout_id32[0,:]+Yout_id33[0,:]+Yout_id34[0,:])
plt.ylabel("y_3,out")
plt.grid()
plt.xlabel("Time")
plt.legend(['Original system','Identified system'])


plt.figure(5)
plt.subplot(3,1,1)#
plt.plot(Time,err_outputH[0,:])
plt.plot(Time,Yerr1[0,:])
plt.ylabel("y_1,err")
plt.grid()
plt.title("He (identification data)")
plt.xlabel("Time")
plt.legend(['Original system','Identified system'])


plt.subplot(3,1,2)
plt.plot(Time,err_outputH[1,:])
plt.plot(Time,Yerr2[0,:])
plt.ylabel("y_2,err")
plt.grid()
plt.xlabel("Time")
plt.legend(['Original system','Identified system'])

plt.subplot(3,1,3)
plt.plot(Time,err_outputH[2,:])
plt.plot(Time,Yerr3[0,:])
plt.ylabel("y_3,err")
plt.grid()
plt.xlabel("Time")
plt.legend(['Original system','Identified system'])


plt.figure(6)
plt.subplot(3,1,1)
plt.plot(Time,Ytot[0,:])
plt.plot(Time,Yout_id11[0,:]+Yout_id12[0,:]+Yout_id13[0,:]+Yout_id14[0,:]+Yerr1[0,:])
plt.xlabel("Time")
plt.ylabel("y_1,tot")
plt.title("Gu+He (identification data)")
plt.legend(['Original system','Identified system'])
plt.grid()

plt.subplot(3,1,2)
plt.plot(Time,Ytot[1,:])
plt.plot(Time,Yout_id21[0,:]+Yout_id22[0,:]+Yout_id23[0,:]+Yout_id24[0,:]+Yerr2[0,:])
plt.xlabel("Time")
plt.ylabel("y_2,tot")

plt.legend(['Original system','Identified system'])

plt.grid()

plt.subplot(3,1,3)
plt.plot(Time,Ytot[2,:])
plt.plot(Time,Yout_id31[0,:]+Yout_id32[0,:]+Yout_id33[0,:]+Yout_id34[0,:]+Yerr3[0,:])
plt.xlabel("Time")
plt.ylabel("y_3,tot")

plt.legend(['Original system','Identified system'])

plt.grid()

#################################################################################
###################################################################################
####################################VALIDATION############################################
##################################################################################
####################################################################################à


tfin = 400
npts = int(tfin/ts) + 1
Time = np.linspace(0, tfin, npts)
#                                   #INPUT#
Usim=np.zeros((4,npts))
Usim_noise=np.zeros((4,npts))
Usim[0,:] = fset.PRBS_seq(npts,0.03,[0.33,0.7])
Usim[1,:] = fset.PRBS_seq(npts,0.03,[-2.,-1.])
Usim[2,:] = fset.PRBS_seq(npts,0.03,[1.3,2.7])
Usim[3,:] = fset.PRBS_seq(npts,0.03,[1.,5.2])
err_inputH=np.zeros((4,npts))

err_inputH=fset.white_noise_var(npts,var_list)

err_outputH=np.ones((3,npts))
err_outputH[0,:], Time, Xsim = cnt.lsim(H_sample1, err_inputH[0,:], Time)
err_outputH[1,:], Time, Xsim = cnt.lsim(H_sample2, err_inputH[1,:], Time)
err_outputH[2,:], Time, Xsim = cnt.lsim(H_sample3, err_inputH[2,:], Time)


Yout=np.zeros((3,npts))

Yout11, Time, Xsim = cnt.lsim(g_sample11, Usim[0,:], Time)
Yout12, Time, Xsim = cnt.lsim(g_sample12, Usim[1,:], Time)
Yout13,Time,Xsim = cnt.lsim(g_sample13, Usim[2,:], Time)
Yout14,Time,Xsim = cnt.lsim(g_sample14, Usim[3,:], Time)
Yout21, Time, Xsim = cnt.lsim(g_sample21, Usim[0,:], Time)
Yout22, Time, Xsim = cnt.lsim(g_sample22, Usim[1,:], Time)
Yout23,Time,Xsim = cnt.lsim(g_sample23, Usim[2,:], Time)
Yout24,Time,Xsim = cnt.lsim(g_sample24, Usim[3,:], Time)
Yout31, Time, Xsim = cnt.lsim(g_sample31, Usim[0,:], Time)
Yout32, Time, Xsim = cnt.lsim(g_sample32, Usim[1,:], Time)
Yout33,Time,Xsim = cnt.lsim(g_sample33, Usim[2,:], Time)
Yout34,Time,Xsim = cnt.lsim(g_sample34, Usim[3,:], Time)


Ytot1=Yout11+Yout12+Yout13+Yout14
Ytot2=Yout21+Yout22+Yout23+Yout24
Ytot3=Yout31+Yout32+Yout33+Yout34

Ytot=np.zeros((3,npts))#

Ytot[0,:]=Ytot1+err_outputH[0,:]
Ytot[1,:]=Ytot2+err_outputH[1,:]
Ytot[2,:]=Ytot3+err_outputH[2,:]

###############################################################################plot


Yout_id11, Time, Xsim = cnt.lsim(gid11, Usim[0,:], Time)
Yout_id12, Time, Xsim = cnt.lsim(gid12, Usim[1,:], Time)
Yout_id13, Time, Xsim = cnt.lsim(gid13, Usim[2,:], Time)
Yout_id14, Time, Xsim = cnt.lsim(gid14, Usim[3,:], Time)
Yout_id21, Time, Xsim = cnt.lsim(gid21, Usim[0,:], Time)
Yout_id22, Time, Xsim = cnt.lsim(gid22, Usim[1,:], Time)
Yout_id23, Time, Xsim = cnt.lsim(gid23, Usim[2,:], Time)
Yout_id24, Time, Xsim = cnt.lsim(gid24, Usim[3,:], Time)
Yout_id31, Time, Xsim = cnt.lsim(gid31, Usim[0,:], Time)
Yout_id32, Time, Xsim = cnt.lsim(gid32, Usim[1,:], Time)
Yout_id33, Time, Xsim = cnt.lsim(gid33, Usim[2,:], Time)
Yout_id34, Time, Xsim = cnt.lsim(gid34, Usim[3,:], Time)
Yerr1, Time, Xsim = cnt.lsim(hid1, err_inputH[0,:], Time)
Yerr2, Time, Xsim = cnt.lsim(hid2, err_inputH[1,:], Time)
Yerr3, Time, Xsim = cnt.lsim(hid3, err_inputH[2,:], Time)


plt.figure(7)
plt.subplot(3,1,1)
plt.plot(Time,Ytot1[0,:])
plt.plot(Time,Yout_id11[0,:]+Yout_id12[0,:]+Yout_id13[0,:]+Yout_id14[0,:])
plt.ylabel("y_1,out")
plt.grid()
plt.xlabel("Time")
plt.title("Gu (validation data)")
plt.legend(['Original system','Identified system'])

plt.subplot(3,1,2)
plt.plot(Time,Ytot2[0,:])
plt.plot(Time,Yout_id21[0,:]+Yout_id22[0,:]+Yout_id23[0,:]+Yout_id24[0,:])
plt.ylabel("y_2,out")
plt.grid()
plt.xlabel("Time")
plt.legend(['Original system','Identified system'])

plt.subplot(3,1,3)
plt.plot(Time,Ytot3[0,:])
plt.plot(Time,Yout_id31[0,:]+Yout_id32[0,:]+Yout_id33[0,:]+Yout_id34[0,:])
plt.ylabel("y_3,out")
plt.grid()
plt.xlabel("Time")
plt.legend(['Original system','Identified system'])


plt.figure(8)
plt.subplot(3,1,1)#
plt.plot(Time,err_outputH[0,:])
plt.plot(Time,Yerr1[0,:])
plt.ylabel("y_1,err")
plt.grid()
plt.title("He (validation data)")
plt.xlabel("Time")
plt.legend(['Original system','Identified system'])


plt.subplot(3,1,2)
plt.plot(Time,err_outputH[1,:])
plt.plot(Time,Yerr2[0,:])
plt.ylabel("y_2,err")
plt.grid()
plt.xlabel("Time")
plt.legend(['Original system','Identified system'])

plt.subplot(3,1,3)
plt.plot(Time,err_outputH[2,:])
plt.plot(Time,Yerr3[0,:])
plt.ylabel("y_3,err")
plt.grid()
plt.xlabel("Time")
plt.legend(['Original system','Identified system'])


plt.figure(9)
plt.subplot(3,1,1)
plt.plot(Time,Ytot[0,:])
plt.plot(Time,Yout_id11[0,:]+Yout_id12[0,:]+Yout_id13[0,:]+Yout_id14[0,:]+Yerr1[0,:])
plt.xlabel("Time")
plt.ylabel("y_1,tot")
plt.title("Gu+He (validation data)")
plt.legend(['Original system','Identified system'])
plt.grid()

plt.subplot(3,1,2)
plt.plot(Time,Ytot[1,:])
plt.plot(Time,Yout_id21[0,:]+Yout_id22[0,:]+Yout_id23[0,:]+Yout_id24[0,:]+Yerr2[0,:])
plt.xlabel("Time")
plt.ylabel("y_2,tot")

plt.legend(['Original system','Identified system'])

plt.grid()

plt.subplot(3,1,3)
plt.plot(Time,Ytot[2,:])
plt.plot(Time,Yout_id31[0,:]+Yout_id32[0,:]+Yout_id33[0,:]+Yout_id34[0,:]+Yerr3[0,:])
plt.xlabel("Time")
plt.ylabel("y_3,tot")
plt.legend(['Original system','Identified system'])
plt.grid()


plt.figure(10)
plt.subplot(4,1,1)
plt.plot(Time,Usim[0,:])
plt.grid()
plt.ylabel("Input 1 PRBS")
plt.xlabel("Time")
plt.title("Input (Switch probability=0.03)")

plt.subplot(4,1,2)
plt.plot(Time,Usim[1,:])
plt.grid()
plt.ylabel("Input 2 PRBS")
plt.xlabel("Time")

plt.subplot(4,1,3)
plt.plot(Time,Usim[2,:])
plt.ylabel("Input 3 PRBS")
plt.xlabel("Time")
plt.grid()

plt.subplot(4,1,4)
plt.plot(Time,Usim[3,:])
plt.ylabel("Input 4 PRBS")
plt.xlabel("Time")
plt.grid()
