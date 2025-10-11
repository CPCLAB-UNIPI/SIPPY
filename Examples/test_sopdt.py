
import matplotlib.pyplot as plt
import pandas as pd
from harold import step_response_plot
from scipy import signal

from sippy import *

# %
#load spteptest data from a TSV file
file = r'data\sopdt.csv'
idinput = pd.read_csv(file,index_col='Time', parse_dates=True, skiprows=[1,2])

#select Inputs and Outputs for the model identification case
inputs = ['U']
outputs = ['Y']

# Create FIR filter to detrend signal

tsample = pd.Timedelta(idinput.index[1] - idinput.index[0]).total_seconds() # data sampling time
tss = 300
tss_sec = tss * 60
mult_factor = 1
filt_tss = tss_sec * mult_factor
cutoff = 1/2/filt_tss
pass_zero= 'lowpass'
nyq_rate = tsample/2.0
width = 0.5/nyq_rate
ripple_db =65
N,beta =signal.kaiserord(ripple_db,width)
window = ('kaiser',beta)
coef = signal.firwin(numtaps=N, cutoff=cutoff, window=window, pass_zero=pass_zero, nyq = nyq_rate)
# plots.plot_freuency_response(coef)
trend = signal.filtfilt(coef, 1.0,idinput, axis=0)
idinput = idinput - trend

u = idinput[inputs].to_numpy()
y = idinput[outputs].to_numpy()
print('Output shape:', y.shape)
print('Input shape:',u.shape)

#specify model identification parameters, reffer the documentation for detais.
model = 'Precalciner.npz' #model file name
id_method='CVA'
IC = 'AIC' # None, AIC, AICc, BIC
TH = 120 # The length of time horizon used for regression
fix_ordr = 22 # Used if and only if IC = 'None'
SS_max_order  = 45
ss_order = [0, 45] # Used if IC = AIC, AICc or BIC
SS_threshold = 1
req_D = False
force_A_stable = False

id_result = system_identification(
    y=y,
    u=u,
    id_method=id_method,
    SS_fixed_order=fix_ordr,
    SS_max_order = SS_max_order,
    SS_orders=ss_order,
    IC=IC,
    SS_f=TH,
    SS_D_required=req_D,
    SS_A_stability=force_A_stable,
    tsample=tsample,
    SS_threshold=SS_threshold
    )

step_response_plot(id_result.G)
plt.show()
