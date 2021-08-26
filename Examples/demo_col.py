
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.getcwd())
from sysidbox.subspace import system_identification
from sysidbox.functionsetSIM import get_model_uncertainty, get_fir_coef, get_step_response
from harold import simulate_step_response, simulate_impulse_response, undiscretize, discretize
from detrend.detrending_filter import DetrendingFilter


# Load spteptest data from a TSV file
file = r'data\FRAC2.csv'
columns = ['Time', 'AI-2020', 'AI-2021', 'AI-2022', 'FIC-2100', 'FIC-2101', 'FIC-2102', 'FI-2005', 'FIC-2001', 'FIC-2002', 'FIC-2004', 'QI-2106', 'TIC-2003']
step_test_data = pd.read_csv(file,skiprows=[1,2,3], usecols=columns, index_col='Time', parse_dates=True)

#slice data for model identification case
slices = {
            "slice1":{"type":"bad", "isGlobal": False, "start":1040, "end":1135, "Description": "OPC bad for AI-2020","tags":['AI-2020']}, 
            "slice2":{"type":"interpolate", "isGlobal": False,"start":3845, "end":3855, "Description": "Suspicious value for  FI-2005", "tags":['FI-2005']}
        }


inputs = ['FIC-2001','FIC-2002', 'TIC-2003', 'FIC-2004','FI-2005']
outputs = ['FIC-2101', 'FIC-2102']

# Create FIR filter to detrend signal 
tags = inputs + outputs
tss = 120
filter_tss_mult_factor = 3
filter_type  = 'highpass' # Valid filters ['highpass', 'difference', 'doubledifference', 'zeromean', 'none']
d_filter = DetrendingFilter().get_filter(filter_type)
if filter_type == 'highpass':
    d_filter.apply_filter(step_test_data[tags], tss, filter_tss_mult_factor, slices)
else:
    d_filter.apply_filter(step_test_data[tags], slices)
    
idinput = d_filter.filterdata.data["output"]

# Resample datadet
idinput_resampled = idinput.resample('1min').mean()

# Convert dataframe to numpy array in the shape requied for SIPPY
u = idinput_resampled[inputs].to_numpy().T
y = idinput_resampled[outputs].to_numpy().T
print('Output shape:', y.shape)
print('Input shape:',u.shape)

#specify model identification parameters, reffer the documentation for detais.
id_method='CVA'
IC = 'None' # None, AIC, AICc, BIC
TH =  100 # The length of time horizon used for regression
fix_ordr = 23 # Used if and only if IC = 'None'
ss_orders = [1, 45] # SS orser min and max, Used if IC = AIC, AICc or BIC
SS_threshold = 0.1
req_D = True
force_A_stable = False
tsample = pd.Timedelta(idinput_resampled.index[1] - idinput_resampled.index[0]).total_seconds() # data sampling time

id_result = system_identification(
    y=y, 
    u=u,
    id_method=id_method,
    tsample= tsample,
    SS_fixed_order=fix_ordr,
    SS_orders=ss_orders,
    SS_threshold=SS_threshold,
    IC=IC,
    SS_f=TH,
    SS_D_required=req_D,
    SS_A_stability=force_A_stable
    )
firmodel = get_fir_coef(model=id_result.G, inds=inputs, deps=outputs, sampling=60, tss=tss)
step_response = get_step_response(firmodel)
t = np.arange(0, tss)

input_tag = 'FIC-2002'
output_tag = 'FIC-2101'
stp_ij = step_response[output_tag][input_tag]
imp_ij = firmodel[output_tag][input_tag]
u = idinput[input_tag]
y = idinput[output_tag]
freqs, mag, ci95, ci68, snr = get_model_uncertainty(u, y, imp_ij)
plt.plot(freqs, mag, color='red')
plt.ylim(-0.1, max(mag)*1.5)
plt.plot(freqs, snr, color='navy',linestyle="--", linewidth=0.5)
plt.fill_between(freqs, (mag-ci95), (mag+ci95), color='yellow', alpha=0.2)
plt.fill_between(freqs, (mag-ci68), (mag+ci68), color='green', alpha=0.3)
plt.semilogx()
plt.grid(True, which="both",color='gray', linestyle="-.", linewidth=0.5)
plt.show()
axes = plt.gca()
ylim = max(abs(stp_ij))*1.1
axes.set_ylim([-ylim,ylim])
colr = "red"
axes.grid(color='k', linestyle='--', linewidth=0.4)
axes.spines.bottom.set_position('zero')
axes.spines.bottom.set_linestyle('-.')
axes.spines.bottom.set_linewidth(0.5)
axes.spines[['left', 'top', 'right']].set_visible(False)
axes.xaxis.set_ticks_position('bottom')
axes.yaxis.set_ticks_position('left')
plt.xticks(np.arange(0, tss+2, 2.0))
plt.yticks(np.linspace(-ylim, ylim, 20))
axes.tick_params(axis='x', colors=colr,size=0,labelsize=4)
axes.tick_params(axis='y', colors=colr,size=0,labelsize=4)
axes.margins(x=0, y=0)
plt.plot(t, stp_ij, color=colr, linewidth=0.8)
plt.show()