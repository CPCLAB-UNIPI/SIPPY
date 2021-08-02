import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append(r'.\..\sippy')
sys.path.append(r'.\sippy\detrend')
from sippy import *
from harold import simulate_step_response
from detrending_filter import DetrendingFilter

# Load spteptest data from a TSV file
file = r'data\FRAC2.csv'
columns = ['Time', 'AI-2020', 'AI-2021', 'AI-2022', 'FIC-2100', 'FIC-2101', 'FIC-2102', 'FI-2005', 'FIC-2001', 'FIC-2002', 'FIC-2004', 'QI-2106', 'TIC-2003']
step_test_data = pd.read_csv(file,skiprows=[1,2,3], usecols=columns, index_col='Time', parse_dates=True)

#slice data for model identification case
slices = {
            "slice1":{"type":"bad", "isGlobal": False, "start":1040, "end":1135, "Description": "OPC bad for AI-2020","tags":['AI-2020']}, 
            "slice2":{"type":"interpolate", "isGlobal": False,"start":3845, "end":3855, "Description": "Suspicious value for  FI-2005", "tags":['FI-2005']}
        }
# for slice in slices.values():
#     start = min(slice)
#     end = max(slice)
#     idinput.iloc[start:end] = np.nan
#     idinput.interpolate(method='linear', inplace=True)

#get time stamp for ploting
Time = step_test_data.index

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
# idinput = idinput.resample('2min').mean()

# Convert dataframe to numpy array in the shape requied for SIPPY
u = idinput[inputs].to_numpy().T
y = idinput[outputs].to_numpy().T
print('Output shape:', y.shape)
print('Input shape:',u.shape)

#specify model identification parameters, reffer the documentation for detais.
id_method='CVA'
IC = 'AIC' # None, AIC, AICc, BIC
TH =  100 # The length of time horizon used for regression
fix_ordr = 8 # Used if and only if IC = 'None'
max_order = 40 # Used if IC = AIC, AICc or BIC\
ss_orders = [1, 45]
SS_threshold = 0.1
req_D = True
force_A_stable = False
tsample = pd.Timedelta(idinput.index[1] - idinput.index[0]).total_seconds() # data sampling time

id_result = system_identification(
    y=y, 
    u=u,
    id_method=id_method,
    tsample= tsample,
    SS_fixed_order=fix_ordr,
    SS_max_order=max_order,
    SS_orders=ss_orders,
    SS_threshold=SS_threshold,
    IC=IC,
    SS_f=TH,
    SS_D_required=req_D,
    SS_A_stability=force_A_stable
    )
t = np.arange(0, tss*60, tsample)

stp_y_out, t_out = simulate_step_response(id_result.G, t)

stp_ij = stp_y_out[:,0,1]

axes = plt.gca()
ylim = max(abs(stp_ij))*1.1
axes.set_ylim([-ylim,ylim])
colr = "red"
axes.grid(color='k', linestyle='--', linewidth=0.4)
axes.spines['left'].set_position('zero')
axes.spines['bottom'].set_position('zero')
axes.spines[['top', 'right']].set_visible(False)
axes.xaxis.set_ticks_position('bottom')
axes.yaxis.set_ticks_position('left')
plt.xticks(np.arange(0, tss+2, 2.0))
plt.yticks(np.linspace(-ylim, ylim, 20))
axes.tick_params(axis='x', colors=colr,size=0,labelsize=4)
axes.tick_params(axis='y', colors=colr,size=0,labelsize=4)
axes.margins(x=0)
plt.plot(t_out/60, stp_ij, color=colr, linewidth=0.8)
plt.show()