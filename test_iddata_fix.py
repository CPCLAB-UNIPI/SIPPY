"""Test script to understand IDData API usage"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import pandas as pd
from sippy.identification.iddata import IDData

# Create test data
npts = 100
Usim = np.random.randn(4, npts)
Ytot = np.random.randn(3, npts)

print("Testing modern IDData API...")
try:
    # Try the modern API
    time_index = pd.date_range("2023-01-01", periods=npts, freq="1s")
    data_dict = {}
    for i in range(4):
        data_dict[f"u{i+1}"] = Usim[i, :]
    for i in range(3):
        data_dict[f"y{i+1}"] = Ytot[i, :]
    
    data_df = pd.DataFrame(data_dict, index=time_index)
    data = IDData(
        data=data_df, 
        inputs=[f"u{i+1}" for i in range(4)], 
        outputs=[f"y{i+1}" for i in range(3)], 
        tsample=1.0
    )
    print(f"✅ Modern API works: {iddata}")
except Exception as e:
    print(f"❌ Modern API failed: {e}")
    
print("Test complete")
