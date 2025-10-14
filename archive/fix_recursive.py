#!/usr/bin/env python
"""Fix script for Ex_RECURSIVE.py"""

import sys
import os

# Read the file
with open('/Users/josephj/Workspace/SIPPY/Examples/Ex_RECURSIVE.py', 'r') as f:
    content = f.read()

# Replace the problematic IDData section
old_code = """# Create IDData object
data = IDData(y=Ytot, u=Usim, tsample=sampling_time)"""
new_code = """# Create IDData object with modern API
import pandas as pd
time_index = pd.date_range("2023-01-01", periods=len(Ytot[0]), freq="1s")
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
)"""

if old_code in content:
    content = content.replace(old_code, new_code)
    with open('/Users/josephj/Workspace/SIPPY/Examples/Ex_RECURSIVE.py', 'w') as f:
        f.write(content)
    print("✅ Fixed Ex_RECURSIVE.py IDData usage")
else:
    print("❌ Old code pattern not found")
