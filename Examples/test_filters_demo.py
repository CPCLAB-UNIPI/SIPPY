#!/usr/bin/env python3
"""Test script for the new filters module."""

import pandas as pd
import numpy as np
import sys
import os

# Add the source directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from sippy.filters import FilterFactory

def test_basic_functionality():
    """Test basic filter functionality."""
    # Create test data
    np.random.seed(42)
    index = pd.date_range('2023-01-01', periods=100, freq='h')
    data = pd.DataFrame({
        'signal1': np.sin(2*np.pi*0.01*np.arange(100)),
        'signal2': np.cos(2*np.pi*0.05*np.arange(100)),
    }, index=index)
    
    print("Testing basic filter functionality...")
    
    # Test zero-mean filter
    try:
        zero_mean = FilterFactory.create('zeromean')
        result = zero_mean.apply_filter(data)
        print(f"✓ Zero-mean filter works! Shape: {result.shape}")
        print(f"Original mean (signal1): {data['signal1'].mean():.6f}")
        print(f"Zero-mean (signal1): {result['signal1'].mean():.6f}")
    except Exception as e:
        print(f"❌ Zero-mean filter error: {e}")
    
    # Test none filter
    try:
        none_filter = FilterFactory.create('none')
        result2 = none_filter.apply_filter(data)
        assert (data.values == result2.values).all(), "None filter should pass data through unchanged"
        print("✓ None filter works correctly")
    except Exception as e:
        print(f"❌ None filter error: {e}")
    
    # Get filter information
    try:
        info = FilterFactory.get_filter_info('zeromean')
        print(f"Zero-mean filter info: {info}")
    except Exception as e:
        print(f"❌ Filter info error: {e}")

if __name__ == '__main__':
    test_basic_functionality()
