"A Class of Filter"
from .filter_data import FilterData
from .interface_filter import IFilter
import numpy as np
import pandas as pd

class NoneFilter(IFilter):
    "The None Filter Concrete Class implements the IFilter interface"

    def __init__(self):
        self.filterdata = FilterData()

    def apply_filter(self, *argv):
        """
        A static method to applay the filter.

        
        This function filter data and stores in a singleton class (FilterData).
    
        Parameters:
        arg1 (Pandas.Dataframe): Data to be filterd.
    
        Returns:
        None
        """
        if len(argv) < 2:
                raise ValueError("This class supports minimum 2 argumnets. i.e. data and slicees")
        if isinstance(argv[0], pd.DataFrame):
            self.filterdata.add_data('input', argv[0])
            self.filterdata.add_data('trend', argv[0])
        else:
            raise ValueError (f"The only Supported data type of first atgumnet of type {pd.DataFrame} but provided {type(argv[0])}")
        if isinstance(argv[1], dict):
            slices = argv[1] 
        else:
            raise TypeError("Slices should be a pyhton dictionary")
        if slices:
            _sliced = argv[0].copy(deep=True)
            for slice in slices.values():
                if slice['type'] == "interpolate" and any((True for tag in slice['tags'] if tag in _sliced.columns)):
                    for tag in slice['tags']:
                        _sliced[tag].iloc[slice["start"]:slice["end"]] = np.nan
                        _sliced[tag].interpolate(method='linear', inplace=True)
                elif slice['type'] == "bad" and(slice["isGlobal"] or any((True for tag in slice['tags'] if tag in _sliced.columns))):
                    _sliced.iloc[slice["start"]:slice["end"]] = np.nan
                    _sliced.fillna(method='ffill', inplace=True)   
            self.filterdata.add_data('output', _sliced)
        else:
            self.filterdata.add_data('output', argv[0])

