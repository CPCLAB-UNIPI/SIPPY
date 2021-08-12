"""A Class of Filter"""
from .filter_data import FilterData
from .interface_filter import IFilter
import numpy as np
import pandas as pd

class DifferenceFilter(IFilter):
    """The Difference Filter Concrete Class implements the IFilter interface"""

    def __init__(self):
        self.filterdata = FilterData()
        self._n = 1

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
            raise ValueError (f"First argumnet dhould be dats of type {pd.DataFrame} but provided {type(argv[0])}")

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
        else:
            _sliced = argv[0].copy(deep=True)

        if self._n == 1:
            self.filterdata.add_data('output', _sliced.diff().fillna(method='backfill'))
        elif self._n == 2:
            self.filterdata.add_data('output', _sliced.diff().diff().fillna(method='backfill'))
        else:
            raise ValueError (f"Filter order takes 1 or 2 but provided {self._n}")
