"A Class of Filter"
from filter_data import FilterData
from interface_filter import IFilter
import pandas as pd

class ZeroMeanFilter(IFilter):
    "The Difference Filter Concrete Class implements the IFilter interface"
    
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
        if isinstance(argv[0], pd.DataFrame):
            if len(argv) > 1:
                raise ValueError('This class supports only one argumnets. i.e. data')
            self.filterdata.add_data('input', argv[0])
            self.filterdata.add_data('trend', None)
        else:
            raise ValueError (f"First argumnet dhould be dats of type {pd.DataFrame} but provided {type(argv[0])}")
        self.filterdata.add_data('output', self.filterdata.data['input'] - self.filterdata.data['input'].mean())
