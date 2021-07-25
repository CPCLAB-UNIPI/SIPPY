"A Class of Filter"
from filter_data import FilterData
from interface_filter import IFilter
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
        if isinstance(argv[0], pd.DataFrame):
            self.filterdata.add_data('input', argv[0])
            self.filterdata.add_data('trend', argv[0])
            self.filterdata.add_data('output', argv[0])
        else:
            raise ValueError (f"The only Supported data type of first atgumnet of type {pd.DataFrame} but provided {type(argv[0])}")
