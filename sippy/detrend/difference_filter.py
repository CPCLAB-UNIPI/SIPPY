"""A Class of Filter"""
from filter_data import FilterData
from interface_filter import IFilter
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
        if isinstance(argv[0], pd.DataFrame):
            if len(argv) > 1:
                raise ValueError('This class supports only one argumnet. i.e. data')
            self.filterdata.add_data('input', argv[0])
            self.filterdata.add_data('trend', argv[0])
        else:
            raise ValueError (f"First argumnet dhould be dats of type {pd.DataFrame} but provided {type(argv[0])}")

        if self._n == 1:
            self.filterdata.add_data('output', self.filterdata.data['input'].diff().fillna(method='backfill'))
        elif self._n == 2:
            self.filterdata.add_data('output', self.filterdata.data['input'].diff().diff().fillna(method='backfill'))
        else:
            raise ValueError (f"Filter order takes 1 or 2 but provided {self._n}")
