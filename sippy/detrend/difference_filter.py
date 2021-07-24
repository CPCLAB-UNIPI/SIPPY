"A Class of Filter"
from filter_data import FilterData
from interface_filter import IFilter
import pandas as pd

class DifferenceFilter(IFilter):
    "The Difference Filter Concrete Class implements the IFilter interface"

    def __init__(self):
        self.filterdata = FilterData()

    def apply_filter(self, *argv):
        if isinstance(argv[0], pd.DataFrame):
            if len(argv) > 2:
                raise ValueError('This class supports only two argumnets. i.e. data and type of diffrence')
            self.filterdata.add_data('input', argv[0])
            self.filterdata.add_data('trend', None)
        else:
            raise ValueError (f"First argumnet dhould be dats of type {pd.DataFrame} but provided {type(argv[0])}")
        if argv[1] not in [1, 2]:
            raise ValueError (f"Supported filter types are 'difference' or 'doubledifference' but provided {argv[1]}")
        else:
            self._n = argv[1]
        if self._n == 1:
           self.filterdata.add_data('output', self.filterdata.data['input'].diff().fillna(method='backfill'))
        else:
            self.filterdata.add_data('output', self.filterdata.data['input'].diff().diff().fillna(method='backfill'))