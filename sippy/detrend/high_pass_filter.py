"A Class of Filter"
from filter_data import FilterData
from interface_filter import IFilter
import pandas as pd
from scipy.signal import kaiserord, firwin, filtfilt

class HighPassFilter(IFilter):
    "The Difference Filter Concrete Class implements the IFilter interface"

    def __init__(self):
        self.filterdata = FilterData()

    def apply_filter(self, *argv):
        """
        A static method to applay the filter.

        
        This function filter data and stores in a singleton class (FilterData).
    
        Parameters:
        arg1 (Pandas.Dataframe): Data to be filterd.
        arg2 (Int): Process time to steady sate in minutes.
        arg3 (float): Multiplication factor to calculate the filter time to steady sate.
    
        Returns:
        None
        """
        if isinstance(argv[0], pd.DataFrame):
            if len(argv) != 3:
                raise ValueError('This class supports only 3 argumnets. i.e. data, process tss and filter tss multiplication factor')
            self.filterdata.add_data('input', argv[0])
        else:
            raise ValueError (f"First argumnet dhould be dats of type {pd.DataFrame} but provided {type(argv[0])}")
        try:
            self._tss = int(argv[1])
        except ValueError as _e:
            print(_e)
            print('Time to steady sate should be a number')
        try:
            self._multfactor = float(argv[2])
        except ValueError as _e:
            print(_e)
            print('filter time to steady sate multiplication factor should be a number')
        
        self._ts = pd.Timedelta(self.filterdata.data['input'].index[1] - self.filterdata.data['input'].index[0]).total_seconds()
        self._tss_sec = self._tss * 60
        self._filt_tss = self._tss_sec * self._multfactor
        self._cutoff = 1 / 2 / self._filt_tss
        self._pass_zero = 'lowpass'
        self._nyq_rate = self._ts / 2.0
        self._width = 0.5 / self._nyq_rate
        self._ripple_db = 65
        self._n,self._beta = kaiserord(self._ripple_db, self._width)
        self._window = ('kaiser', self._beta)
        self._coef = firwin(numtaps=self._n, cutoff=self._cutoff, window=self._window, pass_zero=self._pass_zero, nyq=self._nyq_rate)
        self._trend = self.filterdata.data['input'].copy(deep=True)
        self._trend[self._trend.columns] =  filtfilt(self._coef, 1.0, self.filterdata.data['input'], axis=0)
        self.filterdata.add_data('trend',self._trend)
        self.filterdata.add_data('output', self.filterdata.data['input'] - self.filterdata.data['trend'])