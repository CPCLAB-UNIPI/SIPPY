"A Class of Filter"
from filter_data import FilterData
from interface_filter import IFilter
import numpy as np
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
            if len(argv) < 4:
                raise ValueError(
                    "This class supports minimum 4 argumnets. i.e. data, process tss and filter tss multiplication factor and data slicees"
                )
            self.filterdata.add_data("input", argv[0])
            if isinstance(argv[3], dict):
                slices = argv[3] 
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
                self.filterdata.add_data("sliceed", _sliced)
        else:
            raise ValueError(
                f"First argumnet dhould be dats of type {pd.DataFrame} but provided {type(argv[0])}"
            )
        try:
            _tss = int(argv[1])
        except ValueError as _e:
            _tss = 60
            print(_e)
            print(f"Time to steady sate should be a number, setting{_tss}")
        try:
            _multfactor = float(argv[2])
        except ValueError as _e:
            _multfactor = 6
            print(_e)
            print(
                f"filter time to steady sate multiplication factor should be a number, setting{_multfactor}"
            )

        _ts = pd.Timedelta(
            self.filterdata.data["input"].index[1]
            - self.filterdata.data["input"].index[0]
        ).total_seconds()
        _tss_sec = _tss * 60
        _filt_tss = _tss_sec * _multfactor
        _cutoff = 1 / 2 / _filt_tss
        _pass_zero = "lowpass"
        _nyq_rate = _ts / 2.0
        _width = 0.5 / _nyq_rate
        _ripple_db = 65
        _n, _beta = kaiserord(_ripple_db, _width)
        _window = ("kaiser", _beta)
        _coef = firwin(
            numtaps=_n,
            cutoff=_cutoff,
            window=_window,
            pass_zero=_pass_zero,
            nyq=_nyq_rate,
        )
        _trend = self.filterdata.data["input"].copy(deep=True)
        if slices:
            _trend[_trend.columns] = filtfilt(_coef, 1.0, self.filterdata.data["sliceed"], axis=0)
            for slice in slices.values():
                if slice['type'] == "bad" and (slice["isGlobal"] or any((True for tag in slice['tags'] if tag in _trend.columns))):
                    _trend.iloc[slice["start"]:slice["end"]] = self.filterdata.data["input"].iloc[slice["start"]:slice["end"]]
        else:
            _trend[_trend.columns] = filtfilt(_coef, 1.0, self.filterdata.data["input"], axis=0)
        self.filterdata.add_data("trend", _trend)
        self.filterdata.add_data(
            "output", self.filterdata.data["input"] - self.filterdata.data["trend"]
        )
