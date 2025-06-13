"""test_armax.py"""

__author__ = "A. De Bortoli"

import numpy as np
from numpy.testing import assert_equal, assert_raises

from sippy_unipi.armax import Armax


class TestCtor:
    def test_array_range(self):
        na_range = [1, 2]
        nb_range = [2, 3]
        nc_range = [3, 4]
        delay_range = [4, 5]
        dt = 5

        a = Armax(na_range, nb_range, nc_range, delay_range, dt)
        assert_equal(a.na_range, np.array(na_range))
        assert_equal(a.nb_range, np.array(nb_range))
        assert_equal(a.nc_range, np.array(nc_range))
        assert_equal(a.dt, dt)

    def test_exceptions(self):
        assert_raises(ValueError, Armax, "invalid", 2, 3, 4, 5)
        assert_raises(ValueError, Armax, [1.5, 2], 2, 3, 4, 5)
        assert_raises(ValueError, Armax, (np.nan, 1), 2, 3, 4, 5)
