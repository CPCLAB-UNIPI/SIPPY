"The Factory Class"
from high_pass_filter import HighPassFilter
from difference_filter import DifferenceFilter
from zero_mean_filter import ZeroMeanFilter
from none_filter import NoneFilter


class DetrendingFilter:  # pylint: disable=too-few-public-methods
    "The Factory Class"

    @staticmethod
    def get_filter(fiter_type):
        "A static method to get a filter"
        if fiter_type == 'highpass':
            return HighPassFilter()
        elif fiter_type in ['difference', 'doubledifference']:
            return DifferenceFilter()
        elif fiter_type == 'zeromean':
            return ZeroMeanFilter()
        elif fiter_type == 'none':
            return NoneFilter()
        else:
            return None