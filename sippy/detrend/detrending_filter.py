"""The Factory Class"""
from high_pass_filter import HighPassFilter
from difference_filter import DifferenceFilter
from zero_mean_filter import ZeroMeanFilter
from none_filter import NoneFilter


class DetrendingFilter:  # pylint: disable=too-few-public-methods
    """The Factory Class"""

    @staticmethod
    def get_filter(fiter_type):
        """
        A static method to get a filter.

        
        This function retuns a requested filter class.
    
        Parameters:
        fiter_type (str): Type of filter, valid options are: 'highpass', 
        'difference', 'doubledifference', 'zeromean' and 'none'
    
        Returns:
        A filter lass: Returns a filter class if provided a valid type else None
        """
        if fiter_type == 'highpass':
            return HighPassFilter()
        elif fiter_type in ['difference', 'doubledifference']:
            difffilt = DifferenceFilter()
            difffilt._n = 1 if fiter_type == 'difference' else 2
            return difffilt
        elif fiter_type == 'zeromean':
            return ZeroMeanFilter()
        elif fiter_type == 'none':
            return NoneFilter()
        else:
            return None