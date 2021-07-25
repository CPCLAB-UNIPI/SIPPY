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
        filters = ['highpass', 'difference', 'doubledifference', 'zeromean', 'none']
        if fiter_type == filters[0]:
            return HighPassFilter()
        if fiter_type in filters[1:3]:
            difffilt = DifferenceFilter()
            difffilt._n = 1 if fiter_type == 'difference' else 2
            return difffilt
        if fiter_type == filters[3]:
            return ZeroMeanFilter()
        if fiter_type == filters[4]:
            return NoneFilter()
        if fiter_type not in filters:
            raise ValueError(f'{fiter_type} is not a supported filter. Use one of these: {filters}')