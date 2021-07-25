"""A Filter IO data Singleton Class"""

class FilterData():
    """The Filter IO data as a Singleton"""
    data = {}
    def __new__(cls):
        return cls
    
    @classmethod
    def add_data(cls, series_name, data):
        """A class level method"""
        cls.data[series_name] = data