"The Abstract Factory Interface for Filter"
from abc import ABCMeta, abstractmethod


class IFilter(metaclass=ABCMeta):
    "Abstract Filter Factory Interface"

    @staticmethod
    @abstractmethod
    def apply_filter(data, *argv):
        "The static Abstract factory interface method"
