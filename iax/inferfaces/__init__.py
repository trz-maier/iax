from abc import ABC, abstractmethod
from typing import Tuple


class DistanceFunction(ABC):
    def __init__(self, name):
        """
        Distance Function Interface used to generalise the calculation within the library.
        Implements evaluate function.
        :param name:
        """
        self._name = name

    @staticmethod
    @abstractmethod
    def evaluate(x, y) -> float:
        """
        :param x: value
        :param y: compared to
        :return: distance value as a float
        """
        pass


class Input(ABC):

    @property
    @abstractmethod
    def input(self):
        pass

    @property
    @abstractmethod
    def updated(self):
        pass

    @property
    @abstractmethod
    def mask(self):
        pass

    @property
    @abstractmethod
    def label(self):
        pass

    @abstractmethod
    def update(self, values):
        pass

    @abstractmethod
    def get_distance(self, distance_function: DistanceFunction) -> float:
        pass


class Engine(ABC):
    def __init__(self, name):
        self._name = name

    @property
    @abstractmethod
    def input(self) -> Input:
        pass

    @property
    @abstractmethod
    def max_distance(self):
        pass

    @property
    @abstractmethod
    def output(self):
        pass

    @abstractmethod
    def initialize(self, x: Input, classifier):
        pass

    @abstractmethod
    def search(self, **kwargs):
        """
        Base search function used to find adversarial examples and passing any needed keyword arguments.
        The implemented algorithm's
        :param kwargs: keyword arguments needed to execute given search algorithm
        :return: None
        """
        pass


class CostFunction(ABC):
    def __init__(self, name):
        self._name = name

    @property
    def name(self):
        """
        :return: cost function name for identification
        """
        return self._name

    @staticmethod
    @abstractmethod
    def calculate(classifier, input_array, expected_output) -> Tuple[float, float, bool]:
        pass
