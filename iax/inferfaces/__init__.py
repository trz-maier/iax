import numpy as np
from abc import ABC, abstractmethod


class DistanceFunction(ABC):

    @staticmethod
    @abstractmethod
    def evaluate(x, y) -> float:
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

    @property
    @abstractmethod
    def input(self) -> Input:
        pass

    @abstractmethod
    def initialize(self, x: Input, classifier):
        pass

    @abstractmethod
    def search(self, **kwargs):
        return


class CostFunction(ABC):
    def __init__(self, name):
        self._name = name

    @property
    def name(self):
        return self._name

    @staticmethod
    @abstractmethod
    def calculate(classifier, input_array, expected_output):
        pass
