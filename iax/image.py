import numpy as np
import matplotlib.pyplot as plt
from iax.inferfaces import Input, DistanceFunction


class Image(Input):
    def __init__(self, input_array: np.array, mask: np.array = None, label=None):
        """
        TODO:
        :param input_array:
        :param mask:
        :param label:
        """
        self.__input_array = input_array
        self.__updated_array = input_array
        self.__mask = mask
        self.__label = label

    def plot(self):
        plt.imshow(self.input)
        plt.axis('off')

    @property
    def input(self):
        return self.__input_array

    @property
    def updated(self):
        return self.__updated_array

    @property
    def mask(self):
        return self.__mask

    @property
    def label(self):
        return self.__label

    def update(self, values):
        copy = self.input.copy()
        copy[self.mask] = values
        self.__updated_array = copy
        return copy

    def get_distance(self, distance_function: DistanceFunction):
        return distance_function.evaluate(self.input, self.updated)
