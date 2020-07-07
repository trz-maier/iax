import numpy as np
import scipy.spatial.distance as dist
from iax.inferfaces import DistanceFunction


class EuclideanDistance(DistanceFunction):
    def __init__(self):
        super().__init__("Squared Euclidean Distance")

    @staticmethod
    def evaluate(x: np.array, y: np.array) -> float:
        return dist.sqeuclidean(x.flatten(), y.flatten())
