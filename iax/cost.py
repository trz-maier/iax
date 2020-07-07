import numpy as np
from typing import Tuple
from iax.inferfaces import CostFunction


class DefaultCostFunction(CostFunction):
    def __init__(self):
        super().__init__("Default Cost Function")

    @staticmethod
    def calculate(classifier, input_array, expected_output) -> Tuple[float, float, bool]:
        """
        :param classifier: classifier predict function
        :param input_array: subject array
        :param expected_output: expected label output
        :return: tuple of cost (float), confidence (float) and adversarial indicator (bool)
        """
        predictions = classifier(np.array([input_array]))[0]
        idx = list(predictions).index(max(predictions))
        confidence = predictions[expected_output]
        if idx != expected_output:
            cost = 0
        else:
            cost = confidence
        adversarial = cost == 0
        return cost, confidence, adversarial
