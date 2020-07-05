import numpy as np
from iax.inferfaces import CostFunction


class DefaultCostFunction(CostFunction):
    def __init__(self):
        super().__init__("Default Cost Function")

    @staticmethod
    def calculate(classifier, input_array, expected_output):
        predictions = classifier(np.array([input_array]))[0]
        idx = list(predictions).index(max(predictions))
        confidence = predictions[expected_output]
        if idx != expected_output:
            out = 0
        else:
            out = confidence
        adversarial = out == 0
        return out, confidence, adversarial
