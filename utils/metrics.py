import numpy as np


def MSE(predictions: np.ndarray, targets: np.ndarray) -> float:
    """calculate loss of your model without loops"""
    return ((targets - predictions) ** 2).mean(axis=None)
