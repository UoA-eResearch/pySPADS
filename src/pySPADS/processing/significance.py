import numpy as np


def zero_crossings(x: np.ndarray) -> int:
    """Count the number of times the signal crosses zero"""
    return np.sum(np.diff(np.sign(x)) != 0)


def maxima(x) -> int:
    """Count the number of local maxima"""
    return np.sum(np.diff(np.sign(np.diff(x))) == -2)
