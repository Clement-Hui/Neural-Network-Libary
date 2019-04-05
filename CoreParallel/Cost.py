import numpy as np


def SquaredError(AL,Y):
    E = np.sum((AL-Y) ** 2) / Y.shape[0]
    return E

def SquaredError_backward(AL,Y):
    dE = 2*(AL-Y)
    return dE