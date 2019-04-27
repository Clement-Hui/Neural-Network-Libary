import numpy as np


def SquaredError(AL,Y):
    E = np.mean((AL-Y)**2)
    return E

def SquaredError_backward(AL,Y):
    dE = 2*(AL-Y)
    return dE