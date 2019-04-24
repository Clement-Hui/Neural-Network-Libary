import numpy as np


def relu(Z):
    np.nan_to_num(Z)
    Z[Z == 0] = 1
    A = np.maximum(Z,0)

    return A

def relu_backward(dA,Z):
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ

def sigmoid(x):
    return 1/(1+np.exp(-x))


def tanh(x):
    return np.tanh(x)


def sigmoid_backward(dA, Z):
    A = sigmoid(Z)
    dZ = dA*A*(1-A)
    return dZ