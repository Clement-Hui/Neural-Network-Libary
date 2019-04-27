import numpy as np


def relu(Z):

    
    A = np.maximum(Z,0)

    return A

def relu_backward(dA,Z):
    A = relu(Z)
    dZ  = np.multiply(dA, np.int64(A > 0))
    dZ[Z <= 0] = 0
    return dZ

def sigmoid(x):
    return 1/(1+np.exp(-x))


def tanh(x):
    return np.tanh(x)


def sigmoid_backward(dA, Z):
    dZ = np.exp(-Z)/(1+np.exp(-Z))**2 * dA
    return dZ