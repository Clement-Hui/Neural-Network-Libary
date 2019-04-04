import numpy as np


class Initializer:
    def getWeights(self):
        raise NotImplementedError

    def getBias(self):
        raise NotImplementedError


class HeInitializer(Initializer):
    def getBias(self,
                dim):
        b = np.zeros(dim)
        return b

    def getWeights(self,
                   dim):
        W = np.random.randn(dim[0],dim[1]) * np.sqrt(2 / dim[0])
        return W

    def getConvWeights(self,
                       input_dim,
                       kernel_size,
                       n_C):
        num = input_dim[0] * input_dim[1] * input_dim[2]
        W = np.random.randn(kernel_size,kernel_size,input_dim[2],n_C) * np.sqrt(2/num)
        return W

    def getConvBias(self,
                    output_dim):
        b = np.zeros(output_dim)
        return b

