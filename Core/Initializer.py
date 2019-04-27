import numpy as np


class Initializer:
    def getWeights(self):
        raise NotImplementedError

    def getBias(self):
        raise NotImplementedError


class HeInitializer(Initializer):
    def getBias(self,
                dim):
        b = np.array(np.zeros(dim), dtype=np.float32)
        return b

    def getWeights(self,
                   dim):
        W = np.array(np.random.randn(dim[0],dim[1]) * np.sqrt(2 / dim[0]), dtype=np.float32)
        return W

    def getConvWeights(self,
                       input_dim,
                       kernel_size,
                       n_C):
        num = kernel_size
        stddev = np.sqrt(num) / .87962566103423978
        W = np.array(np.random.randn(kernel_size,kernel_size,input_dim[2],n_C) * np.sqrt(1/stddev), dtype=np.float32)
        return W

    def getConvBias(self,
                    output_dim):
        b = np.array(np.zeros(output_dim), dtype=np.float32)
        return b

