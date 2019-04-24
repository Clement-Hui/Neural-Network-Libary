import numpy as numpy
from numpy import iinfo


class Optimizer:


    def getGradientW(self):
        raise NotImplementedError

    def getGradientb(self):
        raise NotImplementedError

class GradientDescentOptimizer(Optimizer):
    def __init__(self,
                 learning_rate=0.02):
        self.learning_rate = learning_rate

    def getGradientW(self,
                     dW):
        return dW * self.learning_rate

    def getGradientb(self,
                     db):
        return db * self.learning_rate


class AdamOptimizer(Optimizer):
    def __init__(self,
                 learning_rate = 0.001,
                 beta1 = 0.9,
                 beta2 = 0.999,
                 epsilon = 1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon


    def getGradientW(self,
                     dW,
                     vdW,
                     sdW,
                     t = 2):
        vdW = self.beta1 * vdW + (1 - self.beta1) * dW

        v_corrected_dW = vdW / (1 - numpy.power(self.beta1, t))
        numpy.nan_to_num(v_corrected_dW, False)
        v_corrected_dW[v_corrected_dW == 0.0] = 1

        sdW = self.beta2 * sdW + (1 - self.beta2) * numpy.power(dW, 2)

        sdW = numpy.nan_to_num(sdW)
        sdW[sdW == 0.0] = 1

        s_corrected_dW = sdW / (1 - numpy.power(self.beta2, t))
        numpy.nan_to_num(s_corrected_dW, False)
        s_corrected_dW[s_corrected_dW == 0.0] = 1
        a=numpy.sqrt(s_corrected_dW)
        dW_step = v_corrected_dW / (a + self.epsilon) * self.learning_rate


        return dW_step,vdW,sdW

    def getGradientb(self,
                     db,
                     vdb,
                     sdb,
                     t = 2):
        vdb = self.beta1 * vdb + (1 - self.beta1) * db

        v_corrected_dW = vdb / (1 - numpy.power(self.beta1, t))


        sdb = self.beta2 * sdb + (1 - self.beta2) * numpy.power(db, 2)

        sdb = numpy.nan_to_num(sdb)
        sdb[sdb == 0.0] = 1

        s_corrected_dW = sdb / (1 - numpy.power(self.beta2, t))

        dW_step = v_corrected_dW / (
                numpy.sqrt(s_corrected_dW) + self.epsilon) * self.learning_rate

        return dW_step, vdb, sdb



