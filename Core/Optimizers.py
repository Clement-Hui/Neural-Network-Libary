import numpy as np


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
                     m,
                     v,
                     t):
        gt = dW
        mt = self.beta1 * m + (1 - self.beta1) * gt
        vt = self.beta2 * v + (1 - self.beta2) * gt ** 2
        mt_hat = mt/(1 - np.power(self.beta1,t))
        vt_hat = vt/(1 - np.power(self.beta2, t))
        gt_step = self.learning_rate * mt_hat / (np.sqrt(vt_hat) + self.epsilon)


        return gt_step,mt,vt

    def getGradientb(self,
                     db,
                     m,
                     v,
                     t):
        gt = db
        mt = self.beta1 * m + (1 - self.beta1) * gt
        vt = self.beta2 * v + (1 - self.beta2) * gt ** 2
        mt_hat = mt / (1 - self.beta1 ** t)
        vt_hat = vt / (1 - self.beta2 ** t)
        gt_step = self.learning_rate * mt_hat / (np.sqrt(vt_hat) + self.epsilon)

        return gt_step,mt,vt



