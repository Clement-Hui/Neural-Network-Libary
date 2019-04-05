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
