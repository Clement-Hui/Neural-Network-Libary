from Core.Initializer import HeInitializer
from Core.Layers import Layer
from Core.Optimizers import Optimizer
from Core.Cost import *
from Utils.utils import *

class Model:
    def __init__(self):
        raise NotImplementedError
    def add(self,
            layer):
        raise NotImplementedError
    def compile(self):
        raise NotImplementedError
    def train(self,
              X,
              Y,
              optimizer:Optimizer,
              cost_function):
        raise NotImplementedError
    def predict(self,
                X):
        raise NotImplementedError

class Squential(Model):
    def __init__(self):
        self.layers = []


    def add(self,
            layer : Layer):
        self.layers.append(layer)


    def compile(self,
                initiaizer = HeInitializer()):
        self.layers[0].initialize(self.layers[0].input_dim, initiaizer)
        for i in range(1,len(self.layers)):

            self.layers[i].initialize(self.layers[i-1].output_dim,initiaizer)


    def train(self,
              X,
              Y,
              optimizer:Optimizer,
              cost_function,
              epoch = 10000,
              print_cost_divisor = 100,
              minibatch_size = 128,
              validation_divisor = 10,
              validation_X = None,
              validation_Y = None
              ):

        X_batch ,Y_batch,num = minibatch_seperator(X,Y,minibatch_size)

        for i in range(epoch):
            for j in range(num):
                X_current,Y_current = X_batch[j],Y_batch[j]
                AL = self.forward(X_current)
                cost = globals()[cost_function](AL,Y_current)
                dAL = globals()[cost_function+"_backward"](AL,Y_current)
                self.backward(dAL)
                self.optimize(optimizer)

            if (i% print_cost_divisor) == 0:
                print(f"Iteration {i} Cost {cost}")

            if (i% validation_divisor) == 0:
                AL = self.forward(validation_X)
                AL = np.round(AL)

                correct,wrong = evaluate(AL,validation_Y)
                print(f"Iteration {i} Validation Result {correct/validation_Y.shape[0]*100} %")


    def evaluate(self,
                 AL,
                 Y):
        return evaluate(AL,Y)




    def forward(self,
                X):
        for layer in self.layers:

            X = layer.forward(X)
        return X

    def backward(self,
                 dA):
        for layer in reversed(self.layers):
            dA = layer.backward(dA)
    def predict(self,
                X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def optimize(self,
                 optimizer):
        for layer in self.layers:
            layer.optimize(optimizer)

