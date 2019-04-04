from Core.Layers import *
from Core.Models import Squential
from Core.Optimizers import GradientDescentOptimizer
import numpy as np
from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
import time

((x_train,y_train),(x_test,y_test)) = mnist.load_data()

x_train = x_train.astype(np.float32).reshape(-1,28,28,1)/255 -0.5
print(x_train.shape)
y_train = to_categorical(y_train,10)
print(y_train.shape)
x_test = x_test.astype(np.float32).reshape(-1,28,28,1)/255 -0.5
print(x_test.shape)
y_test = to_categorical(y_test,10)


model = Squential()
model.add(Convolution(input_dim=(28,28,1)))


model.compile()

tic = time.time()
model.forward(x_train)
toc = time.time()

print(toc-tic)
#model.train(x_train,y_train,optimizer=GradientDescentOptimizer(0.001),cost_function="SquaredError",epoch=100000,print_cost_divisor=1,validation_X=x_test,validation_Y=y_test)


