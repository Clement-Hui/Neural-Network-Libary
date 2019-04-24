from Core.Layers import *
from Core.Models import Squential
from Core.Optimizers import GradientDescentOptimizer
from Core.Optimizers import AdamOptimizer
import numpy as np
from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
import time

((x_train,y_train),(x_test,y_test)) = mnist.load_data()

x_train = (x_train.astype(np.float64).reshape(-1,28,28,1)/255)
print(x_train.shape)
y_train = to_categorical(y_train,10)
print(y_train.shape)
x_test = x_test.astype(np.float64).reshape(-1,28,28,1)[:1000]/255
print(x_test.shape)
y_test = to_categorical(y_test,10)[:1000]


model = Squential()
model.add(Convolution(input_dim=(28,28,1),filter_num=10))
model.add(Activation("relu"))
model.add(Convolution(filter_num=10))
model.add(Activation("relu"))
model.add(Flatten())
model.add(Dense(100,activation="relu"))
model.add(Dense(10,activation="sigmoid"))
model.compile()




model.train(x_train,y_train,optimizer=AdamOptimizer(0.03,epsilon=1e-8,beta2=0.99),cost_function="SquaredError",epoch=2,print_cost_divisor=1,validation_X=x_test,validation_Y=y_test,minibatch_size=16,validation_divisor=1,minibatch_printcost= True)
model.train(x_train,y_train,optimizer=AdamOptimizer(0.001,epsilon=1e-9),cost_function="SquaredError",epoch=5,print_cost_divisor=1,validation_X=x_test,validation_Y=y_test,minibatch_size=40,validation_divisor=5,minibatch_printcost= True)
model.train(x_train,y_train,optimizer=AdamOptimizer(0.00007,epsilon=1e-9),cost_function="SquaredError",epoch=5,print_cost_divisor=1,validation_X=x_test,validation_Y=y_test,minibatch_size=50,validation_divisor=5,minibatch_printcost= True)
