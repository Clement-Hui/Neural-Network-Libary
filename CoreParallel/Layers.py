import numpy as np


from Core.Activations import *
from Core.Initializer import *
from Core.Optimizers import *
from Utils.utils import *
import multiprocessing as mp

class Layer:
    def __init__(self,
                 output_dim,
                 input_dim = None):
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.W = None
        self.b = None
        self.Z = None
        self.X = None

        self.dA = None
        self.dZ = None
        self.dW = None
        self.db = None
        self.dA_prev = None

        self.m = None



    def forward(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError

    def initialize(self,
                   input_dim,
                   initializer):
        raise NotImplementedError


class Dense(Layer):
    def __init__(self,
                 output_dim,
                 input_dim=None,
                 activation="relu"):
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.W = None
        self.b = None
        self.Z = None
        self.X = None

        self.dA = None
        self.dZ = None
        self.dW = None
        self.db = None
        self.dA_prev = None

        self.m = None

        self.activation = activation


    def forward(self,
                X):
        self.X = X
        self.m = self.X.shape[0]
        self.Z = np.dot(self.X , self.W) + self.b
        self.A = globals()[self.activation](self.Z)
        return self.A


    def backward(self,
                 dA):

        self.dA = dA
        self.dZ = globals()[self.activation+"_backward"](self.dA,self.Z)
        self.dW = np.dot(self.X.T,self.dZ)
        self.db = np.sum(self.dZ,axis = 0,keepdims=True)/self.m
        self.dA_prev = np.dot(self.dZ,self.W.T)

        return self.dA_prev


    def initialize(self,
                   input_dim,
                   initializer = HeInitializer()):
        self.input_dim = input_dim

        self.W = initializer.getWeights((self.input_dim, self.output_dim))
        self.b = initializer.getBias((1,self.output_dim))


    def optimize(self,
                 optimizer = GradientDescentOptimizer(0.02)):
        dW_step = optimizer.getGradientW(self.dW)
        db_step = optimizer.getGradientb(self.db)

        self.W -= dW_step
        self.b -= db_step


class Convolution(Layer):
    def __init__(self,
                 input_dim=None,
                 kernel_size = 3,
                 stride = 1,
                 pad = 'valid',
                 filter_num = 1
                 ):
        self.stride = stride
        self.kernel_size = kernel_size
        self.pad = pad
        self.n_C = filter_num

        self.input_dim = input_dim


        self.W = None
        self.b = None
        self.Z = None
        self.X = None

        self.dA = None
        self.dZ = None
        self.dW = None
        self.db = None
        self.dA_prev = None

        self.m = None





        if kernel_size%2 == 0:
            print("Kernel size must be odd")
            print("Lowering kernel size by one")
            if kernel_size >= 2:
                print(f"New Kernel Size is {kernel_size-1}")
                self.kernel_size = kernel_size-1
            else:
                print("Kernel size smaller than 2, automatically setting kernel_size to 3")
                print(f"New Kernel Size is 3")
                self.kernel_size = 3


        if pad == 'same':
            self.pad = (self.kernel_size-1)/2
        elif pad == 'valid':
            self.pad = 0
        elif isinstance(pad,int) == False:
            print("pad size must be integer or 'same' or 'valid'.")
            print("lowering pad size to 0")
            self.pad = 0


    def forward(self,
                X):

        (n_H_prev, n_W_prev, n_C_prev) = self.input_dim

        self.n_H = int(np.floor((n_H_prev - self.kernel_size + 2 * self.pad) / self.stride) + 1)
        self.n_W = int(np.floor((n_W_prev - self.kernel_size + 2 * self.pad) / self.stride) + 1)


        (m,n_H_prev, n_W_prev, n_C_prev) = X.shape
        self.m = m
        pad = self.pad
        stride = self.stride
        f = self.kernel_size
        n_H = self.n_H
        n_W = self.n_W
        n_C = self.n_C

        self.Z = np.zeros((m,n_H, n_W, n_C))

        self.X_pad = zero_pad(X,self.pad)



        a = range(m)

        pool = mp.Pool(20)
        pool.map(self.one_sample_forward,a)


        self.A_prev = X
        return self.Z

    def one_sample_forward(self,i):
        f = self.kernel_size
        current_pad = self.X_pad[i]
        for h in range(self.n_H):
            for w in range(self.n_W):
                for c in range(self.n_C):
                    vert_start = h * self.stride
                    vert_end = vert_start + f
                    horiz_start = w * self.stride
                    horiz_end = horiz_start + f

                    current_slice = current_pad[vert_start:vert_end, horiz_start:horiz_end, c]

                    self.Z[i, w, h, c] = conv_single_step(current_slice, self.W[:, :, :, c], self.b[:, :, :, c])

    def backward(self):
        (n_H_prev,n_W_prev,n_C_prev,m) = self.input_dim.shape

        pad = self.pad
        stride = self.stride
        f = self.filter_size
        n_H = self.n_H
        n_W = self.n_W
        n_C = self.n_C

        Z = np.zeros((n_H, n_W, n_C, m))

        self.dA_prev = np.empty((n_H_prev, n_W_prev, n_C_prev, m))
        self.dW = np.empty((f, f, n_C_prev, n_C))
        db = np.empty((1, 1, 1, n_C))

        A_pad = zero_pad(self.A, pad)
        dA_prev_pad = zero_pad(self.dA, pad)

        for i in range(m):

            current_A_pad = A_pad[i]
            current_dA_prev_pad = dA_prev_pad[i]
            for h in range(n_H):
                for w in range(n_W):
                    for c in range(n_C):
                        vert_start = h * stride
                        vert_end = vert_start + f
                        horiz_start = w * stride
                        horiz_end = horiz_start + f

                        a_slice = current_A_pad[vert_start:vert_end, horiz_start:horiz_end, :]
                        da_slice = current_dA_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]
                        da_slice += self.W[:, :, :, c] * self.dZ[i, h, w, c]
                        self.dW[:, :, :, c] += a_slice * self.dZ[i, h, w, c]
                        self.db[:, :, :, c] += self.dZ[i, h, w, c]
            self.dA_prev[i, :, :, :] = current_dA_prev_pad[pad:-pad, pad:-pad, :]
        return self.dA_prev


    def initialize(self,
                   input_dim,
                   initializer):
        (n_H_prev, n_W_prev, n_C_prev) = self.input_dim

        self.n_H = int(np.floor((n_H_prev - self.kernel_size + 2 * self.pad) / self.stride) + 1)
        self.n_W = int(np.floor((n_W_prev - self.kernel_size + 2 * self.pad) / self.stride) + 1)

        self.output_dim = (self.n_H, self.n_W, self.n_C)
        if self.input_dim == None:
            self.input_dim = input_dim
        self.W = initializer.getConvWeights(self.input_dim,self.kernel_size,self.n_C)
        self.b = initializer.getConvBias((1,1,1,self.n_C))












