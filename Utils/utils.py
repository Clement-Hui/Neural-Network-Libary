import numpy as np

def minibatch_seperator(X,
                        Y,
                        minibatch_size = 128):
    m = X.shape[0]
    if minibatch_size == 0:
        minibatch_size = m

    p = np.random.permutation(m)
    shuffled_X = X[p]
    shuffled_Y = Y[p]

    divisible = (m % minibatch_size) == 0

    minibatch_num = int(np.floor(m / minibatch_size))

    X_batches = []
    Y_batches = []

    for i in range(minibatch_num):
        X_batches.append(shuffled_X[i * minibatch_size:(i + 1) * minibatch_size])
        Y_batches.append(shuffled_Y[i * minibatch_size:(i + 1) * minibatch_size])

    if divisible == False:
        X_batches.append(shuffled_X[(minibatch_num - 1) * minibatch_size:-1])
        Y_batches.append(shuffled_Y[(minibatch_num - 1) * minibatch_size:-1])
        minibatch_num += 1

    return X_batches,Y_batches,minibatch_num

def evaluate(data, labels):
    corrects, wrongs = 0, 0
    ans = data.astype(int)
    labels = labels.astype(int)
    for i in range(data.shape[0]):
        res = ans[i]
        if (res==labels[i]).all():
            corrects += 1
        else:
            wrongs += 1

    return corrects, wrongs

def zero_pad(X,
             pad):
    X_pad = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant', constant_values=(0, 0))
    return X_pad

def conv_single_step(slice,
                     W,
                     b):
    s = slice*W
    Z = np.sum(s)
    Z = float(Z+b)

    return Z

