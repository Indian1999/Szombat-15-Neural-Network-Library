import numpy as np

def tanh(x):
    return np.tanh(x)

def tanh_deriv(x):
    return 1 - tanh(x)**2

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    return x * (1 - x)

def relu(x):
    return max([0, x.any()])

def relu_deriv(x):
    if x.any() > 0:
        return 1
    else:
        return 0