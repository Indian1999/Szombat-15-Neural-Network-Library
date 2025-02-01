from layer import *
from network import *
from fclayer import *
from activation_layer import *
from activations import *
from losses import *
import numpy as np

#Ha az első és az utolsó bit 1 -> 1
x_train = np.array([
    [[1, 0, 1, 1, 1]], #1
    [[1, 0, 0, 0, 1]], #1
    [[0, 1, 0, 1, 0]], #0
    [[0, 0, 0, 1, 0]], #0
    [[1, 1, 0, 0, 1]], #1
    [[0, 0, 1, 1, 1]], #0
    [[0, 1, 0, 0, 0]], #0
    [[1, 1, 1, 1, 1]], #1
    [[1, 0, 0, 1, 0]], #0
    [[0, 1, 0, 0, 1]]  #0
])
y_train = np.array([
    [[1]],
    [[1]],
    [[0]],
    [[0]],
    [[1]],
    [[0]],
    [[0]],
    [[1]],
    [[0]],
    [[0]],
])

model = Network()


