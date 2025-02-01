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
model.add(FCLayer(5, 10)) # 5-ös inputból generál egy 10 elemű kimenetet
model.add(Activation(tanh, tanh_deriv))
model.add(FCLayer(10,1))
model.add(Activation(sigmoid, sigmoid_deriv))
model.use_loss(mse, mse_deriv)

model.fit(x_train, y_train, 1000, 0.1)

predicted = model.predict(x_train)
print(predicted.reshape(-1))
print("expected:", y_train)


