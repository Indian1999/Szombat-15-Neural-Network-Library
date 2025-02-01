from layer import Layer

class Activation(Layer):
    def __init__(self, activation, activation_deriv):
        self.activation = activation
        self.activation_deriv = activation_deriv

    def forward_propagation(self, input):
        self.input = input
        self.output(self.activation(self.input))
        return self.output