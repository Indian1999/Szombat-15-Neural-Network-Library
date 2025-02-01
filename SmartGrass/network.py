class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_deriv = None

    def add(self, layer):
        self.layers.append(layer)

    def __add__(self, layer):
        self.layers.append(layer)

    def use_loss(self, loss, loss_deriv):
        self.loss = loss
        self.loss_deriv = loss_deriv

    def fit(self, train_x, train_y, epochs, learning_rate):
        samples = len(train_x)
        for i in range(epochs):
            error_in_epoch = 0
            for j in range(samples):
                output = train_x[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)
                
                error_in_epoch += self.loss(train_y[j], output)
                error = self.loss_deriv(train_y[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)
            error_in_epoch /= samples
            print(f"Epoch No. {i+1}/{epochs}, loss = {error_in_epoch}")


    def predict(self, input):
        samples = len(input)
        result = []
        for i in range(samples):
            output = input[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)
        return result