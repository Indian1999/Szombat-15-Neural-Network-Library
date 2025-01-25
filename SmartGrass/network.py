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

    def fit():
        pass

    def predict():
        pass