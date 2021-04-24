

from module import Module


class Layer(Module):

    def __init__(self):
        super(Layer, self).__init__()

    def forward(self, *input):
        pass

    def backward(self, *gradwrtoutput):
        pass


class LinearLayer(Layer):

    def __init__(self):
        super(LinearLayer, self).__init__()

    def forward(self, *input):
        pass

    def backward(self, *gradwrtoutput):
        pass

