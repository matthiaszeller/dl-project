from module import Module
import numpy as np


class Function(Module):

    def __init__(self):
        super(Function, self).__init__()

        self.f = None
        self.df = None

    def __call__(self, *input):
        return self.f(*input)

    def forward(self, *input):
        return self.f(*input)

    def backward(self, *gradwrtoutput):
        return self.df(*gradwrtoutput)


class ActivationFunction(Function):

    def __init__(self):
        super(ActivationFunction, self).__init__()


class ActivationFunIdentity(ActivationFunction):

    def __init__(self):
        super(ActivationFunIdentity, self).__init__()

        self.f = lambda *x: x
        self.df = lambda *x: 1

class relu(ActivationFunction):

   def __init__(self):
        super().__init__()

        self.f = lambda *x: tuple(map(lambda y: np.maximum(0, y), x))
        self.df = lambda *x: tuple(map(lambda y:(y>0).float(), x))
        

class tanh(ActivationFunction):
   def __init__(self):
        super().__init__()
        self.f =  lambda *x: tuple(map(lambda y: np.tanh(y), x))
        self.df = lambda *x: tuple(map(lambda y: 1-np.power(np.tanh(y),2), x)