from module import Module


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

