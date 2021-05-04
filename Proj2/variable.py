import torch


class Tensor:

    def __init__(self, data):
        self.data = data
        self.grad = None
        self.backward_fun = None
        self.zero_grad()

    def zero_grad(self):
        self.grad = torch.zeros_like(self.data)

    def backward(self):
        # Gradient wrt to itself -> 1 (scalar)
        self.grad = torch.tensor(1)
        self.backward_fun()

    def __add__(self, other):
        return Add()(self, other)

    def __mul__(self, other):
        return Mul()(self, other)

    def sum(self):
        return Sum()(self)

    def __repr__(self):
        return f'Tensor({self.data}, grad={self.grad})'


class Module(object):
    def __init__(self):
        pass

    def forward(self, *inputs):
        res = self.forward_(*inputs)
        res.backward_fun = lambda: self.backward_fun(res, *inputs)
        return res

    def forward_(self, *inputs):
        raise NotImplementedError

    def backward_fun(self, output, *inputs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args)


class Function(Module):
    def __init__(self):
        super(Function, self).__init__()


class Add(Function):
    def forward_(self, a, b):
        return Tensor(a.data + b.data)

    def backward_fun(self, output, *inputs):
        inputs[0].grad += output.grad
        inputs[1].grad += output.grad


class Mul(Function):
    def forward_(self, a, b):
        return Tensor(a.data * b.data)

    def backward_fun(self, output, *inputs):
        inputs[0].grad += output.grad * inputs[1].data
        inputs[1].grad += output.grad * inputs[0].data


class Sum(Function):
    def forward_(self, a):
        return Tensor(a.data.sum())

    def backward_fun(self, output, *inputs):
        inputs[0].grad += output.grad * torch.ones_like(inputs[0].data)

