"""
Module gathering API functions.
This module is typically imported as:
    >>> import function as F

All functions below are "elementary operations",
i.e. they reimplement the two protected functions Module._forward, Module._backward
"""

from torch import ones_like

from module import Module
from tensor import Tensor


class Function(Module):
    """Superclass of all functions."""
    def __init__(self):
        super(Function, self).__init__()


class Add(Function):
    """Matrix addition"""
    _name = 'add'

    def _forward(self, a, b):
        return Tensor(a.data + b.data)

    def _backward(self, output, *inputs):
        # dy/dx = tensor of ones
        inputs[0].grad += output.grad
        inputs[1].grad += output.grad


class Sub(Function):
    """Matrix subtraction"""
    _name = 'sub'

    def _forward(self, a, b):
        return Tensor(a.data - b.data)

    def _backward(self, output, *inputs):
        # dy/dx = tensor of ones
        inputs[0].grad += output.grad
        inputs[1].grad -= output.grad


class Mul(Function):
    """Element-wise multiplication (Hadamard product)"""
    _name = 'mul'

    def _forward(self, a, b):
        return Tensor(a.data * b.data)

    def _backward(self, output, *inputs):
        # d(xi * xj)/dxi = xj
        inputs[0].grad += output.grad * inputs[1].data
        inputs[1].grad += output.grad * inputs[0].data


class MatMul(Function):
    """Matrix multiplication"""
    _name = 'matmul'

    def _forward(self, a, b):
        return Tensor(a.data @ b.data)

    def _backward(self, output, *inputs):
        # If matmul results in a scalar (e.g. because it was a dot product), simply multiply with scalar
        if output.grad.dim() < 1:
            inputs[0].grad += output.grad * inputs[1].data
            inputs[1].grad += output.grad * inputs[0].data
        else:
            # C = AB, adjoint{A} = adjoint{C} x B^T     (x is matrix mult)
            #         adjoint{B} = A^T x adjoint{C}
            inputs[0].grad += output.grad @ inputs[1].data.T
            inputs[1].grad += inputs[0].data.T @ output.grad


class Sum(Function):
    """Summing all elements of a matrix"""
    _name = 'sum'

    def _forward(self, a):
        return Tensor(a.data.sum())

    def _backward(self, output, *inputs):
        # d(xi + xj)/dxi = tensor of ones
        inputs[0].grad += output.grad


class Transpose(Function):
    """Matrix transpose"""
    _name = 'transpose'

    def _forward(self, a):
        return Tensor(a.data.T)

    def _backward(self, output, *inputs):
        # d(output)/dx = tensor of ones
        inputs[0].grad += output.grad.T


class MSELoss(Function):
    """Mean Squared Error loss function. For efficiency, we don't create intermediate nodes."""
    _name = 'mse'

    def _forward(self, yhat, y):
        # MSE = 1/n * (yhat - y)^T (yhat - y)
        #     = 1/n * e^T e
        n = y.shape[0]
        err = yhat.data - y.data
        res = err.T @ err / n
        self._context['err_over_n'] = err / n
        return Tensor(res)

    def _backward(self, output, *inputs):
        # dyhat/dy = 1/n * 2 * e
        inputs[0].grad += 2 * self._context['err_over_n']
        inputs[1].grad -= 2 * self._context['err_over_n']


class ReLU(Function):
    """Rectified Linear Unit activation function"""
    _name = 'relu'

    def _forward(self, x) -> Tensor:
        return Tensor((x.data > 0.0) * x.data)

    def _backward(self, output, *inputs) -> None:
        # dy/dx = I{x > 0}
        inputs[0].grad += output.grad * (inputs[0].data > 0.0)


class Tanh(Function):
    """Hyperbolic tangent activation function"""
    _name = 'tanh'

    def _forward(self, x) -> Tensor:
        return Tensor(x.data.tanh())

    def _backward(self, output, *inputs) -> None:
        # d(tanh)/dx = 1 - tanh²
        inputs[0].grad += output.grad * (1. - output.data.pow(2))


class Sigmoid(Function):
    _name = 'sigmoid'

    def _forward(self, x) -> Tensor:
        return Tensor(x.data.sigmoid())

    def _backward(self, output, *inputs) -> None:
        # d(sigmoid)/dx = sigmoid * (1 - sigmoid)
        inputs[0].grad += output.grad * (output.data * (1. - output.data))
