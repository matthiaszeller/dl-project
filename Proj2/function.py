"""
Python module gathering functions, i.e. elementary operations as well as more complex ones (e.g. MSE).
This module is typically imported as:
    >>> import function as F
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
        inputs[0].grad += output.grad
        inputs[1].grad += output.grad


class Sub(Function):
    """Matrix subtraction"""
    _name = 'sub'

    def _forward(self, a, b):
        return Tensor(a.data - b.data)

    def _backward(self, output, *inputs):
        inputs[0].grad += output.grad
        inputs[1].grad -= output.grad


class Mul(Function):
    """Element-wise multiplication (Hadamard product)"""
    _name = 'mul'

    def _forward(self, a, b):
        return Tensor(a.data * b.data)

    def _backward(self, output, *inputs):
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
            inputs[0].grad += output.grad @ inputs[1].data.T
            inputs[1].grad += inputs[0].data.T @ output.grad


class Sum(Function):
    """Summing all elements of a matrix"""
    _name = 'sum'

    def _forward(self, a):
        return Tensor(a.data.sum())

    def _backward(self, output, *inputs):
        inputs[0].grad += output.grad * ones_like(inputs[0].data)


class Dot(Function):
    # TODO: remove?
    _name = 'dot'

    def _forward(self, a, b):
        """a, b are both 1D tensors"""
        return Tensor(a.data.dot(b.data))

    def _backward(self, output, *inputs):
        a, b = inputs
        a.grad += output.grad * b.data
        b.grad += output.grad * a.data


class ReLU(Function):
    """Rectified Linear Unit activation function"""
    _name = 'relu'

    def _forward(self, x) -> Tensor:
        return Tensor((x.data > 0.0) * x.data)

    def _backward(self, output, *inputs) -> None:
        # input and output have the same shape, simply multiply element-wise
        inputs[0].grad += output.grad * \
                          (inputs[0].data > 0.0)


class Transpose(Function):
    """Matrix transpose"""
    _name = 'transpose'

    def _forward(self, a):
        return Tensor(a.data.T)

    def _backward(self, output, *inputs):
        inputs[0].grad += output.grad.T


class MSELoss(Function):
    """Mean Squared Error loss function. For efficiency, we don't create intermediate nodes."""
    _name = 'mse'

    def _forward(self, yhat, y): # TODO: reimplement as function of elementary operations, just as other functions?
        n = y.shape[0]
        err = yhat.data - y.data
        res = err.T @ err / n
        self._context['err_over_n'] = err / n
        return Tensor(res)

    def _backward(self, output, *inputs):
        inputs[0].grad += 2 * self._context['err_over_n']
        inputs[1].grad -= 2 * self._context['err_over_n']

