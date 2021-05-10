from __future__ import annotations

from typing import Union, Any, List

import torch


class Tensor:

    parents: List[Tensor]
    _name = None

    def __init__(self, data):
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data)

        if data.dtype is not torch.float32:
            data = data.to(torch.float32)

        # Work only with matrices
        if data.dim() < 2:
            data = data.reshape(-1, 1)

        self.data = data
        self.grad = None
        self.backward_fun = lambda: ()
        self.parents = []
        self.zero_grad()

    def zero_grad(self) -> None:
        self.grad = torch.zeros_like(self.data)

    def backward(self) -> None:
        # Initialize the root adjoint variable: gradient wrt to itself -> 1 (scalar)
        self.grad = torch.tensor(1.0)
        self.backward_fun()

        # Walk through the graph backwards
        # TODO: currently only works for sequential (parallel branches in graph could not work)
        queue = self.parents.copy()
        for p in queue:
            p.backward_fun()
            queue.extend(p.parents)

    def __add__(self, other) -> Tensor:
        return Add()(self, other)

    def __sub__(self, other):
        return Sub()(self, other)

    def __mul__(self, other) -> Tensor:
        return Mul()(self, other)

    def __matmul__(self, other) -> Tensor:
        return MatMul()(self, other)

    def sum(self) -> Tensor:
        """Sum over all axes."""
        return Sum()(self)

    def dot(self, other) -> Tensor:
        return Dot()(self, other)

    @property
    def T(self) -> Tensor:
        return self.transpose()

    def transpose(self) -> Tensor:
        return Transpose()(self)

    @property
    def shape(self):
        if self.data.dim() == 0:
            return (1, )
        return self.data.shape

    def item(self):
        return self.data.item()

    def __repr__(self) -> str:
        name = '' if self._name is None else f', name={self._name}'
        return f'Tensor({self.data}, grad={self.grad}{name})'


class Module(object):
    _name = ''

    def __init__(self):
        self._context = dict()

    def forward(self, *inputs) -> Tensor:
        """Superclass method wrapping forward computation.
        This method must **not** be subclassed for elementary operations (subclass _forward instead).
        This method **must** be subclassed for operations that are a combination of elementary operations,
        (e.g. linear layer)"""
        # Do actual forward computation
        res = self._forward(*inputs)
        # Initialize the backward function
        res.backward_fun = lambda: self.backward(res, *inputs)

        # Make resulting computation aware of the inputs that created it
        for i in inputs:
            res.parents.append(i)

        return res

    def _forward(self, *inputs) -> Tensor:
        """Actual forward computation for elementary operations."""
        raise NotImplementedError

    def backward(self, output, *inputs):
        """Backward function stored in tensors that are results of an operation.
        The gradient of the 'parent' tensors will be modified in place.

        Example: say c = f(a, b), with a,b,c tensors and f an operator. Then the backward function of f will be stored
        in c and will modify in place the gradient of tensors a and b."""
        self._backward_fun(output, *inputs)

    def _backward_fun(self, output, *inputs):
        """Subclass this method for elementary operations."""
        raise NotImplementedError

    def params(self):
        """Return a list of tuples for each parameter: (tensor, gradient)."""
        return [(e.data, e.grad) for e in self._params()]

    def _params(self):
        """Subclass this method to return a list of Tensors"""
        raise []

    def __call__(self, *args, **kwargs):
        return self.forward(*args)

    def __repr__(self):
        return self._name


class Function(Module):
    def __init__(self):
        super(Function, self).__init__()


class Add(Function):
    _name = 'add'

    def _forward(self, a, b):
        return Tensor(a.data + b.data)

    def _backward_fun(self, output, *inputs):
        inputs[0].grad += output.grad
        inputs[1].grad += output.grad


class Sub(Function):
    _name = 'sub'

    def _forward(self, a, b):
        return Tensor(a.data - b.data)

    def _backward_fun(self, output, *inputs):
        inputs[0].grad += output.grad
        inputs[1].grad -= output.grad


class Mul(Function):
    _name = 'mul'

    def _forward(self, a, b):
        return Tensor(a.data * b.data)

    def _backward_fun(self, output, *inputs):
        inputs[0].grad += output.grad * inputs[1].data
        inputs[1].grad += output.grad * inputs[0].data


class MatMul(Function):
    _name = 'matmul'

    def _forward(self, a, b):
        return Tensor(a.data @ b.data)

    def _backward_fun(self, output, *inputs):
        # If matmul results in a scalar (e.g. because it was a dot product), simply multiply with scalar
        if output.grad.dim() < 1:
            inputs[0].grad += output.grad * inputs[1].data
            inputs[1].grad += output.grad * inputs[0].data
        else:
            inputs[0].grad += output.grad @ inputs[1].data.T
            inputs[1].grad += inputs[0].data.T @ output.grad


class Sum(Function):
    _name = 'sum'

    def _forward(self, a):
        return Tensor(a.data.sum())

    def _backward_fun(self, output, *inputs):
        inputs[0].grad += output.grad * torch.ones_like(inputs[0].data)


class Dot(Function):
    _name = 'dot'

    def _forward(self, a, b):
        """a, b are both 1D tensors"""
        return Tensor(a.data.dot(b.data))

    def _backward_fun(self, output, *inputs):
        a, b = inputs
        a.grad += output.grad * b.data
        b.grad += output.grad * a.data


class Transpose(Function):
    _name = 'transpose'

    def _forward(self, a):
        return Tensor(a.data.T)

    def _backward_fun(self, output, *inputs):
        inputs[0].grad += output.grad.T


class MSELoss(Function):
    """Implement mean squared error loss. For efficiency, we don't create intermediate nodes."""
    def _forward(self, yhat, y):
        n = y.shape[0]
        err = yhat.data - y.data
        res = err.T @ err / n
        self._context['err_over_n'] = err / n
        return Tensor(res)

    def _backward_fun(self, output, *inputs):
        inputs[0].grad += 2 * self._context['err_over_n']
        inputs[1].grad -= 2 * self._context['err_over_n']


class Layer(Module):
    def __init__(self, n_in: int, n_out: int):
        super(Layer, self).__init__()

        self.n_in = n_in
        self.n_out = n_out

    def _backward_fun(self, output, *inputs):
        raise Exception('This is a composite operator and should not have any backward function, you probably '
                        'implemented _forward() instead of forward().')


class LinearLayer(Layer):
    def __init__(self, n_in: int, n_out: int):
        super(LinearLayer, self).__init__(n_in, n_out)

        self.W = Tensor(torch.randn(self.n_out, self.n_in))
        self.b = Tensor(torch.randn(self.n_out))

    def forward(self, x):
        return self.W @ x + self.b

    def _params(self):
        return [self.W, self.b]
