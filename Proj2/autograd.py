

from __future__ import annotations

from typing import List, Tuple, Union

import torch


class Tensor:
    """Wrap torch.tensor class by adding gradients, backward functions and storing parent Tensors."""

    parents: List[Tensor]
    _name = None

    def __init__(self, data, name=None):
        # Preprocess input data: make it a torch tensor of floats (reshape to matrix if data.dim() < 2).
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data)

        if data.dtype is not torch.float32:
            data = data.to(torch.float32)

        # Work only with matrices
        if data.dim() < 2:
            data = data.reshape(-1, 1)

        self._name = name
        self.data = data
        self.grad = None
        self.backward_fun = lambda: ()
        self.parents = []
        self.zero_grad()

    def zero_grad(self) -> None:
        # TODO shoud really clear parents here?
        self.parents = []
        self.grad = torch.zeros_like(self.data)

    def backward(self) -> None:
        # Initialize the root adjoint variable: gradient wrt to itself -> 1 (scalar)
        self.grad = torch.tensor([[1.0]])
        self.backward_fun()

        # Walk through the graph backwards
        # TODO: currently only works for sequential (parallel branches in graph could not work)
        queue = self.parents.copy()
        #print('walking backward through graph')
        for p in queue:
            # if p._name is not None: # TODO remove this
            #     print(f'process {p._name}')
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
        return f'Tensor({self.data}, grad={self.grad}{name}, pnum={len(self.parents)})'


class Module(object):
    _name = 'name for debugging, redefine this in subclasses'

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

    def backward(self, output, *inputs) -> None:
        """Backward function stored in tensors that are results of an operation.
        The gradient of the 'parent' tensors will be modified in place.

        Example: say c = f(a, b), with a,b,c tensors and f an operator. Then the backward function of f will be stored
        in c and will modify in place the gradient of tensors a and b."""
        self._backward(output, *inputs)

    def _backward(self, output, *inputs) -> None:
        """Subclass this method for elementary operations."""
        raise NotImplementedError

    def params(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Return a list of tuples for each parameter: (tensor, gradient)."""
        return [(e.data, e.grad) for e in self._params()]

    def _params(self) -> List[Tensor]:
        """Subclass this method to return a list of Tensors"""
        return []

    def zero_grad(self):
        for p in self._params():
            p.zero_grad()

    def step(self, lr):
        """Make a small step in the gradient descending direction of each parameter tensor."""
        for p in self._params():
            p.data -= lr * p.grad

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

    def _backward(self, output, *inputs):
        inputs[0].grad += output.grad
        inputs[1].grad += output.grad


class Sub(Function):
    _name = 'sub'

    def _forward(self, a, b):
        return Tensor(a.data - b.data)

    def _backward(self, output, *inputs):
        inputs[0].grad += output.grad
        inputs[1].grad -= output.grad


class Mul(Function):
    _name = 'mul'

    def _forward(self, a, b):
        return Tensor(a.data * b.data)

    def _backward(self, output, *inputs):
        inputs[0].grad += output.grad * inputs[1].data
        inputs[1].grad += output.grad * inputs[0].data


class MatMul(Function):
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
    _name = 'sum'

    def _forward(self, a):
        return Tensor(a.data.sum())

    def _backward(self, output, *inputs):
        inputs[0].grad += output.grad * torch.ones_like(inputs[0].data)


class Dot(Function):
    _name = 'dot'

    def _forward(self, a, b):
        """a, b are both 1D tensors"""
        return Tensor(a.data.dot(b.data))

    def _backward(self, output, *inputs):
        a, b = inputs
        a.grad += output.grad * b.data
        b.grad += output.grad * a.data


class ReLU(Function):
    _name = 'relu'

    def _forward(self, x) -> Tensor:
        return Tensor((x.data > 0.0) * x.data)

    def _backward(self, output, *inputs) -> None:
        # input and output have the same shape, simply multiply element-wise
        inputs[0].grad += output.grad * (inputs[0].data > 0.0)


class Transpose(Function):
    _name = 'transpose'

    def _forward(self, a):
        return Tensor(a.data.T)

    def _backward(self, output, *inputs):
        inputs[0].grad += output.grad.T


class MSELoss(Function):
    """Implement mean squared error loss. For efficiency, we don't create intermediate nodes."""
    def _forward(self, yhat, y):
        n = y.shape[0]
        err = yhat.data - y.data
        res = err.T @ err / n
        self._context['err_over_n'] = err / n
        return Tensor(res)

    def _backward(self, output, *inputs):
        inputs[0].grad += 2 * self._context['err_over_n']
        inputs[1].grad -= 2 * self._context['err_over_n']


class Layer(Module):
    """Layers defined as a composition of elementary operations.
    Must reimplement forward (not _forward) and _params (not params) if any."""
    def __init__(self, n_in: int, n_out: int):
        super(Layer, self).__init__()

        self.n_in = n_in
        self.n_out = n_out

    def _forward(self, *inputs) -> Tensor:
        raise Exception('This is a composite operator and should not have and _forward function.')

    def _backward(self, output, *inputs):
        raise Exception('This is a composite operator and should not have any backward function, you probably '
                        'implemented _forward() instead of forward().')


class LinearLayer(Layer):
    _name = 'Linear'

    def __init__(self, n_in: int, n_out: int, xavier_init: bool = True):
        super(LinearLayer, self).__init__(n_in, n_out)

        self.W = torch.randn(self.n_out, self.n_in)
        self.b = torch.randn(self.n_out)

        if xavier_init:
            std = 2 / (self.n_in + self.n_out)
            self.W.normal_(0, std)
            self.b.normal_(0, std)

        self.W = Tensor(self.W, 'W')
        self.b = Tensor(self.b, 'b')

    def forward(self, x):
        # # Handle single input vs minibatch
        # print(self)
        # if x.shape[1] > 1:
        #     return x @ self.W.T + self.b
        return self.W @ x + self.b

    def _params(self):
        return [self.W, self.b]

    def __repr__(self):
        return f'{self._name}({self.n_in}, {self.n_out})'


class Sequential(Layer):
    def __init__(self, *layers: Union[Layer, Function]):
        self.layers = layers
        n_in = self.layers[0].n_in

        n_out = self.layers[-1].n_out

        super(Sequential, self).__init__(n_in, n_out)

        # Add names for debugging
        for i, l in enumerate(self.layers, 1):
            if isinstance(l, LinearLayer):
                l._name = f'FC{i}'
                for p in l._params():
                    p._name = f'{p._name}{i}'
            else:
                l._name = f'{l._name}{i}'

    def forward(self, x) -> Tensor:
        for l in self.layers:
            x = l(x)
        return x

    def _params(self) -> List[Tensor]:
        return [
            p for l in self.layers for p in l._params()
        ]

    def __repr__(self):
        layers = ', '.join(str(l) for l in self.layers)
        return f'Sequential({layers})'
