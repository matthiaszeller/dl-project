

from typing import List, Union, Tuple

import torch

from tensor import Tensor


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
    def __init__(self, *layers):
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

