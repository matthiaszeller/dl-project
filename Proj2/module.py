"""
Implement the generic Module superclass, and the LinearLayer class.
"""

from typing import List, Tuple

from torch import Tensor as TorchTensor
from torch import empty

from tensor import Tensor


class Module(object):
    _name = 'name for debugging, redefine this in subclasses'

    def __init__(self):
        # Context used by some modules to store intermediate results
        self._context = dict()

    def forward(self, *inputs) -> Tensor:
        """Superclass method wrapping forward computation.
        This method must **not** be reimplemented for elementary operations (subclass _forward instead).
        This method **must** be reimplemented for operations that are a combination of elementary operations,
        (e.g. linear layer).

        Calling a Module's forward function is equivalent to directly call the
        """
        # Do actual forward computation
        res = self._forward(*inputs)
        # Initialize the backward function
        res.backward_fun = lambda: self.backward(res, *inputs)

        # Make resulting computation aware of the inputs that created it
        for i in inputs:
            res.parents.append(i)

        return res

    def _forward(self, *inputs) -> Tensor:
        """Actual forward computation for elementary operations (i.e., reimplement this function for those modules)."""
        raise NotImplementedError

    def backward(self, output, *inputs) -> None:
        """Backward function stored in tensors that are results of an operation.
        The gradient of the 'parent' tensors will be modified in place.

        Example: say c = f(a, b), with a,b,c tensors and f an operator. Then the backward function of f will be stored
        in c and will modify in place the gradient of tensors a and b."""
        self._backward(output, *inputs)

    def _backward(self, output, *inputs) -> None:
        """Reimplement this method for elementary operations."""
        raise NotImplementedError

    def params(self) -> List[Tuple[TorchTensor, TorchTensor]]:
        """Return a list of tuples for each parameter: (tensor, gradient)."""
        return [(e.data, e.grad) for e in self._params()]

    def _params(self) -> List[Tensor]:
        """Reimplement this method to return a list of Tensors"""
        return []

    def zero_grad(self) -> None:
        """Resets the gradients of all parameters."""
        for p in self._params():
            p.zero_grad()

    def step(self, lr: float) -> None:
        """Make a small step in the gradient descending direction of each parameter tensor.

        :param lr: learning rate (step size)
        """
        for p in self._params():
            p.data -= lr * p.grad

    def __call__(self, *args, **kwargs) -> Tensor:
        return self.forward(*args)

    def __repr__(self) -> str:
        return self._name


class Layer(Module):
    """
    Layers define a composition of elementary operations.
    Must reimplement forward (not _forward) and _params (not params) if any.
    """
    def __init__(self, n_in: int, n_out: int):
        """
        :param n_in: number of input features
        :param n_out: number of output features
        """
        super(Layer, self).__init__()

        self.n_in = n_in
        self.n_out = n_out

    def _forward(self, *inputs) -> Tensor:
        raise Exception('This is a composite operator and should not have any _forward function.')

    def _backward(self, output, *inputs) -> None:
        raise Exception('This is a composite operator and should not have any backward function, you probably '
                        'implemented _forward() instead of forward().')


class LinearLayer(Layer):
    """
    Linear layer with Xavier weight initialization.
    """
    _name = 'Linear'

    def __init__(self, n_in: int, n_out: int, xavier_init: bool = True):
        """
        :param n_in: number of input features
        :param n_out: number of output features
        :param xavier_init: use Xavier weight initialization if True, or standard normal distribution otherwise
        """
        super(LinearLayer, self).__init__(n_in, n_out)

        std = 1.0
        if xavier_init:
            std = 2 / (self.n_in + self.n_out)

        self.W = empty(self.n_out, self.n_in).normal_(0, std)
        self.b = empty(self.n_out).normal_(0, std)

        self.W = Tensor(self.W, 'W')
        self.b = Tensor(self.b, 'b')

    def forward(self, x) -> Tensor:
        return self.W @ x + self.b

    def _params(self) -> List[Tensor]:
        return [self.W, self.b]

    def __repr__(self) -> str:
        return f'{self._name}({self.n_in}, {self.n_out})'


class Sequential(Layer):
    """Sequence of modules"""
    def __init__(self, *layers):
        """
        :param layers: list of modules, the first module is the input layer, the last module is the output layer
        """
        # Retrieve the number of input and output features of the sequence of modules as a whole
        self.layers = layers
        n_in = self.layers[0].n_in
        n_out = self.layers[-1].n_out

        super(Sequential, self).__init__(n_in, n_out)

        # Name layers and parameters for debugging
        for i, l in enumerate(self.layers, 1):
            if isinstance(l, LinearLayer):
                l._name = f'FC{i}'
                for p in l._params():
                    p._name = f'{p._name}{i}'
            else:
                l._name = f'{l._name}{i}'

    def forward(self, x: Tensor) -> Tensor:
        """
        Sequentially feed the output of a module to the next module, starting with input `x` for the first module.

        :param x: input of the first module
        :return: output of the last layer
        """
        for l in self.layers:
            x = l(x)
        return x

    def _params(self) -> List[Tensor]:
        """Returns a list of parameters, flattened across layers."""
        return [
            p for l in self.layers for p in l._params()
        ]

    def add_layer(self, layer: Module):
        self.layers += (layer, )
        # if layer doesn't have n_out, it doesn't change the dimension
        if hasattr(layer, 'n_out'):
            self.n_out = layer.n_out

    def __repr__(self) -> str:
        layers = ', '.join(str(l) for l in self.layers)
        return f'Sequential({layers})'

