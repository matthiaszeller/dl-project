"""Import of functions is done at the end of the module to avoid errors with circular dependencies."""

from __future__ import annotations

from typing import List

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
        return F.Add()(self, other)

    def __sub__(self, other):
        return F.Sub()(self, other)

    def __mul__(self, other) -> Tensor:
        return F.Mul()(self, other)

    def __matmul__(self, other) -> Tensor:
        return F.MatMul()(self, other)

    def sum(self) -> Tensor:
        """Sum over all axes."""
        return F.Sum()(self)

    def dot(self, other) -> Tensor:
        return F.Dot()(self, other)

    @property
    def T(self) -> Tensor:
        return self.transpose()

    def transpose(self) -> Tensor:
        return F.Transpose()(self)

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


# This is at the end to avoid circular dependencies
import function as F
