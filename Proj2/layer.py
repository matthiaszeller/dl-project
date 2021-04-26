

import torch

from function import ActivationFunction
from module import Module


class Layer(Module):

    def __init__(self, n_in: int, n_out: int, activation_fun: ActivationFunction):
        super(Layer, self).__init__()

        self._activation_fun = activation_fun
        # Elements to store during forward pass for the backward pass
        self._input_previous_layer = None
        self._z = None

    def forward(self, *input):
        # TODO what about several inputs ???
        if len(input) > 1:
            raise ValueError
        input = input[0]

        # Store input for backward pass
        self._input_previous_layer = input
        # Compute preactivation & store for backward pass
        self._z = self._compute_preactivation(input)
        # Apply activation function
        x = self._activation_fun(self._z)

        return x

    def backward(self, *gradwrtoutput):
        # Hadamard product to compute gradient wrt preactivation (z)
        grad_wrt_z = self._activation_fun.backward(self._z) * gradwrtoutput
        return grad_wrt_z

    def _compute_preactivation(self, x):
        pass


class LinearLayer(Layer):

    def __init__(self, n_in: int, n_out: int, activation_fun: ActivationFunction):
        super(LinearLayer, self).__init__(n_in, n_out, activation_fun)

        # Initialize weight matrix and bias
        # TODO: project PDF says "only allowed to import torch.empty", what about random initalization?
        self.W = torch.randn(n_out, n_in)
        self.b = torch.randn(n_out)
        self.gradW = None
        self.gradb = None

    def param(self):
        return [self.gradW, self.gradb]

    def backward(self, *gradwrtoutput):
        grad_wrt_z = super(LinearLayer, self).backward(*gradwrtoutput)

        self.gradb = grad_wrt_z
        self.gradW = grad_wrt_z @ self._input_previous_layer.T

        grad_wrt_prevlayer = self.W.T @ grad_wrt_z

        return grad_wrt_prevlayer

    def _compute_preactivation(self, x):
        return self.W @ x + self.b

