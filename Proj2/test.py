

import setup
import torch
from variable import *


# a = Tensor(torch.tensor([-1,-1,-1]))
# b = Tensor(torch.tensor([1,2,3]))
#
# c = a.dot(b)
# c.backward()
#
#
# b = Tensor(torch.tensor([-1,-1,-1]))
# a = Tensor(torch.tensor([1,2,3]))
# c = Tensor(torch.tensor([0, 1, 0]))
# d = a - c
# e = MSELoss()(b, d)
# e.backward()




layer = LinearLayer(3, 1)
x = Tensor(torch.tensor([1, 2, 3]).reshape(-1, 1))
layer.W = Tensor([[1, -1, 2]])
layer.b = Tensor(2)

target = Tensor(1)
y = layer(x)
l = MSELoss()(y, target)

x._name = 'x'
y._name = 'y'
l._name = 'l'
layer.W._name = 'W'
layer.b._name = 'b'
l.backward()
