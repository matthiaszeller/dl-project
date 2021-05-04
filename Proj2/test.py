

import setup
import torch
from variable import *


a = Tensor(torch.tensor(1))
b = Tensor(torch.tensor(2))

c = a * b

c.backward()
print(a, b)

a = Tensor(torch.tensor([1,2,3]))
b = a.sum()
b.backward()

