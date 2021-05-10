import unittest

from autograd import Tensor, MSELoss, LinearLayer, ReLU
import torch


class TestScalarAutograd(unittest.TestCase):

    def test_scalar_sum(self):
        a = Tensor(torch.tensor(2))
        b = Tensor(torch.tensor(3))
        c = a + b
        c.backward()
        self.assertEqual(a.grad, 1.0)
        self.assertEqual(b.grad, 1.0)

    def test_scalar_mul(self):
        a = Tensor(torch.tensor(2))
        b = Tensor(torch.tensor(3))
        c = a * b
        c.backward()
        self.assertEqual(a.grad, 3.0)
        self.assertEqual(b.grad, 2.0)


class TestFunctions(unittest.TestCase):
    def test_mse(self):
        a = Tensor([1, 2, 3])
        b = Tensor([-1, -1, -1])
        c = MSELoss()(a, b)
        c.backward()

        err = torch.tensor([[2, 3, 4]]).reshape(-1, 1)
        self.assertEqual(c.data, (err**2).sum() / 3)
        self.assertTrue(torch.equal(a.grad, 2/3*err))
        self.assertTrue(torch.equal(b.grad, -2/3*err))

    def test_relu(self):
        a = Tensor([-3.3, 1.0, -0.5, 0.0, 10])
        b = ReLU()(a)
        expected = torch.tensor([.0, 1.0, .0, .0, 10]).reshape(-1, 1)
        self.assertTrue(torch.equal(b.data, expected))

        c = b.sum()
        self.assertEqual(c.item(), 11)
        c.backward()
        agrad = torch.tensor([0., 1, 0, 0, 1]).reshape(-1, 1)
        self.assertTrue(torch.equal(a.grad, agrad))


class TestLayer(unittest.TestCase):
    def test_linear_layer(self):
        layer = LinearLayer(3, 1)
        x = Tensor(torch.tensor([1, 2, 3]))
        layer.W = Tensor([[1, -1, 2]])
        layer.b = Tensor(2)
        y = layer(x)

        self.assertEqual(y.item(), 7)


class TestMatrixAutograd(unittest.TestCase):
    def test_1layer_mse(self):
        layer = LinearLayer(3, 1)
        x = Tensor(torch.tensor([1, 2, 3]))
        layer.W = Tensor([[1, -1, 2]])
        layer.b = Tensor(2)

        target = Tensor(1)
        y = layer(x)
        l = MSELoss()(y, target)
        l.backward()

        self.assertEqual(l.item(), 36.)
        self.assertTrue(torch.equal(x.grad, torch.tensor([[12.], [-12.], [24.]])))
        self.assertTrue(torch.equal(layer.W.grad, torch.tensor([[12., 24., 36.]])))
        self.assertEqual(layer.b.grad.item(), 12.)


if __name__ == '__main__':
    unittest.main()
