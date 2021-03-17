from functools import reduce

import torch
import torch.nn.functional as F
from torch import nn


class FullyDenseNet(nn.Module):
    def __init__(self):
        super(FullyDenseNet, self).__init__()

        self.fc1 = nn.Linear(2 * 14 * 14, 300)
        self.fc2 = nn.Linear(300, 200)
        self.fc3 = nn.Linear(200, 100)
        self.fc4 = nn.Linear(100, 50)
        self.fc5 = nn.Linear(50, 1)

    def forward(self, x):
        x = nn.Flatten(1)(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return torch.sigmoid(x)


class CNN_model1(nn.Module):
    def __init__(self):
        super(CNN_model1, self).__init__()
        self.conv1 = nn.Conv2d(2, 4, kernel_size=3)
        self.conv2 = nn.Conv2d(4, 6, kernel_size=3)

        self.fc1 = nn.Linear(6 * 10 * 10, 100)
        self.fc2 = nn.Linear(100, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))

        x = nn.Flatten(1)(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)

        return torch.sigmoid(x)


class CNN_model2(nn.Module):
    def __init__(self):
        super(CNN_model2, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3)
        self.conv2 = nn.Conv2d(4, 6, kernel_size=3)

        self.fc1 = nn.Linear(1200, 100)
        self.fc2 = nn.Linear(100, 1)

    def forward(self, x):
        # I hate this , how do u make it batch independent ? should be like : x_l = x[0,:,:]
        x_l = x[:, 0, :, :]
        x_r = x[:, 1, :, :]

        x_l = x_l.view(x_l.shape[0], 1, 14, 14)
        x_r = x_r.view(x_r.shape[0], 1, 14, 14)

        x_l = torch.relu(self.conv1(x_l))
        x_l = torch.relu(self.conv2(x_l))

        x_r = torch.relu(self.conv1(x_r))
        x_r = torch.relu(self.conv2(x_r))

        x_r = nn.Flatten(1)(x_r)
        x_l = nn.Flatten(1)(x_l)

        x = torch.cat((x_r, x_l), 1)

        x = torch.relu(self.fc1(x))
        x = self.fc2(x)

        return torch.sigmoid(x)


class CNNAux(nn.Module):
    def __init__(self):
        super(CNNAux, self).__init__()

        # Separate convolutional layers for each image
        self.conv1_im1 = nn.Conv2d(1, 4, kernel_size=3) # kernel size 3 => decrease width & height by 2
        self.conv1_im2 = nn.Conv2d(1, 4, kernel_size=3)
        self.conv2_im1 = nn.Conv2d(4, 6, kernel_size=3)
        self.conv2_im2 = nn.Conv2d(4, 6, kernel_size=3)

        self.maxpool2_im1 = nn.MaxPool2d(kernel_size=3)
        self.maxpool2_im2 = nn.MaxPool2d(kernel_size=3)

        # Auxiliary
        self.conv3_im1 = nn.Conv2d(6, 12, kernel_size=3)
        self.conv3_im2 = nn.Conv2d(6, 12, kernel_size=3)
        self.fc4_im1 = nn.Linear(8**2 * 12, 10)
        self.fc4_im2 = nn.Linear(8**2 * 12, 10)

        # Main
        self.conv4 = nn.Conv2d(6 * 2, 6 * 2 * 2, kernel_size=1)
        self.fc5 = nn.Linear((6*2*2) * 10**2, 1) # 6*2*2 channels from conv4

    def forward(self, x):
        im1 = x[:, :1, :, :] # this way of slicing (`:1` instead of `0`) allows to keep dimension 1
        im2 = x[:, 1:, :, :]
        # im1 and im2 are 14x14
        im1 = torch.relu(self.conv1_im1(im1))
        im2 = torch.relu(self.conv1_im1(im2))
        # im1 and im2 are 12x12
        im1 = torch.relu(self.conv2_im1(im1))
        im2 = torch.relu(self.conv2_im2(im2))
        # im1 and im2 are 10x10

        #im1 = self.maxpool2_im1(im1)
        #im2 = self.maxpool2_im2(im2)

        # Main
        common = torch.cat((im1, im2), dim=1)
        common = torch.relu(self.conv4(common))
        common = nn.Flatten()(common)
        common = torch.relu(self.fc5(common))
        common = torch.sigmoid(common)

        # Auxiliary
        im1 = torch.relu(self.conv3_im1(im1))
        im2 = torch.relu(self.conv3_im2(im2))
        im1 = nn.Flatten()(im1)
        im2 = nn.Flatten()(im2)
        im1 = torch.relu(self.fc4_im1(im1))
        im2 = torch.relu(self.fc4_im2(im2))

        im1 = torch.softmax(im1, dim=0)
        im2 = torch.softmax(im2, dim=0)

        return common, im1, im2


class FullyDenseNetAux(nn.Module):
    def __init__(self):
        super(FullyDenseNetAux, self).__init__()

        # Network basis: common for all losses
        # 14 * 14 = 196
        self.fc1_im1 = nn.Linear(14 * 14, 100)
        self.fc1_im2 = nn.Linear(14 * 14, 100)

        self.fc2_im1 = nn.Linear(100, 50)
        self.fc2_im2 = nn.Linear(100, 50)

        # Auxiliary networks
        self.fc3_im1 = nn.Linear(50, 10)
        self.fc3_im2 = nn.Linear(50, 10)

        # Main task
        self.fc4 = nn.Linear(2 * 50, 10)
        self.fc5 = nn.Linear(10, 1)

    def forward(self, x):
        im1 = nn.Flatten()(x[:, 0, :, :])
        im2 = nn.Flatten()(x[:, 1, :, :])

        im1 = torch.relu(self.fc1_im1(im1))
        im2 = torch.relu(self.fc1_im2(im2))

        im1 = torch.relu(self.fc2_im1(im1))
        im2 = torch.relu(self.fc2_im2(im2))

        # Main task
        common = torch.cat((im1, im2), dim=1)
        common = torch.relu(self.fc4(common))
        common = self.fc5(common)
        common = torch.sigmoid(common)

        # Auxiliary networks
        im1 = self.fc3_im1(im1)
        im1 = F.softmax(im1, dim=0)

        im2 = self.fc3_im2(im2)
        im2 = F.softmax(im2, dim=0)

        return common, im1, im2


def compute_nb_param(net):
    return reduce(
        lambda acc, shape: acc + reduce(lambda acc, dim: acc * dim, shape, 1),
        (p.shape for p in net.parameters()),
        0
    )
