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
