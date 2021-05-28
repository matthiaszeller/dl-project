

# --------------------------------------------------------- #
#                          IMPORTS                          #
# --------------------------------------------------------- #

import torch
import torch.nn.functional as F
import torch.nn as nn


# --------------------------------------------------------- #
#                     MODELS DEFINITION                     #
# --------------------------------------------------------- #

# different FCNN structure (Fully Connected Neural Network)
#
# kwargs["nodes_nb"] is a parameter for the constructor of the model
# to get different number of nodes
class FCNN_1LAYER(nn.Module):
    def __init__(self, **kwargs):
        super(FCNN_1LAYER, self).__init__()
        nb_n = kwargs["nodes_nb"]
        self.input = nn.Linear(2 * 14 * 14, nb_n)
        self.fc1 = nn.Linear(nb_n, 1)

    def forward(self, x):
        x = nn.Flatten(1)(x)
        x = torch.relu(self.input(x))
        x = self.fc1(x)

        return torch.sigmoid(x)


class FCNN_2LAYER(nn.Module):
    def __init__(self, **kwargs):
        super(FCNN_2LAYER, self).__init__()
        self.input = nn.Linear(2 * 14 * 14, 300)
        self.fc1 = nn.Linear(300, 50)
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        x = nn.Flatten(1)(x)
        x = torch.relu(self.input(x))
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)

        return torch.sigmoid(x)


class FCNN_3LAYER(nn.Module):
    def __init__(self, **kwargs):
        super(FCNN_3LAYER, self).__init__()
        self.input = nn.Linear(2 * 14 * 14, 300)
        self.fc1 = nn.Linear(300, 150)
        self.fc2 = nn.Linear(150, 50)
        self.fc3 = nn.Linear(50, 1)

    def forward(self, x):
        x = nn.Flatten(1)(x)
        x = torch.relu(self.input(x))
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)

        return torch.sigmoid(x)


class FCNN_4LAYER(nn.Module):
    def __init__(self, **kwargs):
        super(FCNN_4LAYER, self).__init__()

        self.input = nn.Linear(2 * 14 * 14, 300)
        self.fc1 = nn.Linear(300, 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, 50)
        self.fc4 = nn.Linear(50, 1)

    def forward(self, x):
        x = nn.Flatten(1)(x)
        x = torch.relu(self.input(x))
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)

        return torch.sigmoid(x)


class FCNN3(nn.Module):
    def __init__(self, **kwargs):
        super(FCNN3, self).__init__()

        nb_n = kwargs["nodes_nb"]

        self.inputR = nn.Linear(14 * 14, nb_n)
        self.inputL = nn.Linear(14 * 14, nb_n)
        self.fc1R = nn.Linear(nb_n, 10)
        self.fc1L = nn.Linear(nb_n, 10)
        self.fc2 = nn.Linear(20, 15)
        self.fc3 = nn.Linear(15, 1)

    def forward(self, x):
        x_l = x[:, 0, :, :]
        x_r = x[:, 1, :, :]

        x_l = x_l.view(x_l.shape[0], 1, 14, 14)
        x_r = x_r.view(x_r.shape[0], 1, 14, 14)

        x_l = torch.relu(self.inputL(nn.Flatten(1)(x_l)))
        x_l = torch.relu(self.fc1L(x_l))

        x_r = torch.relu(self.inputR(nn.Flatten(1)(x_r)))
        x_r = torch.relu(self.fc1R(x_r))

        x = torch.relu(self.fc2(torch.cat([x_l, x_r], 1)))
        x = self.fc3(x)
        return torch.sigmoid(x)


# different WS structure ( Weight Sharing )
#
# kwargs["nodes_nb"] is a parameter for the constructor of the model
# to get different number of nodes
class WS2(nn.Module):
    def __init__(self, **kwargs):
        super(WS2, self).__init__()

        nb_n = kwargs["nodes_nb"]

        self.input = nn.Linear(14 * 14, nb_n)
        self.fc1 = nn.Linear(nb_n, 10)

        self.fc2 = nn.Linear(20, 15)
        self.fc3 = nn.Linear(15, 1)

    def forward(self, x):
        x_l = x[:, 0, :, :]
        x_r = x[:, 1, :, :]

        x_l = x_l.view(x_l.shape[0], 1, 14, 14)
        x_r = x_r.view(x_r.shape[0], 1, 14, 14)

        x_l = torch.relu(self.input(nn.Flatten(1)(x_l)))
        x_l = torch.relu(self.fc1(x_l))

        x_r = torch.relu(self.input(nn.Flatten(1)(x_r)))
        x_r = torch.relu(self.fc1(x_r))

        x = torch.relu(self.fc2(torch.cat([x_l, x_r], 1)))
        x = self.fc3(x)
        return torch.sigmoid(x)


class WS1(nn.Module):
    def __init__(self, **kwargs):
        super(WS1, self).__init__()

        self.conv1l = nn.Conv2d(1, 5, kernel_size=3)
        self.conv2l = nn.Conv2d(5, 8, kernel_size=3)

        self.fc3l = nn.Linear(800, 200)
        self.fc4l = nn.Linear(200, 10)

        self.conv1r = nn.Conv2d(1, 5, kernel_size=3)
        self.conv2r = nn.Conv2d(5, 8, kernel_size=3)

        self.fc3r = nn.Linear(800, 200)
        self.fc4r = nn.Linear(200, 10)

        self.fc5 = nn.Linear(20, 15)
        self.fc6 = nn.Linear(15, 1)

    def forward(self, x):
        x_l = x[:, 0, :, :]
        x_r = x[:, 1, :, :]

        x_l = x_l.view(x_l.shape[0], 1, 14, 14)
        x_r = x_r.view(x_r.shape[0], 1, 14, 14)

        x_l = F.relu(self.conv1l(x_l))
        x_r = F.relu(self.conv1r(x_r))

        x_l = F.relu(self.conv2l(x_l))
        x_r = F.relu(self.conv2r(x_r))

        x_l = nn.Flatten(1)(x_l)
        x_r = nn.Flatten(1)(x_r)

        # print(x1.shape)
        x_l = F.relu(self.fc3l(x_l))
        x_r = F.relu(self.fc3r(x_r))

        x_l = self.fc4l(x_l)
        x_r = self.fc4r(x_r)

        x = F.relu(self.fc5(torch.cat((
            F.relu(x_l),
            F.relu(x_r)
        ), 1)))
        x = self.fc6(x)

        return torch.sigmoid(x)


class WS3(nn.Module):
    def __init__(self, **kwargs):
        super(WS3, self).__init__()

        self.conv1 = nn.Conv2d(1, 5, kernel_size=3)
        self.conv2 = nn.Conv2d(5, 8, kernel_size=3)

        self.fc3 = nn.Linear(800, 200)
        self.fc4 = nn.Linear(200, 10)

        self.fc5 = nn.Linear(20, 15)
        self.fc6 = nn.Linear(15, 1)

    def forward(self, x):
        x_l = x[:, 0, :, :]
        x_r = x[:, 1, :, :]

        x_l = x_l.view(x_l.shape[0], 1, 14, 14)
        x_r = x_r.view(x_r.shape[0], 1, 14, 14)

        x_l = F.relu(self.conv1(x_l))
        x_r = F.relu(self.conv1(x_r))

        x_l = F.relu(self.conv2(x_l))
        x_r = F.relu(self.conv2(x_r))

        x_l = nn.Flatten(1)(x_l)
        x_r = nn.Flatten(1)(x_r)

        # print(x1.shape)
        x_l = F.relu(self.fc3(x_l))
        x_r = F.relu(self.fc3(x_r))

        x_l = self.fc4(x_l)
        x_r = self.fc4(x_r)

        x = F.relu(self.fc5(torch.cat((
            F.relu(x_l),
            F.relu(x_r)
        ), 1)))
        x = self.fc6(x)

        return torch.sigmoid(x)


# different AL structure (Auxiliary loss)
#
# kwargs["nodes_nb"] is a parameter for the constructor of the model
# to get different number of nodes
class AL1(nn.Module):
    def __init__(self, **kwargs):
        super(AL1, self).__init__()

        nb_n = kwargs["nodes_nb"]

        self.inputR = nn.Linear(14 * 14, nb_n)
        self.inputL = nn.Linear(14 * 14, nb_n)
        self.fc1R = nn.Linear(nb_n, 10)
        self.fc1L = nn.Linear(nb_n, 10)
        self.fc2 = nn.Linear(20, 15)
        self.fc3 = nn.Linear(15, 1)

    def forward(self, x):
        x_l = x[:, 0, :, :]
        x_r = x[:, 1, :, :]

        x_l = x_l.view(x_l.shape[0], 1, 14, 14)
        x_r = x_r.view(x_r.shape[0], 1, 14, 14)

        x_l = torch.relu(self.inputL(nn.Flatten(1)(x_l)))
        x_l = self.fc1L(x_l)

        x_r = torch.relu(self.inputR(nn.Flatten(1)(x_r)))
        x_r = self.fc1R(x_r)

        x = torch.relu(self.fc2(torch.cat([
            torch.relu(x_l),
            torch.relu(x_r)
        ], 1)))
        x = self.fc3(x)
        return torch.sigmoid(x), x_l, x_r


class AL3(nn.Module):
    def __init__(self, **kwargs):
        super(AL3, self).__init__()

        nb_n = kwargs["nodes_nb"]

        self.input = nn.Linear(14 * 14, nb_n)
        self.fc1 = nn.Linear(nb_n, 10)

        self.fc2 = nn.Linear(20, 15)
        self.fc3 = nn.Linear(15, 1)

    def forward(self, x):
        x_l = x[:, 0, :, :]
        x_r = x[:, 1, :, :]

        x_l = x_l.view(x_l.shape[0], 1, 14, 14)
        x_r = x_r.view(x_r.shape[0], 1, 14, 14)

        x_l = torch.relu(self.input(nn.Flatten(1)(x_l)))
        x_l = self.fc1(x_l)

        x_r = torch.relu(self.input(nn.Flatten(1)(x_r)))
        x_r = self.fc1(x_r)

        x = torch.relu(self.fc2(torch.cat([
            torch.relu(x_l),
            torch.relu(x_r)
        ], 1)))
        x = self.fc3(x)

        return torch.sigmoid(x), x_l, x_r


class AL2(nn.Module):
    def __init__(self, **kwargs):
        super(AL2, self).__init__()

        self.conv1l = nn.Conv2d(1, 5, kernel_size=3)
        self.conv2l = nn.Conv2d(5, 8, kernel_size=3)

        self.fc3l = nn.Linear(800, 200)
        self.fc4l = nn.Linear(200, 10)

        self.conv1r = nn.Conv2d(1, 5, kernel_size=3)
        self.conv2r = nn.Conv2d(5, 8, kernel_size=3)

        self.fc3r = nn.Linear(800, 200)
        self.fc4r = nn.Linear(200, 10)

        self.fc5 = nn.Linear(20, 15)
        self.fc6 = nn.Linear(15, 1)

    def forward(self, x):
        x_l = x[:, 0, :, :]
        x_r = x[:, 1, :, :]

        x_l = x_l.view(x_l.shape[0], 1, 14, 14)
        x_r = x_r.view(x_r.shape[0], 1, 14, 14)

        x_l = F.relu(self.conv1l(x_l))
        x_r = F.relu(self.conv1r(x_r))

        x_l = F.relu(self.conv2l(x_l))
        x_r = F.relu(self.conv2r(x_r))

        x_l = nn.Flatten(1)(x_l)
        x_r = nn.Flatten(1)(x_r)

        # print(x1.shape)
        x_l = F.relu(self.fc3l(x_l))
        x_r = F.relu(self.fc3r(x_r))

        x_l = self.fc4l(x_l)
        x_r = self.fc4r(x_r)

        x = F.relu(self.fc5(torch.cat((
            F.relu(x_l),
            F.relu(x_r)
        ), 1)))
        x = self.fc6(x)

        return torch.sigmoid(x), x_l, x_r


class AL4(nn.Module):
    ## lr = 0.001
    def __init__(self, **kwargs):
        super(AL4, self).__init__()

        self.conv1 = nn.Conv2d(1, 5, kernel_size=3)
        self.conv2 = nn.Conv2d(5, 8, kernel_size=3)

        self.fc3 = nn.Linear(800, 200)
        self.fc4 = nn.Linear(200, 10)

        self.fc5 = nn.Linear(20, 15)
        self.fc6 = nn.Linear(15, 1)

    def forward(self, x):
        x_l = x[:, 0, :, :]
        x_r = x[:, 1, :, :]

        x_l = x_l.view(x_l.shape[0], 1, 14, 14)
        x_r = x_r.view(x_r.shape[0], 1, 14, 14)

        x_l = F.relu(self.conv1(x_l))
        x_r = F.relu(self.conv1(x_r))

        x_l = F.relu(self.conv2(x_l))
        x_r = F.relu(self.conv2(x_r))

        x_l = nn.Flatten(1)(x_l)
        x_r = nn.Flatten(1)(x_r)

        # print(x1.shape)
        x_l = F.relu(self.fc3(x_l))
        x_r = F.relu(self.fc3(x_r))

        x_l = self.fc4(x_l)
        x_r = self.fc4(x_r)

        x = F.relu(self.fc5(torch.cat((
            F.relu(x_l),
            F.relu(x_r)
        ), 1)))
        x = self.fc6(x)

        return torch.sigmoid(x), x_l, x_r

