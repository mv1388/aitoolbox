import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from aitoolbox import TrainLoop, TTModel


class CNNNet(TTModel):
    def __init__(self):
        super(CNNNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

    def get_loss(self, batch_data, criterion, device):
        data, target = batch_data
        data, target = data.to(device), target.to(device)

        output = self(data)
        loss = criterion(output, target)

        return loss

    def get_predictions(self, batch_data, device):
        data, y_test = batch_data
        data = data.to(device)

        output = self(data)
        y_pred = output.argmax(dim=1, keepdim=False)

        return y_pred.cpu(), y_test, {}


class TestMnistCNN(unittest.TestCase):
    def test_mnist_cnn(self):
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=100, shuffle=True, num_workers=4)
        val_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])),
            batch_size=100, num_workers=4)

        model = CNNNet()
        optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
        criterion = nn.NLLLoss()

        print('Starting train loop')
        TrainLoop(
            model,
            train_loader, val_loader, None,
            optimizer, criterion
        ).fit(num_epochs=5)
