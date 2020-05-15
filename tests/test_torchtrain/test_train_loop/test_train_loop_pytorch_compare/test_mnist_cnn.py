import unittest

import os
import shutil
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from aitoolbox import TrainLoop, TTModel

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


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


class TestMNISTCNN(unittest.TestCase):
    def test_trainloop_core_pytorch_compare(self):
        loss_tl, y_pred_tl, y_true_tl = self.train_eval_trainloop(num_epochs=5)
        loss_pt, y_pred_pt, y_true_pt = self.train_eval_core_pytorch(num_epochs=5)

        self.assertEqual(loss_tl, loss_pt)
        self.assertEqual(y_pred_tl, y_pred_pt)
        self.assertEqual(y_true_tl, y_true_pt)

        # project_path = os.path.join(THIS_DIR, 'data')
        # if os.path.exists(project_path):
        #     shutil.rmtree(project_path)

    def train_eval_trainloop(self, num_epochs):
        self.set_seeds()
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./data', train=False, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=100, shuffle=True, num_workers=2)
        val_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])),
            batch_size=100, num_workers=2)

        self.set_seeds()
        model = CNNNet()
        optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
        criterion = nn.NLLLoss()

        print('Starting train loop')
        tl = TrainLoop(
            model,
            train_loader, val_loader, None,
            optimizer, criterion
        )
        tl.fit(num_epochs=num_epochs)

        loss = tl.evaluate_loss_on_validation_set(force_prediction=True)
        y_pred, y_true, _ = tl.predict_on_validation_set(force_prediction=True)

        return loss, y_pred.tolist(), y_true.tolist()

    def train_eval_core_pytorch(self, num_epochs):
        self.set_seeds()
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./data', train=False, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=100, shuffle=True, num_workers=2)
        val_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])),
            batch_size=100, num_workers=2)

        self.set_seeds()
        model = CNNNet()
        optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
        criterion = nn.NLLLoss()

        model.train()
        for epoch in range(num_epochs):
            print(f'Epoch: {epoch}')
            for i, (input_data, target) in enumerate(train_loader):
                predicted = model(input_data)
                loss = criterion(predicted, target)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        print('Evaluating')
        val_pred, val_true, val_loss = [], [], []
        model.eval()
        with torch.no_grad():
            for input_data, target in val_loader:
                predicted = model(input_data)
                loss_batch = criterion(predicted, target).item()
                val_loss.append(loss_batch)
                val_pred += predicted.argmax(dim=1, keepdim=False).tolist()
                val_true += target.tolist()
            val_loss = np.mean(val_loss)

        return val_loss, val_pred, val_true

    @staticmethod
    def set_seeds():
        manual_seed = 0
        np.random.seed(manual_seed)
        random.seed(manual_seed)
        torch.manual_seed(manual_seed)
        # if you are suing GPU
        torch.cuda.manual_seed(manual_seed)
        torch.cuda.manual_seed_all(manual_seed)

        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
