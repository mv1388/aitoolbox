import unittest

import os
import shutil
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from transformers import AdamW

from aitoolbox import TrainLoop, TTModel
from aitoolbox.torchtrain.multi_loss_optim import MultiLoss, MultiOptimizer

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class CNNNet(TTModel):
    def __init__(self):
        super(CNNNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc_pipe = nn.Sequential(*tuple([nn.Linear(128, 128) for _ in range(10)]))

        self.fc2 = nn.Linear(128, 10)

        self.fc1_2 = nn.Linear(9216, 1)
        self.fc2_2 = nn.Linear(1, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.gelu(x)
        x = self.conv2(x)
        x = F.gelu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        cnn_out = torch.flatten(x, 1)

        x = self.fc1(cnn_out)
        x = F.gelu(x)
        x = self.dropout2(x)
        x = self.fc_pipe(x)
        x = self.fc2(x)

        x_2 = self.fc1_2(cnn_out)
        x_2 = F.gelu(x_2)
        x_2 = self.dropout2(x_2)
        x_2 = self.fc2_2(x_2)

        return x, x_2

    def get_loss(self, batch_data, criterion, device):
        data, target = batch_data
        data, target = data.to(device), target.to(device)

        x, x_2 = self(data)
        assert x.dtype is torch.float16
        assert x_2.dtype is torch.float16

        output_1 = F.log_softmax(x, dim=1)
        output_2 = F.log_softmax(x_2, dim=1)
        assert output_1.dtype is torch.float32
        assert output_2.dtype is torch.float32

        loss_1 = criterion(output_1, target)
        loss_2 = criterion(output_2, target)
        assert loss_1.dtype is torch.float32
        assert loss_2.dtype is torch.float32

        loss = MultiLoss({'loss_1': loss_1, 'loss_2': loss_2})

        return loss

    def get_predictions(self, batch_data, device):
        data, y_test = batch_data
        data = data.to(device)

        x, _ = self(data)
        assert x.dtype is torch.float16

        output = F.log_softmax(x, dim=1)
        assert output.dtype is torch.float32

        y_pred = output.argmax(dim=1, keepdim=False)

        return y_pred.cpu(), y_test, {}


class TestAMPMultiLossOptimizerMNISTCNN(unittest.TestCase):
    def test_trainloop_core_pytorch_compare(self):
        torch.autograd.set_detect_anomaly(True)
        val_loss_tl, y_pred_tl, y_true_tl = self.train_eval_trainloop(num_epochs=5, use_real_train_data=True)
        val_loss_pt, y_pred_pt, y_true_pt = self.train_eval_core_pytorch(num_epochs=5, use_real_train_data=True)

        self.assertEqual(val_loss_tl, val_loss_pt)
        self.assertEqual(y_pred_tl, y_pred_pt)
        self.assertEqual(y_true_tl, y_true_pt)

        val_dataset = datasets.MNIST(
            os.path.join(THIS_DIR, 'data'), train=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        )
        self.assertEqual(val_dataset.targets.tolist(), y_true_tl)
        self.assertEqual(val_dataset.targets.tolist(), y_true_pt)

        project_path = os.path.join(THIS_DIR, 'data')
        if os.path.exists(project_path):
            shutil.rmtree(project_path)

    def train_eval_trainloop(self, num_epochs, use_real_train_data=False):
        self.set_seeds()
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(os.path.join(THIS_DIR, 'data'), train=use_real_train_data, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=100, shuffle=True)
        val_loader = torch.utils.data.DataLoader(
            datasets.MNIST(os.path.join(THIS_DIR, 'data'), train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])),
            batch_size=100)

        model = CNNNet()
        optimizer = MultiOptimizer([
            AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999)),
            AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999))
        ])
        criterion = nn.NLLLoss()

        print('Starting train loop')
        tl = TrainLoop(
            model,
            train_loader, val_loader, None,
            optimizer, criterion,
            use_amp=True
        )

        self.assertEqual(tl.device.type, "cuda")

        tl.fit(num_epochs=num_epochs)

        val_loss = tl.evaluate_loss_on_validation_set(force_prediction=True)
        y_pred, y_true, _ = tl.predict_on_validation_set(force_prediction=True)

        return val_loss, y_pred.tolist(), y_true.tolist()

    def train_eval_core_pytorch(self, num_epochs, use_real_train_data=False):
        self.set_seeds()
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(os.path.join(THIS_DIR, 'data'), train=use_real_train_data, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=100, shuffle=True)
        val_loader = torch.utils.data.DataLoader(
            datasets.MNIST(os.path.join(THIS_DIR, 'data'), train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])),
            batch_size=100)

        USE_CUDA = torch.cuda.is_available()
        device = torch.device("cuda" if USE_CUDA else "cpu")
        self.assertEqual(device.type, "cuda")

        model_pt = CNNNet().to(device)
        optimizer_pt_1 = AdamW(model_pt.parameters(), lr=0.001, betas=(0.9, 0.999))
        optimizer_pt_2 = AdamW(model_pt.parameters(), lr=0.001, betas=(0.9, 0.999))
        criterion_pt = nn.NLLLoss()

        scaler = GradScaler()

        model_pt.train()
        for epoch in range(num_epochs):
            print(f'Epoch: {epoch}')
            for i, (input_data, target) in enumerate(train_loader):
                with autocast():
                    input_data = input_data.to(device)
                    target = target.to(device)

                    x_1, x_2 = model_pt(input_data)
                    assert x_1.dtype is torch.float16
                    assert x_2.dtype is torch.float16

                    output_1 = F.log_softmax(x_1, dim=1)
                    output_2 = F.log_softmax(x_2, dim=1)
                    assert output_1.dtype is torch.float32
                    assert output_2.dtype is torch.float32

                    loss_1 = criterion_pt(output_1, target)
                    loss_2 = criterion_pt(output_2, target)
                    assert loss_1.dtype is torch.float32
                    assert loss_2.dtype is torch.float32

                scaler.scale(loss_1).backward(retain_graph=True)
                scaler.step(optimizer_pt_1)
                optimizer_pt_1.zero_grad()

                scaler.scale(loss_2).backward()
                scaler.step(optimizer_pt_2)
                optimizer_pt_2.zero_grad()

                scaler.update()

            # Imitate what happens in auto_execute_end_of_epoch() in TrainLoop
            for _ in train_loader:
                pass
            for _ in val_loader:
                pass

        print('Evaluating')
        val_loss_1, val_loss_2, val_pred, val_true = [], [], [], []
        model_pt.eval()
        with torch.no_grad():
            for input_data, target in val_loader:
                with autocast():
                    input_data = input_data.to(device)
                    target = target.to(device)

                    x_1, x_2 = model_pt(input_data)
                    assert x_1.dtype is torch.float16
                    assert x_2.dtype is torch.float16

                    output_1 = F.log_softmax(x_1, dim=1)
                    output_2 = F.log_softmax(x_2, dim=1)
                    assert output_1.dtype is torch.float32
                    assert output_2.dtype is torch.float32

                    loss_batch_1 = criterion_pt(output_1, target).cpu().item()
                    loss_batch_2 = criterion_pt(output_2, target).cpu().item()
                    predicted_argmax = output_1.argmax(dim=1, keepdim=False).cpu().tolist()

                val_pred += predicted_argmax
                val_true += target.cpu().tolist()
                val_loss_1.append(loss_batch_1)
                val_loss_2.append(loss_batch_2)
            val_loss_1 = np.mean(val_loss_1)
            val_loss_2 = np.mean(val_loss_2)

        return {'loss_1': val_loss_1, 'loss_2': val_loss_2}, val_pred, val_true

    @staticmethod
    def set_seeds():
        manual_seed = 0
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        np.random.seed(manual_seed)
        random.seed(manual_seed)
        torch.manual_seed(manual_seed)
        # if you are suing GPU
        torch.cuda.manual_seed(manual_seed)
        torch.cuda.manual_seed_all(manual_seed)
