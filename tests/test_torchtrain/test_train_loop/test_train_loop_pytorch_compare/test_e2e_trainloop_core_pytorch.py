import unittest

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import TensorDataset

from aitoolbox.torchtrain.train_loop import TrainLoop
from aitoolbox.torchtrain.model import TTModel


class FFNetAIToolbox(TTModel):
    def __init__(self):
        super().__init__()
        self.ff_1 = nn.Linear(50, 100)
        self.ff_2 = nn.Linear(100, 100)
        self.ff_3 = nn.Linear(100, 10)

    def forward(self, batch_data):
        ff_out = F.relu(self.ff_1(batch_data))
        ff_out = F.relu(self.ff_2(ff_out))
        ff_out = self.ff_3(ff_out)
        out_softmax = F.log_softmax(ff_out, dim=1)
        return out_softmax

    def get_loss(self, batch_data, criterion, device):
        input_data, target = batch_data
        input_data = input_data.to(device)
        target = target.to(device)

        predicted = self(input_data)
        loss = criterion(predicted, target)

        return loss

    def get_predictions(self, batch_data, device):
        input_data, target = batch_data
        input_data = input_data.to(device)

        predicted = self(input_data).argmax(dim=1, keepdim=False)

        return predicted.cpu(), target, {}


class FFNetPyTorch(nn.Module):
    def __init__(self):
        super().__init__()
        self.ff_1 = nn.Linear(50, 100)
        self.ff_2 = nn.Linear(100, 100)
        self.ff_3 = nn.Linear(100, 10)

    def forward(self, batch_data):
        ff_out = F.relu(self.ff_1(batch_data))
        ff_out = F.relu(self.ff_2(ff_out))
        ff_out = self.ff_3(ff_out)
        out_softmax = F.log_softmax(ff_out, dim=1)
        return out_softmax


class TestTrainLoopVSCorePyTorch(unittest.TestCase):
    def test_trainloop_core_pytorch_prediction_loss_compare(self):
        batch_size = 50
        num_epochs = 10

        self.set_seeds()
        model_aitb = FFNetAIToolbox()
        optimizer_aitb = optim.Adam(model_aitb.parameters(), lr=0.001, betas=(0.9, 0.999))
        criterion_aitb = nn.NLLLoss()

        self.set_seeds()
        model_pt = FFNetPyTorch()
        optimizer_pt = optim.Adam(model_pt.parameters(), lr=0.001, betas=(0.9, 0.999))
        criterion_pt = nn.NLLLoss()

        train_dataset = TensorDataset(torch.randn(1000, 50), torch.randint(low=0, high=10, size=(1000,)))
        val_dataset = TensorDataset(torch.randn(300, 50), torch.randint(low=0, high=10, size=(300,)))
        test_dataset = TensorDataset(torch.randn(300, 50), torch.randint(low=0, high=10, size=(300,)))
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

        train_loop = TrainLoop(
            model_aitb,
            train_dataloader, val_dataloader, test_dataloader,
            optimizer_aitb, criterion_aitb
        )
        train_loop.fit(num_epochs=num_epochs)

        train_pred_aitb, _, _ = train_loop.predict_on_train_set()
        val_pred_aitb, _, _ = train_loop.predict_on_validation_set()
        test_pred_aitb, _, _ = train_loop.predict_on_test_set()
        train_loss_aitb = train_loop.evaluate_loss_on_train_set()
        val_loss_aitb = train_loop.evaluate_loss_on_validation_set()
        test_loss_aitb = train_loop.evaluate_loss_on_test_set()

        model_pt.train()
        for epoch in range(num_epochs):
            for input_data, target in train_dataloader:
                predicted = model_pt(input_data)
                loss = criterion_pt(predicted, target)
                optimizer_pt.zero_grad()
                loss.backward()
                optimizer_pt.step()

        train_pred_pt, train_loss_pt = [], []
        model_pt.eval()
        with torch.no_grad():
            for input_data, target in train_dataloader:
                predicted = model_pt(input_data)
                loss_batch = criterion_pt(predicted, target).item()
                train_pred_pt += predicted.argmax(dim=1, keepdim=False).tolist()
                train_loss_pt.append(loss_batch)
            train_loss_pt = np.mean(train_loss_pt)

        val_pred_pt, val_loss_pt = [], []
        model_pt.eval()
        with torch.no_grad():
            for input_data, target in val_dataloader:
                predicted = model_pt(input_data)
                loss_batch = criterion_pt(predicted, target).item()
                val_pred_pt += predicted.argmax(dim=1, keepdim=False).tolist()
                val_loss_pt.append(loss_batch)
            val_loss_pt = np.mean(val_loss_pt)

        test_pred_pt, test_loss_pt = [], []
        model_pt.eval()
        with torch.no_grad():
            for input_data, target in test_dataloader:
                predicted = model_pt(input_data)
                loss_batch = criterion_pt(predicted, target).item()
                test_pred_pt += predicted.argmax(dim=1, keepdim=False).tolist()
                test_loss_pt.append(loss_batch)
            test_loss_pt = np.mean(test_loss_pt)

        self.assertEqual(train_pred_aitb.tolist(), train_pred_pt)
        self.assertEqual(val_pred_aitb.tolist(), val_pred_pt)
        self.assertEqual(test_pred_aitb.tolist(), test_pred_pt)

        self.assertEqual(train_loss_aitb, train_loss_pt)
        self.assertEqual(val_loss_aitb, val_loss_pt)
        self.assertEqual(test_loss_aitb, test_loss_pt)

    def test_grad_accumulate_trainloop_core_pytorch_prediction_loss_compare(self):
        num_epochs = 10

        batch_size = 10
        grad_accumulation = 5

        self.set_seeds()
        model_aitb = FFNetAIToolbox()
        optimizer_aitb = optim.Adam(model_aitb.parameters(), lr=0.001, betas=(0.9, 0.999))
        criterion_aitb = nn.NLLLoss()

        self.set_seeds()
        model_pt = FFNetPyTorch()
        optimizer_pt = optim.Adam(model_pt.parameters(), lr=0.001, betas=(0.9, 0.999))
        criterion_pt = nn.NLLLoss()

        train_dataset = TensorDataset(torch.randn(1000, 50), torch.randint(low=0, high=10, size=(1000,)))
        val_dataset = TensorDataset(torch.randn(300, 50), torch.randint(low=0, high=10, size=(300,)))
        test_dataset = TensorDataset(torch.randn(300, 50), torch.randint(low=0, high=10, size=(300,)))
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

        train_loop = TrainLoop(
            model_aitb,
            train_dataloader, val_dataloader, test_dataloader,
            optimizer_aitb, criterion_aitb
        )
        train_loop.fit(num_epochs=num_epochs, grad_accumulation=grad_accumulation)

        train_pred_aitb, _, _ = train_loop.predict_on_train_set()
        val_pred_aitb, _, _ = train_loop.predict_on_validation_set()
        test_pred_aitb, _, _ = train_loop.predict_on_test_set()
        train_loss_aitb = train_loop.evaluate_loss_on_train_set()
        val_loss_aitb = train_loop.evaluate_loss_on_validation_set()
        test_loss_aitb = train_loop.evaluate_loss_on_test_set()

        model_pt.train()
        for epoch in range(num_epochs):
            for i, (input_data, target) in enumerate(train_dataloader):
                predicted = model_pt(input_data)
                loss = criterion_pt(predicted, target)
                loss = loss / grad_accumulation
                loss.backward()

                if (i+1) % grad_accumulation == 0:
                    optimizer_pt.step()
                    optimizer_pt.zero_grad()

        train_pred_pt, train_loss_pt = [], []
        model_pt.eval()
        with torch.no_grad():
            for input_data, target in train_dataloader:
                predicted = model_pt(input_data)
                loss_batch = criterion_pt(predicted, target).item()
                train_pred_pt += predicted.argmax(dim=1, keepdim=False).tolist()
                train_loss_pt.append(loss_batch)
            train_loss_pt = np.mean(train_loss_pt)

        val_pred_pt, val_loss_pt = [], []
        model_pt.eval()
        with torch.no_grad():
            for input_data, target in val_dataloader:
                predicted = model_pt(input_data)
                loss_batch = criterion_pt(predicted, target).item()
                val_pred_pt += predicted.argmax(dim=1, keepdim=False).tolist()
                val_loss_pt.append(loss_batch)
            val_loss_pt = np.mean(val_loss_pt)

        test_pred_pt, test_loss_pt = [], []
        model_pt.eval()
        with torch.no_grad():
            for input_data, target in test_dataloader:
                predicted = model_pt(input_data)
                loss_batch = criterion_pt(predicted, target).item()
                test_pred_pt += predicted.argmax(dim=1, keepdim=False).tolist()
                test_loss_pt.append(loss_batch)
            test_loss_pt = np.mean(test_loss_pt)

        self.assertEqual(train_pred_aitb.tolist(), train_pred_pt)
        self.assertEqual(val_pred_aitb.tolist(), val_pred_pt)
        self.assertEqual(test_pred_aitb.tolist(), test_pred_pt)

        self.assertEqual(train_loss_aitb, train_loss_pt)
        self.assertEqual(val_loss_aitb, val_loss_pt)
        self.assertEqual(test_loss_aitb, test_loss_pt)

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
