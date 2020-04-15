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


class FFNet(TTModel):
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

        predicted = self(input_data)

        return predicted.cpu(), target, {'example_feat_sum': input_data.sum(dim=1).tolist()}


class TestEnd2EndTrainLoop(unittest.TestCase):
    def test_e2e_ff_net_train_loop(self):
        self.set_seeds()
        batch_size = 10

        train_dataset = TensorDataset(torch.randn(100, 50), torch.randint(low=0, high=10, size=(100,)))
        val_dataset = TensorDataset(torch.randn(30, 50), torch.randint(low=0, high=10, size=(30,)))
        test_dataset = TensorDataset(torch.randn(30, 50), torch.randint(low=0, high=10, size=(30,)))

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

        model = FFNet()
        optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
        criterion = nn.NLLLoss()

        train_loop = TrainLoop(
            model,
            train_dataloader, val_dataloader, test_dataloader,
            optimizer, criterion
        )

        train_loop.fit(num_epochs=5)

        print(train_loop.train_history.train_history)

        self.assertEqual(
            train_loop.train_history.train_history,
            {'loss': [2.224587655067444, 2.1440203189849854, 2.0584306001663206, 1.962017869949341, 1.8507084131240845],
             'accumulated_loss': [2.3059947967529295, 2.1976317405700683, 2.114974856376648, 2.0259472250938417, 1.9252637863159179],
             'val_loss': [2.330514828364054, 2.345397472381592, 2.363233725229899, 2.3853348096211753, 2.4111196994781494],
             'train_end_test_loss': [2.31626296043396]}
        )
        self.assertEqual(train_loop.epoch, 4)

    # def test_e2e_ff_net_train_loop_loss(self):
    #     self.set_seeds()
    #     batch_size = 10
    #
    #     train_dataset = TensorDataset(torch.randn(100, 50), torch.randint(low=0, high=10, size=(100,)))
    #     val_dataset = TensorDataset(torch.randn(30, 50), torch.randint(low=0, high=10, size=(30,)))
    #     test_dataset = TensorDataset(torch.randn(30, 50), torch.randint(low=0, high=10, size=(30,)))
    #
    #     train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    #     val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    #     test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    #
    #     model = FFNet()
    #     optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    #     criterion = nn.NLLLoss()
    #
    #     train_loop = TrainLoop(
    #         model,
    #         train_dataloader, val_dataloader, test_dataloader,
    #         optimizer, criterion
    #     )
    #     train_loop.fit(num_epochs=5)
    #
    #     self.assertEqual(train_loop.evaluate_loss_on_train_set(), 1.8507084131240845)
    #     self.assertEqual(train_loop.evaluate_loss_on_validation_set(), 2.4111196994781494)
    #     self.assertEqual(train_loop.evaluate_loss_on_test_set(), 2.31626296043396)

    def test_e2e_ff_net_train_loop_predictions(self):
        self.set_seeds()
        batch_size = 10

        train_dataset = TensorDataset(torch.randn(100, 50), torch.randint(low=0, high=10, size=(100,)))
        val_dataset = TensorDataset(torch.randn(30, 50), torch.randint(low=0, high=10, size=(30,)))
        test_dataset = TensorDataset(torch.randn(30, 50), torch.randint(low=0, high=10, size=(30,)))

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

        model = FFNet()
        optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
        criterion = nn.NLLLoss()

        train_loop = TrainLoop(
            model,
            train_dataloader, val_dataloader, test_dataloader,
            optimizer, criterion
        )
        train_loop.fit(num_epochs=5)

        train_pred, train_target, train_meta = train_loop.predict_on_train_set()
        self.assertEqual(
            train_pred.argmax(dim=1).tolist(),
            [0, 4, 8, 0, 1, 1, 6, 1, 6, 1, 8, 3, 0, 0, 8, 8, 1, 8, 8, 1, 8, 0, 8, 0, 0, 1, 3, 4, 8, 8, 0, 8, 8, 1, 8, 8,
             5, 5, 1, 8, 8, 8, 8, 8, 9, 0, 8, 8, 5, 8, 8, 1, 1, 8, 5, 1, 8, 8, 5, 5, 1, 8, 8, 8, 8, 0, 0, 1, 1, 0, 8, 8,
             3, 0, 5, 0, 8, 9, 1, 8, 8, 8, 5, 8, 5, 8, 0, 1, 8, 5, 8, 6, 5, 1, 8, 1, 0, 8, 1, 8]
        )
        self.assertEqual(train_target.tolist(), train_dataset.tensors[1].tolist())
        self.assertEqual(train_dataset.tensors[0].sum(dim=1).tolist(), train_meta['example_feat_sum'])

        val_pred, val_target, val_meta = train_loop.predict_on_validation_set()
        self.assertEqual(
            val_pred.argmax(dim=1).tolist(),
            [1, 1, 1, 1, 5, 8, 0, 8, 1, 1, 5, 8, 8, 1, 8, 8, 1, 8, 8, 8, 1, 0, 0, 8, 0, 1, 1, 0, 1, 8]
        )
        self.assertEqual(val_target.tolist(), val_dataset.tensors[1].tolist())
        self.assertEqual(val_dataset.tensors[0].sum(dim=1).tolist(), val_meta['example_feat_sum'])

        test_pred, test_target, test_meta = train_loop.predict_on_test_set()
        self.assertEqual(
            test_pred.argmax(dim=1).tolist(),
            [4, 8, 0, 8, 1, 8, 1, 1, 8, 8, 8, 8, 8, 0, 8, 8, 5, 8, 8, 5, 8, 1, 0, 5, 1, 8, 8, 8, 8, 1]
        )
        self.assertEqual(test_target.tolist(), test_dataset.tensors[1].tolist())
        self.assertEqual(test_dataset.tensors[0].sum(dim=1).tolist(), test_meta['example_feat_sum'])

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
