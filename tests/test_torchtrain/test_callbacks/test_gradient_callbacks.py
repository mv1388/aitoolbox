import unittest
import numpy as np
import io
import sys
import torch
from torch.utils.data.dataset import TensorDataset

from tests.utils import *
from torch.utils.data import DataLoader
import torch.optim as optim

from aitoolbox.torchtrain.callbacks.gradient import GradNormClip, GradientStatsPrint
from aitoolbox.torchtrain.train_loop import TrainLoop


def build_train_loop(model):
    dummy_optimizer = DummyOptimizer()
    dummy_train_loader = list(range(4))
    dummy_val_loader = list(range(3))
    dummy_test_loader = list(range(2))
    return TrainLoop(model, dummy_train_loader, dummy_val_loader, dummy_test_loader, dummy_optimizer, None)


class TestGradNormClipCallback(unittest.TestCase):
    def test_on_train_loop_registration(self):
        callback = GradNormClip(0.1)
        model = NetUnifiedBatchFeed()
        train_loop = build_train_loop(model)
        train_loop.callbacks_handler.register_callbacks([callback])
        self.assertTrue(train_loop.grad_cb_used)


class TestGradientStatsPrintCallback(unittest.TestCase):
    def test_on_train_loop_registration(self):
        callback = GradientStatsPrint(lambda m: [m.conv1, m.conv2, m.fc1, m.fc2])
        model = NetUnifiedBatchFeed()
        train_loop = build_train_loop(model)
        train_loop.callbacks_handler.register_callbacks([callback])
        self.assertTrue(train_loop.grad_cb_used)

        callback = GradientStatsPrint(lambda m: [m.conv1, m.conv2, m.fc1, m.fc2], stats_eval_freq=10)
        model = NetUnifiedBatchFeed()
        train_loop = build_train_loop(model)
        train_loop.callbacks_handler.register_callbacks([callback])
        self.assertTrue(train_loop.grad_cb_used)

    def test_gradients_report(self):
        callback = GradientStatsPrint(lambda m: [m.l1, m.l2])
        model = SmallFFNet()

        x = torch.Tensor(np.random.rand(10, 10))
        y = torch.Tensor(np.random.rand(10))

        train_loader = DataLoader(TensorDataset(x, y), batch_size=10)
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.BCELoss()

        for data_batch in train_loader:
            batch_loss = model.get_loss(data_batch, criterion, torch.device('cpu'))
            batch_loss.backward()

        gradients_l1 = model.l1.weight.grad.cpu().numpy()
        gradients_l2 = model.l2.weight.grad.cpu().numpy()
        model.zero_grad()

        captured_output = io.StringIO()  # Create StringIO object
        sys.stdout = captured_output

        TrainLoop(model, train_loader, None, None, optimizer, criterion).fit(num_epochs=1, callbacks=[callback])
        sys.stdout = sys.__stdout__
        cb_output = '\n'.join(captured_output.getvalue().split('\n')[5:-4]).strip()

        expected_print = \
            f"""
---> Model layers gradients stats
\tLayer 0 grads: Mean: {np.mean(gradients_l1)}; Std {np.std(gradients_l1)}
\t\tRatio of zero gradients: {float(np.count_nonzero(gradients_l1 == 0)) / gradients_l1.size}
\tLayer 1 grads: Mean: {np.mean(gradients_l2)}; Std {np.std(gradients_l2)}
\t\tRatio of zero gradients: {float(np.count_nonzero(gradients_l2 == 0)) / gradients_l2.size}
        """.strip()

        self.assertEqual(expected_print, cb_output)
