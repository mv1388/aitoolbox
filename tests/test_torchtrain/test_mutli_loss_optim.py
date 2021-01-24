import unittest
import torch
from tests.utils import DummyOptimizer, DummyBatch as DummyLoss

from aitoolbox.torchtrain.multi_loss_optim import MultiLoss, MultiOptimizer


class TestMultiLoss(unittest.TestCase):
    def test_init(self):
        loss_1, loss_2, loss_3, _ = self.build_loss()
        multi_loss = MultiLoss({'loss1': loss_1, 'loss2': loss_2, 'loss3': loss_3},
                               {'loss1': 0, 'loss2': 2, 'loss3': 1})

        self.assertEqual(multi_loss.optimizer_loss_map, {0: 'loss1', 2: 'loss2', 1: 'loss3'})

        multi_loss_2 = MultiLoss({'loss1': loss_1, 'loss2': loss_2, 'loss3': loss_3})
        self.assertEqual(multi_loss_2.optimizer_loss_map, {0: 'loss1', 1: 'loss2', 2: 'loss3'})

    def test_backward(self):
        loss_1, loss_2, loss_3, multi_loss = self.build_loss()
        grad_scaler = torch.cuda.amp.GradScaler(enabled=False)

        for i in range(1, 100):
            for opti_idx in range(3):
                multi_loss.backward(opti_idx, i, grad_scaler)
            self.assertEqual(loss_1.back_ctr, i)
            self.assertEqual(loss_2.back_ctr, i)
            self.assertEqual(loss_3.back_ctr, i)

        self.assertEqual(loss_1.retain_graph_ctr, 0)
        self.assertEqual(loss_2.retain_graph_ctr, 0)
        self.assertEqual(loss_3.retain_graph_ctr, 0)

    def test_backward_backward_remaining(self):
        loss_1, loss_2, loss_3, multi_loss = self.build_loss()
        multi_loss.retain_graph_until_last = True
        grad_scaler = torch.cuda.amp.GradScaler(enabled=False)

        for opti_idx in range(3):
            multi_loss.backward(opti_idx, 0, grad_scaler)
            self.assertEqual(multi_loss.loss_backward_remaining, 2-opti_idx)

        self.assertEqual(loss_1.retain_graph_ctr, 1)
        self.assertEqual(loss_2.retain_graph_ctr, 1)
        self.assertEqual(loss_3.retain_graph_ctr, 0)

    def test_item(self):
        loss_1, loss_2, loss_3, multi_loss = self.build_loss()

        for i in range(1, 100):
            items = multi_loss.item()
            self.assertEqual(items, {'loss1': i, 'loss2': i, 'loss3': i})

            self.assertEqual(loss_1.item_ctr, i)
            self.assertEqual(loss_2.item_ctr, i)
            self.assertEqual(loss_3.item_ctr, i)

    def test_object_div(self):
        loss_1, loss_2, loss_3, multi_loss = self.build_loss()
        loss_1.value = 10.
        loss_2.value = 20.
        loss_3.value = 30.

        multi_loss = multi_loss / 10.

        self.assertEqual({k: v.value for k, v in multi_loss.loss_dict.items()},
                         {'loss1': 1.0, 'loss2': 2.0, 'loss3': 3.0})

    @staticmethod
    def build_loss():
        loss_1 = DummyLossItemChg()
        loss_2 = DummyLossItemChg()
        loss_3 = DummyLossItemChg()
        multi_loss = MultiLoss({'loss1': loss_1, 'loss2': loss_2, 'loss3': loss_3}, retain_graph_until_last=False)
        return loss_1, loss_2, loss_3, multi_loss


class TestMultiOptimizer(unittest.TestCase):
    def test_zero_grad(self):
        opti_1, opti_2, opti_3, multi_opti = self.build_optimizers()

        for i in range(1, 100):
            for j in range(3):
                multi_opti.zero_grad(j, i)
            self.assertEqual(opti_1.zero_grad_ctr, i)
            self.assertEqual(opti_2.zero_grad_ctr, i)
            self.assertEqual(opti_3.zero_grad_ctr, i)

    def test_step(self):
        opti_1, opti_2, opti_3, multi_opti = self.build_optimizers()
        grad_scaler = torch.cuda.amp.GradScaler(enabled=False)

        for i in range(1, 100):
            for j in range(3):
                multi_opti.step(j, i, grad_scaler)
            self.assertEqual(opti_1.step_ctr, i)
            self.assertEqual(opti_2.step_ctr, i)
            self.assertEqual(opti_3.step_ctr, i)

    def test_state_dict(self):
        _, _, _, multi_opti = self.build_optimizers()
        grad_scaler = torch.cuda.amp.GradScaler(enabled=False)

        for i in range(1, 100):
            for j in range(3):
                multi_opti.zero_grad(j, i)
                multi_opti.step(j, i, grad_scaler)

        self.assertEqual(multi_opti.state_dict(),
                         [{'zero_grad_ctr': 99, 'step_ctr': 99}, {'zero_grad_ctr': 99, 'step_ctr': 99},
                          {'zero_grad_ctr': 99, 'step_ctr': 99}])

    @staticmethod
    def build_optimizers():
        opti_1 = DummyOptimizerStateD()
        opti_2 = DummyOptimizerStateD()
        opti_3 = DummyOptimizerStateD()
        multi_opti = MultiOptimizer([opti_1, opti_2, opti_3])
        return opti_1, opti_2, opti_3, multi_opti


class DummyOptimizerStateD(DummyOptimizer):
    def __init__(self):
        super().__init__()

    def state_dict(self):
        return {'zero_grad_ctr': self.zero_grad_ctr, 'step_ctr': self.step_ctr}


class DummyLossItemChg(DummyLoss):
    def __init__(self, value=0):
        super().__init__()
        self.value = value
        self.retain_graph_ctr = 0

    def backward(self, retain_graph=False):
        self.back_ctr += 1
        if retain_graph:
            self.retain_graph_ctr += 1

    def item(self):
        self.item_ctr += 1
        return self.item_ctr

    def __truediv__(self, other):
        self.value = self.value / other
        return self
