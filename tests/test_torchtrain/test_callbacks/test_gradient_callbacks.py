import unittest

from tests.utils import *

from AIToolbox.torchtrain.callbacks.gradient_callbacks import GradNormClipCallback, GradientStatsPrintCallback
from AIToolbox.torchtrain.train_loop import TrainLoop


def build_train_loop(model):
    dummy_optimizer = DummyOptimizer()
    dummy_train_loader = list(range(4))
    dummy_val_loader = list(range(3))
    dummy_test_loader = list(range(2))
    return TrainLoop(model, dummy_train_loader, dummy_val_loader, dummy_test_loader, dummy_optimizer, None)


class TestGradNormClipCallback(unittest.TestCase):
    def test_on_train_loop_registration(self):
        callback = GradNormClipCallback(0.1)
        model = NetUnifiedBatchFeed()
        train_loop = build_train_loop(model)
        train_loop.callbacks_handler.register_callbacks([callback])
        self.assertTrue(train_loop.grad_cb_used)


class TestGradientStatsPrintCallback(unittest.TestCase):
    def test_on_train_loop_registration(self):
        callback = GradientStatsPrintCallback(lambda m: [m.conv1, m.conv2, m.fc1, m.fc2])
        model = NetUnifiedBatchFeed()
        train_loop = build_train_loop(model)
        train_loop.callbacks_handler.register_callbacks([callback])
        self.assertFalse(train_loop.grad_cb_used)

        callback = GradientStatsPrintCallback(lambda m: [m.conv1, m.conv2, m.fc1, m.fc2], on_every_grad_update=True)
        model = NetUnifiedBatchFeed()
        train_loop = build_train_loop(model)
        train_loop.callbacks_handler.register_callbacks([callback])
        self.assertTrue(train_loop.grad_cb_used)
