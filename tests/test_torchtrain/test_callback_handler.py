import unittest

from tests.utils import *
from AIToolbox.torchtrain.callbacks.callback_handler import CallbacksHandler
from AIToolbox.torchtrain.train_loop import TrainLoop


class TestCallbacksHandler(unittest.TestCase):
    def test_callback_handler_has_hook_methods(self):
        callback_handler_1 = CallbacksHandler(None)
        self.check_callback_handler_for_hooks(callback_handler_1)

        train_loop = TrainLoop(Net(), None, 100, DeactivateModelFeedDefinition(), None, None)
        callback_handler_2 = CallbacksHandler(train_loop)
        self.check_callback_handler_for_hooks(callback_handler_2)

    def check_callback_handler_for_hooks(self, callback_handler):
        self.assertTrue(function_exists(callback_handler, 'register_callbacks'))
        self.assertTrue(function_exists(callback_handler, 'execute_epoch_begin'))
        self.assertTrue(function_exists(callback_handler, 'execute_epoch_end'))
        self.assertTrue(function_exists(callback_handler, 'execute_train_begin'))
        self.assertTrue(function_exists(callback_handler, 'execute_train_end'))
        self.assertTrue(function_exists(callback_handler, 'execute_batch_begin'))
        self.assertTrue(function_exists(callback_handler, 'execute_batch_end'))
