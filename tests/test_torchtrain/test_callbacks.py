import unittest

from tests.test_torchtrain.utils import *

from AIToolbox.torchtrain.callbacks.callbacks import AbstractCallback
from AIToolbox.torchtrain.train_loop import TrainLoop


class TestAbstractCallback(unittest.TestCase):
    def test_abstract_callback_has_hook_methods(self):
        callback = AbstractCallback('test_callback')

        self.assertTrue(function_exists(callback, 'on_train_loop_registration'))
        self.assertTrue(function_exists(callback, 'on_epoch_begin'))
        self.assertTrue(function_exists(callback, 'on_epoch_end'))
        self.assertTrue(function_exists(callback, 'on_train_begin'))
        self.assertTrue(function_exists(callback, 'on_train_end'))
        self.assertTrue(function_exists(callback, 'on_batch_begin'))
        self.assertTrue(function_exists(callback, 'on_batch_end'))

    def test_on_train_loop_registration_hook(self):
        train_loop = TrainLoop(Net(), None, 100, DeactivateModelFeedDefinition(), None, None)
        callback = CallbackTracker()
        callback.register_train_loop_object(train_loop)

        self.assertIsInstance(callback, AbstractCallback)
        self.assertEqual(callback.callback_calls, ['on_train_loop_registration'])


