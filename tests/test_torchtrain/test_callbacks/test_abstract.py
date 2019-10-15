import unittest

from aitoolbox.torchtrain.callbacks.callbacks import AbstractCallback
from aitoolbox.torchtrain.train_loop import TrainLoop
from tests.utils import function_exists, NetUnifiedBatchFeed, CallbackTracker


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
        self.assertTrue(function_exists(callback, 'on_after_gradient_update'))
        self.assertTrue(function_exists(callback, 'on_after_optimizer_step'))

    def test_on_train_loop_registration_hook(self):
        train_loop = TrainLoop(NetUnifiedBatchFeed(), None, None, None, None, None)
        callback = CallbackTracker()
        callback.register_train_loop_object(train_loop)

        self.assertIsInstance(callback, AbstractCallback)
        self.assertEqual(callback.callback_calls, ['on_train_loop_registration'])


class TestAbstractExperimentCallback(unittest.TestCase):
    def test_init(self):
        pass

    def test_try_infer_experiment_details(self):
        pass
