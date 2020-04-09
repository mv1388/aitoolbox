import unittest
import torch.nn as nn

from tests.test_torchtrain.test_model import MyModel

from aitoolbox import TTDataParallel
from aitoolbox.torchtrain.parallel import TTParallelBase


class TestTTParallelBase(unittest.TestCase):
    def test_init_default(self):
        model = MyModel()
        model_parallel = TTParallelBase(model)

        self.assertTrue(hasattr(model_parallel, 'get_loss'))
        self.assertTrue(hasattr(model_parallel, 'get_loss_eval'))
        self.assertTrue(hasattr(model_parallel, 'get_predictions'))

        self.assertTrue(hasattr(model_parallel, 'my_new_fn'))
        self.assertTrue(hasattr(model_parallel, 'get_model_level_str'))

        self.assertFalse(hasattr(model_parallel, 'model_level_str'))

    def test_init_attr_transfer(self):
        model = MyModel()
        model_parallel = TTParallelBase(model, add_model_attributes=['model_level_str'])

        self.assertTrue(hasattr(model_parallel, 'model_level_str'))

        self.assertEqual(model_parallel.get_model_level_str(), 'test string')
        self.assertEqual(model_parallel.get_model_level_str(), model.model_level_str)

    def test_core_model_transferred_fns(self):
        model = MyModel()
        model_parallel = TTParallelBase(model)
        self.assertEqual(model_parallel.get_loss(None, None, None), 'loss_return')
        self.assertEqual(model_parallel.get_loss_eval(None, None, None), 'loss_eval_return')
        self.assertEqual(model_parallel.get_predictions(None, None), 'predictions_return')
        self.assertEqual(model_parallel.my_new_fn(), 'my_new_fn return value')


class TestTTDataParallel(unittest.TestCase):
    def test_init_default(self):
        model = MyModel()
        model_parallel = TTDataParallel(model)

        self.assertTrue(hasattr(model_parallel, 'get_loss'))
        self.assertTrue(hasattr(model_parallel, 'get_loss_eval'))
        self.assertTrue(hasattr(model_parallel, 'get_predictions'))

        self.assertTrue(hasattr(model_parallel, 'my_new_fn'))
        self.assertTrue(hasattr(model_parallel, 'get_model_level_str'))

        self.assertFalse(hasattr(model_parallel, 'model_level_str'))

    def test_inheritance(self):
        model = MyModel()
        model_parallel = TTDataParallel(model)

        self.assertTrue(isinstance(model_parallel, nn.DataParallel))
        self.assertTrue(isinstance(model_parallel, TTParallelBase))

    def test_init_attr_transfer(self):
        model = MyModel()
        model_parallel = TTDataParallel(model, add_model_attributes=['model_level_str'])

        self.assertTrue(hasattr(model_parallel, 'model_level_str'))

        self.assertEqual(model_parallel.get_model_level_str(), 'test string')
        self.assertEqual(model_parallel.get_model_level_str(), model.model_level_str)

    def test_core_model_transferred_fns(self):
        model = MyModel()
        model_parallel = TTDataParallel(model)
        self.assertEqual(model_parallel.get_loss(None, None, None), 'loss_return')
        self.assertEqual(model_parallel.get_loss_eval(None, None, None), 'loss_eval_return')
        self.assertEqual(model_parallel.get_predictions(None, None), 'predictions_return')
        self.assertEqual(model_parallel.my_new_fn(), 'my_new_fn return value')
