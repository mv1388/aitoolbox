import unittest
from torch import nn

from tests.utils import *

from aitoolbox.torchtrain.model import TTModel, TTDataParallel
from aitoolbox.utils.util import function_exists


class TestTTModel(unittest.TestCase):
    def test_methods_found(self):
        self.assertTrue(function_exists(TTModel, 'get_loss'))
        self.assertTrue(function_exists(TTModel, 'get_loss_eval'))
        self.assertTrue(function_exists(TTModel, 'get_predictions'))
        self.assertTrue(function_exists(TTModel, 'forward'))

        class FailingModel(TTModel):
            def __init__(self):
                super().__init__()

        with self.assertRaises(TypeError):
            FailingModel()

    def test_inherited_nn_module(self):
        class MyModel(TTModel):
            def __init__(self):
                super().__init__()

            def get_loss(self, batch_data, criterion, device):
                pass

            def get_predictions(self, batch_data, device):
                pass

        self.assertTrue(isinstance(MyModel(), nn.Module))


class MyModel(NetUnifiedBatchFeed):
    def __init__(self):
        super().__init__()
        self.model_level_str = 'test string'

    def get_loss(self, batch_data, criterion, device):
        return 'loss_return'

    def get_loss_eval(self, batch_data, criterion, device):
        return 'loss_eval_return'

    def get_predictions(self, batch_data, device):
        return 'predictions_return'

    def my_new_fn(self):
        return 'my_new_fn return value'

    def get_model_level_str(self):
        return self.model_level_str


class TestTTDataParallel(unittest.TestCase):
    def test_init_default(self):
        model = MyModel()
        model_parallel = TTDataParallel(model)

        self.assertTrue(hasattr(model_parallel, 'get_loss'))
        self.assertTrue(hasattr(model_parallel, 'get_loss_eval'))
        self.assertTrue(hasattr(model_parallel, 'get_predictions'))

        self.assertTrue(hasattr(model_parallel, 'my_new_fn'))
        self.assertTrue(hasattr(model_parallel, 'get_model_level_str'))

        self.assertEqual(model_parallel.my_new_fn(), 'my_new_fn return value')

    def test_init_attr_transfer(self):
        model = MyModel()
        model_parallel = TTDataParallel(model, add_model_attributes=['model_level_str'])

        self.assertEqual(model_parallel.get_model_level_str(), 'test string')
        self.assertEqual(model_parallel.get_model_level_str(), model.model_level_str)

    def test_core_model_transferred_fns(self):
        model = MyModel()
        model_parallel = TTDataParallel(model)
        self.assertEqual(model_parallel.get_loss(None, None, None), 'loss_return')
        self.assertEqual(model_parallel.get_loss_eval(None, None, None), 'loss_eval_return')
        self.assertEqual(model_parallel.get_predictions(None, None), 'predictions_return')
