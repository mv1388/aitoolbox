import unittest
from torch import nn

from aitoolbox.torchtrain.model import TTModel
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
