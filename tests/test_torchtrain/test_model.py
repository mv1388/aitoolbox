import unittest
from torch import nn

from tests.utils import *

from aitoolbox.torchtrain.model import TTModel, TTModelBasic, TTDataParallel
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


class MyBasicModel(TTModelBasic):
    def __init__(self):
        super().__init__()

    def forward(self, *input):
        pass

    def __call__(self, *args):
        return DummyData([el.value + 10 for el in args], device='gpu')


class DummyData:
    def __init__(self, value, device='cpu'):
        self.value = value
        self.device = device

    def to(self, device):
        self.device = device
        return self

    def cpu(self):
        self.device = 'cpu'
        return self


class TestTTModelBasic(unittest.TestCase):
    def test_inheritance(self):
        model = TTModelBasic()
        self.assertTrue(isinstance(model, nn.Module))
        self.assertTrue(isinstance(model, TTModel))

    def test_get_loss(self):
        d1 = DummyData(1)
        d2 = DummyData(2)
        d3 = DummyData(300)

        model = MyBasicModel()
        loss = model.get_loss([d1, d2, d3], lambda y_pred, y: sum(y_pred.value + [y.value]), 'gpu')
        self.assertEqual(loss, 323)
        self.assertEqual(d1.device, 'gpu')
        self.assertEqual(d2.device, 'gpu')
        self.assertEqual(d3.device, 'gpu')

    def test_get_predictions(self):
        d1 = DummyData(1)
        d2 = DummyData(2)
        d3 = DummyData(300)

        model = MyBasicModel()
        predictions, targets, metadata = model.get_predictions([d1, d2, d3], 'gpu')
        self.assertEqual(predictions.value, [11, 12])
        self.assertEqual(predictions.device, 'cpu')
        self.assertEqual(targets.value, 300)
        self.assertEqual(targets.device, 'cpu')
        self.assertEqual(d1.device, 'gpu')
        self.assertEqual(d2.device, 'gpu')
        self.assertEqual(d3.device, 'cpu')


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
