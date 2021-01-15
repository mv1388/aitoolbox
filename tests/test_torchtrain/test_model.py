import unittest

from tests.utils import *

from aitoolbox.torchtrain.model import TTModel, TTBasicModel
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


class MyBasicModel(TTBasicModel):
    def __init__(self):
        super().__init__()

    def forward(self, *input_data):
        pass

    def __call__(self, a, b):
        return DummyData([a.value + 10, b.value + 20], device='gpu')


class MyComplexModel(TTBasicModel):
    def __init__(self):
        super().__init__()

    def forward(self, *input_data):
        pass

    def __call__(self, seq_batch, seq_len_batch, example_weights_batch):
        re_weighted_weights = [w + 10 if len(x) == l else w
                               for x, w, l in zip(seq_batch.value, example_weights_batch.value, seq_len_batch.value)]

        pred = [sum(x) * w for x, w in zip(seq_batch.value, re_weighted_weights)]
        pred = DummyData(pred, device=seq_batch.device)

        return pred


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
        model = TTBasicModel()
        self.assertTrue(isinstance(model, nn.Module))
        self.assertTrue(isinstance(model, TTModel))

    def test_get_loss(self):
        d1 = DummyData(1)
        d2 = DummyData(2)
        d3 = DummyData(300)

        model = MyBasicModel()
        loss = model.get_loss([d1, d2, d3], criterion=lambda y_pred, y: sum(y_pred.value + [y.value]), device='gpu')
        self.assertEqual(loss, 333)
        self.assertEqual(d1.device, 'gpu')
        self.assertEqual(d2.device, 'gpu')
        self.assertEqual(d3.device, 'gpu')

    def test_get_predictions(self):
        d1 = DummyData(1)
        d2 = DummyData(2)
        d3 = DummyData(300)

        model = MyBasicModel()
        predictions, targets, metadata = model.get_predictions([d1, d2, d3], 'gpu')
        self.assertEqual(predictions.value, [11, 22])
        self.assertEqual(predictions.device, 'cpu')
        self.assertEqual(targets.value, 300)
        self.assertEqual(targets.device, 'cpu')
        self.assertEqual(metadata, {})
        self.assertEqual(d1.device, 'gpu')
        self.assertEqual(d2.device, 'gpu')
        self.assertEqual(d3.device, 'cpu')

    def test_get_loss_complex(self):
        d_seq = DummyData([[1, 2, 3, 4],
                           [3, 3],
                           [5, 6, 7, 8, 9]], 'cpu')
        d_seq_lens = DummyData([4, 2, 5], 'cpu')
        d_example_weights = DummyData([100, 10, 1000], 'cpu')
        d_target = DummyData([1., 0.5, 1.5], 'cpu')

        model = MyComplexModel()
        loss = model.get_loss([d_seq, d_seq_lens, d_example_weights, d_target],
                              criterion=lambda y_pred, y: sum([el_pred * el_true for el_pred, el_true in zip(y_pred.value, y.value)]),
                              device='gpu')
        self.assertEqual(loss, 54185.0)

        re_weighted_weights = [w + 10 if len(x) == l else w
                               for x, w, l in zip(d_seq.value, d_example_weights.value, d_seq_lens.value)]
        pred = [sum(x) * w for x, w in zip(d_seq.value, re_weighted_weights)]
        correct_loss = sum([el_pred * el_true for el_pred, el_true in zip(pred, d_target.value)])
        self.assertEqual(loss, correct_loss)

        self.assertEqual(d_seq.device, 'gpu')
        self.assertEqual(d_seq_lens.device, 'gpu')
        self.assertEqual(d_example_weights.device, 'gpu')
        self.assertEqual(d_target.device, 'gpu')

    def test_get_predictions_complex(self):
        d_seq = DummyData([[1, 2, 3, 4],
                           [3, 3],
                           [5, 6, 7, 8, 9]], 'cpu')
        d_seq_lens = DummyData([4, 2, 5], 'cpu')
        d_example_weights = DummyData([100, 10, 1000], 'cpu')
        d_target = DummyData([1., 0.5, 1.5], 'cpu')

        model = MyComplexModel()
        predictions, targets, metadata = model.get_predictions([d_seq, d_seq_lens, d_example_weights, d_target], 'gpu')
        self.assertEqual(predictions.value, [1100, 120, 35350])
        self.assertEqual(predictions.device, 'cpu')
        self.assertEqual(targets.value, [1., 0.5, 1.5])
        self.assertEqual(targets.device, 'cpu')
        self.assertEqual(metadata, {})

        self.assertEqual(d_seq.device, 'gpu')
        self.assertEqual(d_seq_lens.device, 'gpu')
        self.assertEqual(d_example_weights.device, 'gpu')
        self.assertEqual(d_target.device, 'cpu')

        re_weighted_weights = [w + 10 if len(x) == l else w
                               for x, w, l in zip(d_seq.value, d_example_weights.value, d_seq_lens.value)]
        pred = [sum(x) * w for x, w in zip(d_seq.value, re_weighted_weights)]
        self.assertEqual(predictions.value, pred)

    def test_get_loss_keep_cpu(self):
        d_seq = DummyData([[1, 2, 3, 4],
                           [3, 3],
                           [5, 6, 7, 8, 9]], 'cpu')
        d_seq_lens = DummyData([4, 2, 5], 'cpu')
        d_example_weights = DummyData([100, 10, 1000], 'cpu')
        d_target = DummyData([1., 0.5, 1.5], 'cpu')

        model = MyComplexModel()
        loss = model.get_loss([d_seq, d_seq_lens, d_example_weights, d_target],
                              criterion=lambda y_pred, y: sum([el_pred * el_true for el_pred, el_true in zip(y_pred.value, y.value)]),
                              device='cpu')
        self.assertEqual(loss, 54185.0)
        self.assertEqual(d_seq.device, 'cpu')
        self.assertEqual(d_seq_lens.device, 'cpu')
        self.assertEqual(d_example_weights.device, 'cpu')
        self.assertEqual(d_target.device, 'cpu')

    def test_get_predictions_keep_cpu(self):
        d_seq = DummyData([[1, 2, 3, 4],
                           [3, 3],
                           [5, 6, 7, 8, 9]], 'cpu')
        d_seq_lens = DummyData([4, 2, 5], 'cpu')
        d_example_weights = DummyData([100, 10, 1000], 'cpu')
        d_target = DummyData([1., 0.5, 1.5], 'cpu')

        model = MyComplexModel()
        predictions, targets, metadata = model.get_predictions([d_seq, d_seq_lens, d_example_weights, d_target], 'cpu')
        self.assertEqual(predictions.value, [1100, 120, 35350])
        self.assertEqual(predictions.device, 'cpu')
        self.assertEqual(targets.value, [1., 0.5, 1.5])
        self.assertEqual(targets.device, 'cpu')
        self.assertEqual(metadata, {})

        self.assertEqual(d_seq.device, 'cpu')
        self.assertEqual(d_seq_lens.device, 'cpu')
        self.assertEqual(d_example_weights.device, 'cpu')
        self.assertEqual(d_target.device, 'cpu')


class MyModel(NetUnifiedBatchFeed):
    def __init__(self):
        super().__init__()
        self.model_level_str = 'test string'

        self.transfer_model_attributes = ['model_level_str']

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
