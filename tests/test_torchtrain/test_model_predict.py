import unittest
import torch.nn as nn

from tests.utils import *

from aitoolbox.torchtrain.model import ModelWrap
from aitoolbox.torchtrain.model_predict import PyTorchModelPredictor


class TestAbstractModelPredictor(unittest.TestCase):
    def test_if_has_abstractmethod(self):
        model = NetUnifiedBatchFeed()
        dummy_val_loader = list(range(2))
        re_runner = PyTorchModelPredictor(model, dummy_val_loader)

        self.assertTrue(function_exists(re_runner, 'model_predict'))
        self.assertTrue(function_exists(re_runner, 'model_get_loss'))
        self.assertTrue(function_exists(re_runner, 'evaluate_result_package'))

    def test_predict(self):
        model = NetUnifiedBatchFeed()
        dummy_val_loader = list(range(2))
        re_runner = PyTorchModelPredictor(model, dummy_val_loader)

        y_pred, y_test, metadata = re_runner.model_predict()

        r = []
        for i in range(1, len(dummy_val_loader) + 1):
            r += [i] * 64
        r2 = []
        for i in range(1, len(dummy_val_loader) + 1):
            r2 += [i + 100] * 64
        self.assertEqual(y_test.tolist(), r)
        self.assertEqual(y_pred.tolist(), r2)

        d = {'bla': []}
        for i in range(1, len(dummy_val_loader) + 1):
            d['bla'] += [i + 200] * 64
        self.assertEqual(metadata, d)

    def test_predict_separate_batch_feed(self):
        model = Net()
        batch_loader = DeactivateModelFeedDefinition()
        model_wrap = ModelWrap(model=model, batch_model_feed_def=batch_loader)

        dummy_val_loader = list(range(2))
        re_runner = PyTorchModelPredictor(model_wrap, dummy_val_loader)

        y_pred, y_test, metadata = re_runner.model_predict()

        r = []
        for i in range(1, len(dummy_val_loader) + 1):
            r += [i] * 64
        r2 = []
        for i in range(1, len(dummy_val_loader) + 1):
            r2 += [i + 100] * 64
        self.assertEqual(y_test.tolist(), r)
        self.assertEqual(y_pred.tolist(), r2)

        d = {'bla': []}
        for i in range(1, len(dummy_val_loader) + 1):
            d['bla'] += [i + 200] * 64
        self.assertEqual(metadata, d)

    def test_get_loss(self):
        model = NetUnifiedBatchFeed()
        dummy_val_loader = list(range(2))
        re_runner = PyTorchModelPredictor(model, dummy_val_loader)

        loss = re_runner.model_get_loss(nn.CrossEntropyLoss())

        self.assertEqual(loss, 1.0)
        self.assertEqual(model.dummy_batch.item_ctr, 2)
        self.assertEqual(model.dummy_batch.back_ctr, 0)

    def test_get_loss_separate_batch_feed(self):
        model = Net()
        batch_loader = DeactivateModelFeedDefinition()
        model_wrap = ModelWrap(model=model, batch_model_feed_def=batch_loader)

        dummy_val_loader = list(range(2))
        re_runner = PyTorchModelPredictor(model_wrap, dummy_val_loader)

        loss = re_runner.model_get_loss(nn.CrossEntropyLoss())

        self.assertEqual(loss, 1.0)
        self.assertEqual(batch_loader.dummy_batch.item_ctr, 2)
        self.assertEqual(batch_loader.dummy_batch.back_ctr, 0)
        
    def test_evaluate_result_package(self):
        model = NetUnifiedBatchFeed()
        dummy_val_loader = list(range(2))
        re_runner = PyTorchModelPredictor(model, dummy_val_loader)

        result_pkg = DummyResultPackageExtend()
        result_pkg_return = re_runner.evaluate_result_package(result_package=result_pkg, return_result_package=True)

        self.assertEqual(result_pkg, result_pkg_return)
        self.assertEqual(result_pkg_return.results_dict, {'dummy': 111, 'extended_dummy': 1323123.44})
        self.assertEqual(result_pkg_return.get_results(), {'dummy': 111, 'extended_dummy': 1323123.44})
        self.assertEqual(result_pkg_return.ctr, 12.)

        y_test = result_pkg_return.y_true
        y_pred = result_pkg_return.y_predicted
        metadata = result_pkg_return.additional_results['additional_results']

        r = []
        for i in range(1, len(dummy_val_loader) + 1):
            r += [i] * 64
        r2 = []
        for i in range(1, len(dummy_val_loader) + 1):
            r2 += [i + 100] * 64
        self.assertEqual(y_test.tolist(), r)
        self.assertEqual(y_pred.tolist(), r2)

        d = {'bla': []}
        for i in range(1, len(dummy_val_loader) + 1):
            d['bla'] += [i + 200] * 64
        self.assertEqual(metadata, d)

    def test_evaluate_result_package_separate_batch_feed(self):
        model = Net()
        dummy_feed_def = DeactivateModelFeedDefinition()
        model_wrap = ModelWrap(model=model, batch_model_feed_def=dummy_feed_def)

        dummy_val_loader = list(range(2))
        re_runner = PyTorchModelPredictor(model_wrap, dummy_val_loader)

        result_pkg = DummyResultPackageExtend()
        result_pkg_return = re_runner.evaluate_result_package(result_package=result_pkg, return_result_package=True)

        self.assertEqual(result_pkg, result_pkg_return)
        self.assertEqual(result_pkg_return.results_dict, {'dummy': 111, 'extended_dummy': 1323123.44})
        self.assertEqual(result_pkg_return.get_results(), {'dummy': 111, 'extended_dummy': 1323123.44})
        self.assertEqual(result_pkg_return.ctr, 12.)

        y_test = result_pkg_return.y_true
        y_pred = result_pkg_return.y_predicted
        metadata = result_pkg_return.additional_results['additional_results']

        r = []
        for i in range(1, len(dummy_val_loader) + 1):
            r += [i] * 64
        r2 = []
        for i in range(1, len(dummy_val_loader) + 1):
            r2 += [i + 100] * 64
        self.assertEqual(y_test.tolist(), r)
        self.assertEqual(y_pred.tolist(), r2)

        d = {'bla': []}
        for i in range(1, len(dummy_val_loader) + 1):
            d['bla'] += [i + 200] * 64
        self.assertEqual(metadata, d)

    def test_evaluate_result_package_get_results(self):
        model = NetUnifiedBatchFeed()
        dummy_val_loader = list(range(2))
        re_runner = PyTorchModelPredictor(model, dummy_val_loader)

        result_pkg = DummyResultPackageExtend()
        result_dict = re_runner.evaluate_result_package(result_package=result_pkg, return_result_package=False)

        self.assertEqual(result_dict, {'dummy': 111, 'extended_dummy': 1323123.44})
