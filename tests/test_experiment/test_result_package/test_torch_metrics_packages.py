import unittest

from tests.utils import DummyTorchMetrics

from aitoolbox.experiment.result_package.torch_metrics_packages import TorchMetricsPackage


class TestTorchMetricsPackage(unittest.TestCase):
    def test_prepare_results_dict_formatting_float_result(self):
        metric = DummyTorchMetrics(return_float=True)
        result_package = TorchMetricsPackage(metric)

        result_package.prepare_result_package([], [])
        result_dict = result_package.get_results()

        self.assertEqual(result_dict, {'DummyTorchMetrics_PTLMetrics': 123.4})

    def test_prepare_results_dict_formatting_dict_result(self):
        metric = DummyTorchMetrics(return_float=False)
        result_package = TorchMetricsPackage(metric)

        result_package.prepare_result_package([], [])
        result_dict = result_package.get_results()

        self.assertEqual(result_dict, {'metric_1_PTLMetrics': 12.34, 'metric_2_PTLMetrics': 56.78})

    def test_metric_compute(self):
        metric = DummyTorchMetrics()
        result_package = TorchMetricsPackage(metric)

        for i in range(100):
            result_package.metric_compute()
            self.assertEqual(metric.compute_ctr, i + 1)

    def test_metric_reset(self):
        metric = DummyTorchMetrics()
        result_package = TorchMetricsPackage(metric)

        for i in range(100):
            result_package.metric_reset()
            self.assertEqual(metric.reset_ctr, i + 1)
