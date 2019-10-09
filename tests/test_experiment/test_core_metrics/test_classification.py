import unittest

from tests.utils import *
from aitoolbox.experiment.core_metrics.classification import AccuracyMetric


class TestAccuracyMetric(unittest.TestCase):
    def test_basic(self):
        acc_1 = AccuracyMetric([1.] * 100, [1.] * 100)
        self.assertEqual(acc_1.get_metric(), 1.)
        self.assertEqual(acc_1.get_metric_dict(), {'Accuracy': 1.})

        acc_2 = AccuracyMetric([1., 0.], [1., 1.])
        self.assertEqual(acc_2.get_metric(), 0.5)
        self.assertEqual(acc_2.get_metric_dict(), {'Accuracy': 0.5})

    def test_2d_prediction(self):
        acc_1 = AccuracyMetric([1.] * 100, [[0, 1]] * 100)
        self.assertEqual(acc_1.get_metric(), 1.)
        self.assertEqual(acc_1.get_metric_dict(), {'Accuracy': 1.})

        acc_2 = AccuracyMetric([1.], [[0, 1]])
        self.assertEqual(acc_2.get_metric(), 1.)
        self.assertEqual(acc_2.get_metric_dict(), {'Accuracy': 1.})

        acc_3 = AccuracyMetric([1.], [[1, 0]])
        self.assertEqual(acc_3.get_metric(), 0.)
        self.assertEqual(acc_3.get_metric_dict(), {'Accuracy': 0.})

        acc_4 = AccuracyMetric([0, 1], [[0, 1], [0, 1]])
        self.assertEqual(acc_4.get_metric(), 0.5)
        self.assertEqual(acc_4.get_metric_dict(), {'Accuracy': 0.5})

        acc_5 = AccuracyMetric([0, 1], [[1, 0], [0, 1]])
        self.assertEqual(acc_5.get_metric(), 1.)
        self.assertEqual(acc_5.get_metric_dict(), {'Accuracy': 1.})
