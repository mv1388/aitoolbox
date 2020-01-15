import unittest
import numpy as np

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

    def test_2d_vector_non_argmax(self):
        y_pred = np.array([0, 1, 0, 0, 1, 1]).reshape((-1, 1))
        y_true = np.array([1, 0, 1, 0, 1, 1]).reshape((-1, 1))
        acc_1 = AccuracyMetric(y_true, y_pred, positive_class_thresh=None)
        self.assertEqual(acc_1.get_metric(), 0.5)
        self.assertEqual(acc_1.get_metric_dict(), {'Accuracy': 0.5})

        y_pred = np.array([0, 1, 0, 2, 3, 1]).reshape((-1, 1))
        y_true = np.array([1, 0, 1, 2, 3, 1]).reshape((-1, 1))
        acc_2 = AccuracyMetric(y_true, y_pred, positive_class_thresh=None)
        self.assertEqual(acc_2.get_metric(), 0.5)
        self.assertEqual(acc_2.get_metric_dict(), {'Accuracy': 0.5})

        y_pred = np.array([0, 1, 0, 2, 3, 1])
        y_true = np.array([1, 0, 1, 2, 3, 1])
        acc_3 = AccuracyMetric(y_true, y_pred, positive_class_thresh=None)
        self.assertEqual(acc_3.get_metric(), 0.5)
        self.assertEqual(acc_3.get_metric_dict(), {'Accuracy': 0.5})

    def test_threshold_binary_classification(self):
        y_pred = np.array([0.1, 0.6, 0.3, 0.2, 1.0, 0.9]).reshape((-1, 1))
        y_true = np.array([1, 0, 1, 0, 1, 1]).reshape((-1, 1))
        acc_1 = AccuracyMetric(y_true, y_pred, positive_class_thresh=0.5)
        self.assertEqual(acc_1.get_metric(), 0.5)
        self.assertEqual(acc_1.get_metric_dict(), {'Accuracy': 0.5})

        y_pred = np.array([0.1, 0.6, 0.3, 0.2, 1.0, 0.9])
        y_true = np.array([1, 0, 1, 0, 1, 1])
        acc_2 = AccuracyMetric(y_true, y_pred, positive_class_thresh=0.5)
        self.assertEqual(acc_2.get_metric(), 0.5)
        self.assertEqual(acc_2.get_metric_dict(), {'Accuracy': 0.5})
