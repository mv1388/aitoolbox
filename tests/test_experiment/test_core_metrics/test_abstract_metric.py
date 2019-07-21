import unittest

from tests.utils import *


class TestAbstractBaseMetric(unittest.TestCase):
    def test_basic(self):
        val = 123
        metric = DummyAbstractBaseMetric(val)

        self.assertEqual(metric.get_metric(), val)
        self.assertEqual(metric.get_metric_dict(), {'dummy_metric': val})
        self.assertEqual(str(metric), f'dummy_metric: {val}')
        self.assertEqual(len(metric), 1)

    def test_metric_compare(self):
        metric_1 = DummyAbstractBaseMetric(10)
        metric_2 = DummyAbstractBaseMetric(12)
        metric_3 = DummyAbstractBaseMetric(10)

        self.assertTrue(metric_1 < metric_2)
        self.assertTrue(metric_1 <= metric_2)
        self.assertTrue(metric_1 <= metric_3)
        self.assertFalse(metric_1 > metric_2)
        self.assertFalse(metric_1 >= metric_2)
        self.assertTrue(metric_1 >= metric_3)

        self.assertTrue(metric_1 < 12)
        self.assertTrue(metric_1 <= 12.0)
        self.assertTrue(metric_1 <= 10.)
        self.assertFalse(10. > metric_2)
        self.assertFalse(10 >= metric_2)
        self.assertTrue(metric_1 >= 10)

        metric_fail = DummyAbstractBaseMetric(10)
        metric_fail.metric_result = {'more_complex_metric': 34, 'bla': 44}

        with self.assertRaises(ValueError):
            result = metric_fail > 3

        metric_fail = DummyAbstractBaseMetric(10)
        metric_fail.metric_result = [3213, 554]

        with self.assertRaises(ValueError):
            result = metric_fail > 3

        with self.assertRaises(ValueError):
            result = metric_fail > metric_1

        with self.assertRaises(ValueError):
            result = metric_1 < metric_fail

    def test_metric_contains(self):
        metric_1 = DummyAbstractBaseMetric(10)
        self.assertTrue('dummy_metric' in metric_1)
        self.assertFalse('dummy_metricFAIL' in metric_1)

        metric_dict = DummyAbstractBaseMetric(10)
        metric_dict.metric_result = {'more_complex_metric': 34, 'bla': 44}
        self.assertTrue('dummy_metric' in metric_dict)
        self.assertFalse('dummy_metricFAIL' in metric_dict)
        self.assertTrue('more_complex_metric' in metric_dict)
        self.assertTrue('bla' in metric_dict)
        self.assertFalse('blaFAIL' in metric_dict)

    def test_metric_get_item(self):
        metric_1 = DummyAbstractBaseMetric(10)
        self.assertEqual(metric_1['dummy_metric'], 10)
        with self.assertRaises(KeyError):
            val = metric_1['dummy_metricFAIL']

        metric_dict = DummyAbstractBaseMetric(10)
        metric_dict.metric_result = {'more_complex_metric': 34, 'bla': 44}
        self.assertEqual(metric_dict['dummy_metric'], {'more_complex_metric': 34, 'bla': 44})
        with self.assertRaises(KeyError):
            val = metric_dict['dummy_metricFAIL']
        self.assertEqual(metric_dict['more_complex_metric'], 34)
        self.assertEqual(metric_dict['bla'], 44)
        with self.assertRaises(KeyError):
            val = metric_dict['blaFAIL']
