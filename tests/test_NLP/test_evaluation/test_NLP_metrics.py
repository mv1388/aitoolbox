import unittest
from AIToolbox.NLP.evaluation.NLP_metrics import *


class TestROGUENonOfficialMetric(unittest.TestCase):
    def test_calculate_metric(self):
        rogue1 = ROUGEMetric('bla bla bla', 'bla bla bla')
        self.assertEqual(rogue1.get_metric(),
                         {'rouge-1': {'f': 0.999999995, 'p': 1.0, 'r': 1.0},
                          'rouge-2': {'f': 0.999999995, 'p': 1.0, 'r': 1.0},
                          'rouge-l': {'f': 0.9999999999995, 'p': 1.0, 'r': 1.0}})
        self.assertEqual(rogue1.get_metric_dict(),
                         {rogue1.metric_name: {'rouge-1': {'f': 0.999999995, 'p': 1.0, 'r': 1.0},
                                               'rouge-2': {'f': 0.999999995, 'p': 1.0, 'r': 1.0},
                                               'rouge-l': {'f': 0.9999999999995, 'p': 1.0, 'r': 1.0}}})

        rogue2 = ROUGEMetric('Today the sun shines not', 'Today the sun does shine')
        self.assertEqual(rogue2.get_metric(),
                         {'rouge-1': {'f': 0.5999999950000001, 'p': 0.6, 'r': 0.6},
                          'rouge-2': {'f': 0.4999999950000001, 'p': 0.5, 'r': 0.5},
                          'rouge-l': {'f': 0.5999999999994999, 'p': 0.6, 'r': 0.6}})

        rogue3 = ROUGEMetric('today we go to the university', 'yesterday he played basketball very long')
        self.assertEqual(rogue3.get_metric(),
                         {'rouge-1': {'f': 0.0, 'p': 0.0, 'r': 0.0}, 'rouge-2': {'f': 0.0, 'p': 0.0, 'r': 0.0},
                          'rouge-l': {'f': 0.0, 'p': 0.0, 'r': 0.0}})

        rogue4 = ROUGEMetric('today we go to the university', 'bla mjav how w!?')
        self.assertEqual(rogue4.get_metric(),
                         {'rouge-1': {'f': 0.0, 'p': 0.0, 'r': 0.0}, 'rouge-2': {'f': 0.0, 'p': 0.0, 'r': 0.0},
                          'rouge-l': {'f': 0.0, 'p': 0.0, 'r': 0.0}})


if __name__ == '__main__':
    unittest.main()
