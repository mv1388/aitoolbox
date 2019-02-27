import unittest
from AIToolbox.NLP.evaluation.NLP_metrics import *


class TestROGUENonOfficialMetric(unittest.TestCase):
    def test_calculate_metric(self):
        rogue1 = ROUGENonOfficialMetric('bla bla bla'.split(), 'bla bla bla'.split())
        self.assertEqual(rogue1.get_metric(),
                         {'rouge-1': {'f': 0.999999995, 'p': 1.0, 'r': 1.0}, 'rouge-2': {'f': 0.0, 'p': 0.0, 'r': 0.0},
                          'rouge-l': {'f': 0.9999999999995, 'p': 1.0, 'r': 1.0}})
        self.assertEqual(rogue1.get_metric_dict(),
                         {rogue1.metric_name: {'rouge-1': {'f': 0.999999995, 'p': 1.0, 'r': 1.0},
                                               'rouge-2': {'f': 0.0, 'p': 0.0, 'r': 0.0},
                                               'rouge-l': {'f': 0.9999999999995, 'p': 1.0, 'r': 1.0}}})

        rogue2 = ROUGENonOfficialMetric('Today the sun shines not'.split(), 'Today the sun does shine'.split())
        self.assertEqual(rogue2.get_metric(),
                         {'rouge-1': {'f': 0.5999999970000001, 'p': 0.6, 'r': 0.6},
                          'rouge-2': {'f': 0.0, 'p': 0.0, 'r': 0.0},
                          'rouge-l': {'f': 0.5999999999997, 'p': 0.6, 'r': 0.6}})

        rogue3 = ROUGENonOfficialMetric('today we go to the university'.split(),
                                        'yesterday he played basketball very long'.split())
        self.assertEqual(rogue3.get_metric(),
                         {'rouge-1': {'f': 0.0, 'p': 0.0, 'r': 0.0}, 'rouge-2': {'f': 0.0, 'p': 0.0, 'r': 0.0},
                          'rouge-l': {'f': 0.0, 'p': 0.0, 'r': 0.0}})


if __name__ == '__main__':
    unittest.main()
