import unittest
from AIToolbox.NLP.evaluation.NLP_metrics import *


class TestROGUENonOfficialMetric(unittest.TestCase):
    def test_calculate_metric(self):
        rogue1 = ROUGEMetric(['bla bla bla'.split()], ['bla bla bla'.split()])
        self.compare_rogue_dict(rogue1.get_metric(),
                                {'rouge-1': {'f': 0.999999995, 'p': 1.0, 'r': 1.0},
                                 'rouge-2': {'f': 0.999999995, 'p': 1.0, 'r': 1.0},
                                 'rouge-l': {'f': 0.9999999999995, 'p': 1.0, 'r': 1.0}})
        self.compare_rogue_dict(rogue1.get_metric_dict(),
                                {rogue1.metric_name:
                                     {'rouge-1': {'f': 0.999999995, 'p': 1.0, 'r': 1.0},
                                      'rouge-2': {'f': 0.999999995, 'p': 1.0, 'r': 1.0},
                                      'rouge-l': {'f': 0.9999999999995, 'p': 1.0, 'r': 1.0}}})

        rogue2 = ROUGEMetric(['Today the sun shines not'.split()], ['Today the sun does shine'.split()])
        self.assertEqual(rogue2.get_metric(),
                         {'rouge-1': {'f': 0.5999999950000001, 'p': 0.6, 'r': 0.6},
                          'rouge-2': {'f': 0.4999999950000001, 'p': 0.5, 'r': 0.5},
                          'rouge-l': {'f': 0.5999999999994999, 'p': 0.6, 'r': 0.6}})
        rogue2_actual_txt = ROUGEMetric(['Today the sun shines not'], ['Today the sun does shine'.split()], target_actual_text=True)
        self.assertEqual(rogue2_actual_txt.get_metric(),
                         {'rouge-1': {'f': 0.5999999950000001, 'p': 0.6, 'r': 0.6},
                          'rouge-2': {'f': 0.4999999950000001, 'p': 0.5, 'r': 0.5},
                          'rouge-l': {'f': 0.5999999999994999, 'p': 0.6, 'r': 0.6}})

        rogue3 = ROUGEMetric(['today we go to the university'.split()], ['yesterday he played basketball very long'.split()])
        self.assertEqual(rogue3.get_metric(),
                         {'rouge-1': {'f': 0.0, 'p': 0.0, 'r': 0.0}, 'rouge-2': {'f': 0.0, 'p': 0.0, 'r': 0.0},
                          'rouge-l': {'f': 0.0, 'p': 0.0, 'r': 0.0}})

        rogue4 = ROUGEMetric(['today we go to the university'.split()], ['bla mjav how w!?'.split()])
        self.assertEqual(rogue4.get_metric(),
                         {'rouge-1': {'f': 0.0, 'p': 0.0, 'r': 0.0}, 'rouge-2': {'f': 0.0, 'p': 0.0, 'r': 0.0},
                          'rouge-l': {'f': 0.0, 'p': 0.0, 'r': 0.0}})

    def compare_rogue_dict(self, dict_1, dict_2):
        """

        Args:
            dict_1 (dict):
            dict_2 (dict):

        Returns:

        """
        self.assertEqual(sorted(dict_1.keys()), sorted(dict_2.keys()))

        for k1 in dict_1:
            self.assertEqual(sorted(dict_1[k1].keys()), sorted(dict_2[k1].keys()))

            for k2 in dict_1[k1]:
                self.assertAlmostEqual(dict_1[k1][k2], dict_2[k1][k2])


if __name__ == '__main__':
    unittest.main()
