import unittest
from aitoolbox.nlp.experiment_evaluation.NLP_metrics import *


class TestROUGEMetric(unittest.TestCase):
    def test_calculate_metric(self):
        rogue1 = ROUGEMetric(['bla bla bla'.split()], ['bla bla bla'.split()])
        self.compare_rogue_dict(rogue1.get_metric(),
                                {'rouge-1': {'f': 0.999999995, 'p': 1.0, 'r': 1.0},
                                 'rouge-2': {'f': 0.999999995, 'p': 1.0, 'r': 1.0},
                                 'rouge-l': {'f': 0.999999995, 'p': 1.0, 'r': 1.0}})
        self.compare_rogue_dict(rogue1.get_metric_dict(),
                                {rogue1.metric_name:
                                     {'rouge-1': {'f': 0.999999995, 'p': 1.0, 'r': 1.0},
                                      'rouge-2': {'f': 0.999999995, 'p': 1.0, 'r': 1.0},
                                      'rouge-l': {'f': 0.999999995, 'p': 1.0, 'r': 1.0}}})

        rogue2 = ROUGEMetric(['Today the sun shines not'.split()], ['Today the sun does shine'.split()])
        self.assertEqual(rogue2.get_metric(),
                         {'rouge-1': {'f': 0.5999999950000001, 'p': 0.6, 'r': 0.6},
                          'rouge-2': {'f': 0.4999999950000001, 'p': 0.5, 'r': 0.5},
                          'rouge-l': {'f': 0.5999999950000001, 'p': 0.6, 'r': 0.6}})
        rogue2_actual_txt = ROUGEMetric(['Today the sun shines not'], ['Today the sun does shine'.split()], target_actual_text=True)
        self.assertEqual(rogue2_actual_txt.get_metric(),
                         {'rouge-1': {'f': 0.5999999950000001, 'p': 0.6, 'r': 0.6},
                          'rouge-2': {'f': 0.4999999950000001, 'p': 0.5, 'r': 0.5},
                          'rouge-l': {'f': 0.5999999950000001, 'p': 0.6, 'r': 0.6}})

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


class TestExactMatchMetric(unittest.TestCase):
    def test_calculate_metric(self):
        em_match = ExactMatchTextMetric(['bla bla bla'.split()], ['bla bla bla'.split()])
        self.assertEqual(em_match.get_metric(), 100.)

        em_non_match = ExactMatchTextMetric(['bla not match bla'.split()], ['bla bla bla'.split()])
        self.assertEqual(em_non_match.get_metric(), 0.)

        em_semi_match = ExactMatchTextMetric(['bla bla bla'.split(), 'bla NON bla'.split()],
                                             ['bla bla bla'.split(), 'bla bla bla'.split()])
        self.assertEqual(em_semi_match.get_metric(), 50.)

        em_semi_match_2 = ExactMatchTextMetric(['bla bla bla'.split(), 'bla NON dasdadad dqewq'.split(),
                                                'bla blla'.split(), 'Today is sunny'.split()],
                                               ['bla bla bla'.split(), 'bla bla bla'.split(),
                                                'uuuu'.split(), 'Today is Not sunny'.split()])
        self.assertEqual(em_semi_match_2.get_metric(), 25.)

    def test_raise_exception(self):
        with self.assertRaises(ValueError):
            ExactMatchTextMetric(['bla bla bla'.split()], ['bla bla bla'.split(), 'bla bla bla'.split()])


class TestF1TextMetric(unittest.TestCase):
    def test_calculate_metric(self):
        em_match = F1TextMetric(['bla bla bla'.split()], ['bla bla bla'.split()])
        self.assertEqual(em_match.get_metric(), 100.)

        em_non_match = F1TextMetric(['not match'.split()], ['bla bla bla'.split()])
        self.assertEqual(em_non_match.get_metric(), 0.)

        em_semi_match = F1TextMetric(['bla bla bla'.split(), 'bla NON bla'.split()],
                                     ['bla bla bla'.split(), 'bla bla bla'.split()])
        self.assertEqual(em_semi_match.get_metric(), 83.33333333333333)

        em_semi_match_2 = F1TextMetric(['bla bla bla'.split(), 'bla NON dasdadad dqewq'.split(),
                                        'bla blla'.split(), 'Today is sunny'.split()],
                                       ['bla bla bla'.split(), 'bla bla bla'.split(),
                                        'uuuu'.split(), 'Today is Not sunny'.split()])
        self.assertEqual(em_semi_match_2.get_metric(), 53.57142857142857)

    def test_raise_exception(self):
        with self.assertRaises(ValueError):
            F1TextMetric(['bla bla bla'.split()], ['bla bla bla'.split(), 'bla bla bla'.split()])


class TestBLEUSentenceScoreMetric(unittest.TestCase):
    def test_single_sentence_calculate_metric(self):
        bleu_1 = BLEUSentenceScoreMetric(['bla bla bla bla'.split()], ['bla bla bla bla'.split()])
        self.assertEqual(bleu_1.get_metric(), 1.)
        self.assertEqual(bleu_1.get_metric_dict(), {'BLEU_sentence_score': 1.})

        bleu_2 = BLEUSentenceScoreMetric(['Today the sun does shines'.split()], ['Today the sun does shine'.split()])
        self.assertAlmostEqual(bleu_2.get_metric(), 0.668740304976422)
        self.assertEqual(bleu_2.get_metric_dict(), {'BLEU_sentence_score': bleu_2.get_metric()})
        self.assertEqual(bleu_2.get_metric_dict(), {bleu_2.metric_name: bleu_2.metric_result})
        self.assertEqual(bleu_2.get_metric_dict(), {bleu_2.metric_name: bleu_2.get_metric()})

        bleu_3 = BLEUSentenceScoreMetric(['Today the sun does not shine'.split()], ['Today the sun does shine'.split()])
        self.assertAlmostEqual(bleu_3.get_metric(), 0.5789300674674098)

        bleu_4 = BLEUSentenceScoreMetric(['Today the sun does not shine'.split()], ['Today it is cloudy'.split()])
        self.assertAlmostEqual(bleu_4.get_metric(), 0.)

        bleu_5 = BLEUSentenceScoreMetric(['Today the sun does not shine'.split()],
                                         ['Today it is cloudy and the sun does not shine'.split()])
        self.assertAlmostEqual(bleu_5.get_metric(), 0.4111336169005197)

        bleu_6 = BLEUSentenceScoreMetric(['Today the sun does not shine'.split()],
                                         ["Today it is cloudy and the sun doesn't shine".split()])
        self.assertAlmostEqual(bleu_6.get_metric(), 0.)
        self.assertEqual(bleu_6.get_metric_dict(), {'BLEU_sentence_score': bleu_6.get_metric()})
        self.assertEqual(bleu_6.get_metric_dict(), {bleu_6.metric_name: bleu_6.metric_result})
        self.assertEqual(bleu_6.get_metric_dict(), {bleu_6.metric_name: bleu_6.get_metric()})

    def test_multiple_sentence_calculate_metric(self):
        bleu_1 = BLEUSentenceScoreMetric(['bla bla bla bla'.split(), 'mjaw how how mjav mjaw'.split()],
                                         ['bla bla bla bla'.split(), 'mjaw how how mjav mjaw'.split()])
        self.assertAlmostEqual(bleu_1.get_metric(), 1.)
        self.assertEqual(bleu_1.get_metric_dict(), {'BLEU_sentence_score': 1.})
        self.assertEqual(bleu_1.get_metric_dict(), {'BLEU_sentence_score': bleu_1.get_metric()})
        self.assertEqual(bleu_1.get_metric_dict(), {bleu_1.metric_name: bleu_1.metric_result})
        self.assertEqual(bleu_1.get_metric_dict(), {bleu_1.metric_name: bleu_1.get_metric()})

        bleu_2 = BLEUSentenceScoreMetric(['bla bla bla bla'.split(), 'mjaw how bla mjav mjaw'.split()],
                                         ['bla mjav bla bla'.split(), 'mjaw how how mjav mjaw'.split()])
        self.assertAlmostEqual(bleu_2.get_metric(), 0.)

        bleu_3 = BLEUSentenceScoreMetric(['bla bla bla bla'.split(), 'mjaw how how mjav mjaw'.split()],
                                         ['bla mjav bla bla bla'.split(), 'mjaw how how mjav mjaw'.split()])
        self.assertAlmostEqual(bleu_3.get_metric(), 0.5)

        bleu_4 = BLEUSentenceScoreMetric(['bla bla bla bla'.split(), 'mjaw how how how how mjav mjaw'.split()],
                                         ['bla mjav bla bla bla'.split(), 'mjaw how how mjav mjaw'.split()])
        self.assertAlmostEqual(bleu_4.get_metric(), 0.281834861892037)

        bleu_5 = BLEUSentenceScoreMetric(['bla bla bla bla'.split(), 'mjaw how how how how bla mjav mjaw'.split()],
                                         ['bla mjav bla bla bla'.split(), 'mjaw how how mjav mjaw'.split()])
        self.assertAlmostEqual(bleu_5.get_metric(), 0.)

    def test_catch_not_equal_num_of_examples(self):
        with self.assertRaises(ValueError):
            BLEUSentenceScoreMetric(['bla bla bla bla'.split(), 'mjaw how how mjav mjaw'.split()],
                                    ['bla bla bla bla'.split()])

        with self.assertRaises(ValueError):
            BLEUSentenceScoreMetric(['bla bla bla bla'.split(), 'mjaw how how mjav mjaw'.split(), 'dsad qwqw'.split()],
                                    ['bla bla bla bla'.split()])

        with self.assertRaises(ValueError):
            BLEUSentenceScoreMetric(['bla bla bla bla'.split(), 'mjaw how how mjav mjaw'.split(), 'dsad qwqw'.split()],
                                    ['bla bla bla bla'.split(), '1122 mm44 fadfdas'.split()])


class TestBLEUCorpusScoreMetric(unittest.TestCase):
    def test_single_sentence_calculate_metric(self):
        bleu_1 = BLEUCorpusScoreMetric(['bla bla bla bla'.split()], ['bla bla bla bla'.split()])
        self.assertEqual(bleu_1.get_metric(), 1.)
        self.assertEqual(bleu_1.get_metric_dict(), {'BLEU_corpus_score': 1.})

        bleu_2 = BLEUCorpusScoreMetric(['Today the sun does shines'.split()], ['Today the sun does shine'.split()])
        self.assertAlmostEqual(bleu_2.get_metric(), 0.668740304976422)
        self.assertEqual(bleu_2.get_metric_dict(), {'BLEU_corpus_score': bleu_2.get_metric()})
        self.assertEqual(bleu_2.get_metric_dict(), {bleu_2.metric_name: bleu_2.metric_result})
        self.assertEqual(bleu_2.get_metric_dict(), {bleu_2.metric_name: bleu_2.get_metric()})

        bleu_3 = BLEUCorpusScoreMetric(['Today the sun does not shine'.split()], ['Today the sun does shine'.split()])
        self.assertAlmostEqual(bleu_3.get_metric(), 0.5789300674674098)

        bleu_4 = BLEUCorpusScoreMetric(['Today the sun does not shine'.split()], ['Today it is cloudy'.split()])
        self.assertAlmostEqual(bleu_4.get_metric(), 0.)

        bleu_5 = BLEUCorpusScoreMetric(['Today the sun does not shine'.split()],
                                         ['Today it is cloudy and the sun does not shine'.split()])
        self.assertAlmostEqual(bleu_5.get_metric(), 0.4111336169005197)

        bleu_6 = BLEUCorpusScoreMetric(['Today the sun does not shine'.split()],
                                         ["Today it is cloudy and the sun doesn't shine".split()])
        self.assertAlmostEqual(bleu_6.get_metric(), 0.)
        self.assertEqual(bleu_6.get_metric_dict(), {'BLEU_corpus_score': bleu_6.get_metric()})
        self.assertEqual(bleu_6.get_metric_dict(), {bleu_6.metric_name: bleu_6.metric_result})
        self.assertEqual(bleu_6.get_metric_dict(), {bleu_6.metric_name: bleu_6.get_metric()})

    def test_sss(self):
        bleu_1 = BLEUCorpusScoreMetric(['bla bla bla bla'.split(), 'mjaw how how mjav mjaw'.split()],
                                         ['bla bla bla bla'.split(), 'mjaw how how mjav mjaw'.split()])
        self.assertAlmostEqual(bleu_1.get_metric(), 1.)
        self.assertEqual(bleu_1.get_metric_dict(), {'BLEU_corpus_score': 1.})
        self.assertEqual(bleu_1.get_metric_dict(), {'BLEU_corpus_score': bleu_1.get_metric()})
        self.assertEqual(bleu_1.get_metric_dict(), {bleu_1.metric_name: bleu_1.metric_result})
        self.assertEqual(bleu_1.get_metric_dict(), {bleu_1.metric_name: bleu_1.get_metric()})

        bleu_2 = BLEUCorpusScoreMetric(['bla bla bla bla'.split(), 'mjaw how bla mjav mjaw'.split()],
                                       ['bla mjav bla bla'.split(), 'mjaw how how mjav mjaw'.split()])
        self.assertAlmostEqual(bleu_2.get_metric(), 0.)

        bleu_3 = BLEUCorpusScoreMetric(['bla bla bla bla'.split(), 'mjaw how how mjav mjaw'.split()],
                                       ['bla mjav bla bla bla'.split(), 'mjaw how how mjav mjaw'.split()])
        self.assertAlmostEqual(bleu_3.get_metric(), 0.6887246539984299)

        bleu_4 = BLEUCorpusScoreMetric(['bla bla bla bla'.split(), 'mjaw how how how how mjav mjaw'.split()],
                                       ['bla mjav bla bla bla'.split(), 'mjaw how how mjav mjaw'.split()])
        self.assertAlmostEqual(bleu_4.get_metric(), 0.5240330551337333)

        bleu_5 = BLEUCorpusScoreMetric(['bla bla bla bla'.split(), 'mjaw how how how how bla mjav mjaw'.split()],
                                       ['bla mjav bla bla bla'.split(), 'mjaw how how mjav mjaw'.split()])
        self.assertAlmostEqual(bleu_5.get_metric(), 0.)

    def test_catch_not_equal_num_of_examples(self):
        with self.assertRaises(ValueError):
            BLEUCorpusScoreMetric(['bla bla bla bla'.split(), 'mjaw how how mjav mjaw'.split()],
                                  ['bla bla bla bla'.split()])

        with self.assertRaises(ValueError):
            BLEUCorpusScoreMetric(['bla bla bla bla'.split(), 'mjaw how how mjav mjaw'.split(), 'dsad qwqw'.split()],
                                  ['bla bla bla bla'.split()])

        with self.assertRaises(ValueError):
            BLEUCorpusScoreMetric(['bla bla bla bla'.split(), 'mjaw how how mjav mjaw'.split(), 'dsad qwqw'.split()],
                                  ['bla bla bla bla'.split(), '1122 mm44 fadfdas'.split()])


class TestBLEUScoreStrTorchNLPMetric(unittest.TestCase):
    def test_single_sentence_calculate_metric(self):
        bleu_1 = BLEUScoreStrTorchNLPMetric(['bla bla bla bla'.split()], ['bla bla bla bla'.split()])
        self.assertEqual(bleu_1.get_metric(), 100.)
        self.assertEqual(bleu_1.get_metric_dict(), {'BLEU_str_torchNLP_score': 100.})

        bleu_2 = BLEUScoreStrTorchNLPMetric(['Today the sun does shines'.split()], ['Today the sun does shine'.split()])
        self.assertAlmostEqual(bleu_2.get_metric(), 66.87000274658203)
        self.assertEqual(bleu_2.get_metric_dict(), {'BLEU_str_torchNLP_score': bleu_2.get_metric()})
        self.assertEqual(bleu_2.get_metric_dict(), {bleu_2.metric_name: bleu_2.metric_result})
        self.assertEqual(bleu_2.get_metric_dict(), {bleu_2.metric_name: bleu_2.get_metric()})

        bleu_3 = BLEUScoreStrTorchNLPMetric(['Today the sun does not shine'.split()], ['Today the sun does shine'.split()])
        self.assertAlmostEqual(bleu_3.get_metric(), 53.72999954223633)

        bleu_4 = BLEUScoreStrTorchNLPMetric(['Today the sun does not shine'.split()], ['Today it is cloudy'.split()])
        self.assertAlmostEqual(bleu_4.get_metric(), 0.)

        bleu_5 = BLEUScoreStrTorchNLPMetric(['Today the sun does not shine'.split()],
                                            ['Today it is cloudy and the sun does not shine'.split()])
        self.assertAlmostEqual(bleu_5.get_metric(), 40.83000183105469)

        bleu_6 = BLEUScoreStrTorchNLPMetric(['Today the sun does not shine'.split()],
                                            ["Today it is cloudy and the sun doesn't shine".split()])
        self.assertAlmostEqual(bleu_6.get_metric(), 0.)
        self.assertEqual(bleu_6.get_metric_dict(), {'BLEU_str_torchNLP_score': bleu_6.get_metric()})
        self.assertEqual(bleu_6.get_metric_dict(), {bleu_6.metric_name: bleu_6.metric_result})
        self.assertEqual(bleu_6.get_metric_dict(), {bleu_6.metric_name: bleu_6.get_metric()})

    def test_sss(self):
        bleu_1 = BLEUScoreStrTorchNLPMetric(['bla bla bla bla'.split(), 'mjaw how how mjav mjaw'.split()],
                                            ['bla bla bla bla'.split(), 'mjaw how how mjav mjaw'.split()])
        self.assertAlmostEqual(bleu_1.get_metric(), 100.)
        self.assertEqual(bleu_1.get_metric_dict(), {'BLEU_str_torchNLP_score': 100.})
        self.assertEqual(bleu_1.get_metric_dict(), {'BLEU_str_torchNLP_score': bleu_1.get_metric()})
        self.assertEqual(bleu_1.get_metric_dict(), {bleu_1.metric_name: bleu_1.metric_result})
        self.assertEqual(bleu_1.get_metric_dict(), {bleu_1.metric_name: bleu_1.get_metric()})

        bleu_2 = BLEUScoreStrTorchNLPMetric(['bla bla bla bla'.split(), 'mjaw how bla mjav mjaw'.split()],
                                            ['bla mjav bla bla'.split(), 'mjaw how how mjav mjaw'.split()])
        self.assertAlmostEqual(bleu_2.get_metric(), 0.)

        bleu_3 = BLEUScoreStrTorchNLPMetric(['bla bla bla bla'.split(), 'mjaw how how mjav mjaw'.split()],
                                            ['bla mjav bla bla bla'.split(), 'mjaw how how mjav mjaw'.split()])
        self.assertAlmostEqual(bleu_3.get_metric(), 50.0)

        bleu_4 = BLEUScoreStrTorchNLPMetric(['bla bla bla bla'.split(), 'mjaw how how how how mjav mjaw'.split()],
                                            ['bla mjav bla bla bla'.split(), 'mjaw how how mjav mjaw'.split()])
        self.assertAlmostEqual(bleu_4.get_metric(), 25.850000381469727)

        bleu_5 = BLEUScoreStrTorchNLPMetric(['bla bla bla bla'.split(), 'mjaw how how how how bla mjav mjaw'.split()],
                                       ['bla mjav bla bla bla'.split(), 'mjaw how how mjav mjaw'.split()])
        self.assertAlmostEqual(bleu_5.get_metric(), 0.)

    def test_catch_not_equal_num_of_examples(self):
        with self.assertRaises(ValueError):
            BLEUScoreStrTorchNLPMetric(['bla bla bla bla'.split(), 'mjaw how how mjav mjaw'.split()],
                                       ['bla bla bla bla'.split()])

        with self.assertRaises(ValueError):
            BLEUScoreStrTorchNLPMetric(['bla bla bla bla'.split(), 'mjaw how how mjav mjaw'.split(), 'dsad qwqw'.split()],
                                       ['bla bla bla bla'.split()])

        with self.assertRaises(ValueError):
            BLEUScoreStrTorchNLPMetric(['bla bla bla bla'.split(), 'mjaw how how mjav mjaw'.split(), 'dsad qwqw'.split()],
                                       ['bla bla bla bla'.split(), '1122 mm44 fadfdas'.split()])
