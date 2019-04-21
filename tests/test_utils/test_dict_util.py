import unittest

import AIToolbox.utils.dict_util as dict_util


class TestFlattenDict(unittest.TestCase):
    def test_flatten_dict(self):
        input_dict = {'bla': 12, 'www': 455, 'pppp': 4004}
        self.assertEqual(dict_util.flatten_dict(input_dict), input_dict)
        
        input_dict_1_level = {'bla': {'uuu': 334, 'www': 1010}, 'rogue': {'ppp': 123}}
        self.assertEqual(dict_util.flatten_dict(input_dict_1_level),
                         {'bla_uuu': 334, 'bla_www': 1010, 'rogue_ppp': 123})
        
        input_dict_2_level = {'bla': {'rogue_1': {'m': 10, 'p': 11}, 'rogue_2': {'m': 20, 'p': 22}}}
        self.assertEqual(dict_util.flatten_dict(input_dict_2_level),
                         {'bla_rogue_1_m': 10, 'bla_rogue_1_p': 11, 'bla_rogue_2_m': 20, 'bla_rogue_2_p': 22})
