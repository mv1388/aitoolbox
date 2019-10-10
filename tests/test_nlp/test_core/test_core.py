import unittest
from aitoolbox.nlp.core import core


class TestCore_find_sub_list(unittest.TestCase):
    def test_find_sub_list_do_find(self):
        self.assertEqual(
            core.find_sub_list([1, 2, 3], [1, 1, 1, 5, 1, 2, 3, 5, 6, 7]),
            (4, 6)
        )
        self.assertEqual(
            core.find_sub_list([1, 1], [1, 1, 1, 5, 1, 2, 3, 5, 6, 7]),
            (0, 1)
        )

    def test_find_sub_list_no_find(self):
        self.assertEqual(
            core.find_sub_list([10, 10], [1, 1, 1, 5, 1, 2, 3, 5, 6, 7]),
            None
        )
        self.assertEqual(
            core.find_sub_list([7, 7], [1, 1, 1, 5, 1, 2, 3, 5, 6, 7]),
            None
        )

    def test_find_sub_list_sublist_too_long_exception(self):
        self.assertRaises(
            ValueError,
            core.find_sub_list, [1, 1, 1], [1, 1]
        )
        self.assertRaises(
            ValueError,
            core.find_sub_list, range(10), range(5)
        )


class TestCore_normalize_string(unittest.TestCase):
    def test_single_sent(self):
        self.assertEqual(
            core.normalize_string('For this purpose, we propose a hierarchical attention model to capture the '
                                  'context in a structured and dynamic manner.'),
            'for this purpose we propose a hierarchical attention model to capture the context in a structured and dynamic manner .'
        )

    def test_numeric_include(self):
        self.assertEqual(
            core.normalize_string('For this purpose, we propose a 2-hierarchical 345 attention model to capture.'),
            'for this purpose we propose a 2 hierarchical 345 attention model to capture .'
        )
