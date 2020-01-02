import unittest

from aitoolbox.utils import util
from tests.utils import DummyOptimizer


class TestUtil(unittest.TestCase):
    def test_function_exists(self):
        self.assertTrue(util.function_exists(DummyOptimizer(), 'state_dict'))
        self.assertTrue(util.function_exists(DummyOptimizer(), 'zero_grad'))

        self.assertFalse(util.function_exists(DummyOptimizer(), 'missing_fn'))
        self.assertFalse(util.function_exists(DummyOptimizer(), 'zero_grad_ctr'))
        self.assertFalse(util.function_exists(DummyOptimizer(), 'step_ctr'))

    def test_is_empty_function(self):
        def empty_fn():
            pass

        def empty_with_doc_fn():
            """

            Returns:

            """
            pass

        def return_only_fn():
            return "value"

        def full_fn():
            print('aaaa')

        def full_fn_arg(a):
            return a

        def full_fn_arg_sum(a, b):
            c = a + b
            return c

        self.assertTrue(util.is_empty_function(empty_fn))
        self.assertTrue(util.is_empty_function(empty_with_doc_fn))
        self.assertFalse(util.is_empty_function(return_only_fn))
        self.assertFalse(util.is_empty_function(full_fn))
        self.assertFalse(util.is_empty_function(full_fn_arg))
        self.assertFalse(util.is_empty_function(full_fn_arg_sum))

    def test_is_empty_function_inside_object(self):
        obj = EmptyFunctions()

        self.assertTrue(util.is_empty_function(obj.empty_fn))
        self.assertTrue(util.is_empty_function(obj.empty_with_doc_fn))
        self.assertFalse(util.is_empty_function(obj.return_only_fn))
        self.assertFalse(util.is_empty_function(obj.full_fn))
        self.assertFalse(util.is_empty_function(obj.full_fn_arg))
        self.assertFalse(util.is_empty_function(obj.full_fn_arg_sum))

    def test_flatten_list_of_lists(self):
        self.assertEqual(util.flatten_list_of_lists([[1, 2, 3], [4, 5], [3, 3, 3, 3]]),
                         [1, 2, 3, 4, 5, 3, 3, 3, 3])


class EmptyFunctions:
    def __init__(self):
        self.a = 2

    def empty_fn(self):
        pass

    def empty_with_doc_fn(self):
        """

        Returns:

        """
        pass

    def return_only_fn(self):
        return "value" + str(self.a)

    @staticmethod
    def full_fn(self):
        print('aaaa')

    @staticmethod
    def full_fn_arg(self, a):
        return a

    def full_fn_arg_sum(self, a, b):
        c = a + b + self.a
        return c
