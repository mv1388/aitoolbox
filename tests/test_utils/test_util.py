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

    def test_copy_function(self):
        src_fn_obj = SourceFn()
        my_fn_copy = util.copy_function(src_fn_obj.my_fn)
        my_fn_input_copy = util.copy_function(src_fn_obj.my_fn_input)

        self.assertEqual(my_fn_copy(None), 'my_fn_return_value')
        self.assertEqual(my_fn_input_copy(None, 'Value 1'), 'my_fn_return_value: Value 1')

    def test_copy_function_another_object(self):
        src_fn_obj = SourceFn()
        my_fn_copy = util.copy_function(src_fn_obj.my_fn)
        my_fn_input_copy = util.copy_function(src_fn_obj.my_fn_input)

        target_fn_obj = TargetFnCopy(my_fn_copy, my_fn_input_copy)

        self.assertEqual(target_fn_obj.copy_my_fn(), 'my_fn_return_value')
        self.assertEqual(target_fn_obj.copy_my_fn_input('Value 2'), 'my_fn_return_value: Value 2')

    def test_copy_function_another_object_fn_call_another_fn(self):
        src_fn_obj = SourceFnCallAnotherFn()
        my_fn_copy = util.copy_function(src_fn_obj.my_fn)
        my_fn_input_copy = util.copy_function(src_fn_obj.my_fn_input)

        target_fn_obj = TargetFnCopy(my_fn_copy, my_fn_input_copy)

        self.assertEqual(target_fn_obj.copy_my_fn(), 'my_fn_return_value: MyValue_another_fn_call')
        self.assertEqual(target_fn_obj.copy_my_fn_input('Value 2'), 'my_fn_return_value: Value 2')

    def test_copy_function_another_object_access_attribute(self):
        src_fn_obj = SourceFnAccessAttrVal()
        my_fn_copy = util.copy_function(src_fn_obj.my_fn)
        my_fn_input_copy = util.copy_function(src_fn_obj.my_fn_input)

        target_fn_obj = TargetFnCopy(my_fn_copy, my_fn_input_copy, attribute_val='my_attribute_value')

        self.assertEqual(target_fn_obj.copy_my_fn(), 'my_fn attribute value: my_attribute_value')
        self.assertEqual(target_fn_obj.copy_my_fn_input('Value 2'),
                         'my_fn_input attribute value: my_attribute_value; fn input value Value 2')

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

        self.assertEqual(
            util.flatten_list_of_lists(
                [
                    [[1, 2, 3], [4, 5, 3], [3, 3, 3]],
                    [[10, 2, 3], [40, 5, 3], [30, 3, 3]],
                    [[100, 2, 3], [400, 5, 3], [300, 3, 3]]
                ]),
            [[1, 2, 3], [4, 5, 3], [3, 3, 3], [10, 2, 3], [40, 5, 3], [30, 3, 3], [100, 2, 3], [400, 5, 3], [300, 3, 3]]
        )


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
    def full_fn():
        print('aaaa')

    @staticmethod
    def full_fn_arg(a):
        return a

    def full_fn_arg_sum(self, a, b):
        c = a + b + self.a
        return c


class SourceFn:
    def my_fn(self):
        return 'my_fn_return_value'

    def my_fn_input(self, value):
        return f'my_fn_return_value: {value}'


class SourceFnCallAnotherFn:
    def my_fn(self):
        return self.copy_my_fn_input('MyValue_another_fn_call')

    def my_fn_input(self, value):
        return f'my_fn_return_value: {value}'


class SourceFnAccessAttrVal:
    def my_fn(self):
        return f'my_fn attribute value: {self.attribute_val}'

    def my_fn_input(self, value):
        return f'my_fn_input attribute value: {self.attribute_val}; fn input value {value}'


class TargetFnCopy:
    def __init__(self, source_my_fn, source_my_fn_input, attribute_val='my_attribute_value'):
        self.source_my_fn = source_my_fn
        self.source_my_fn_input = source_my_fn_input

        self.attribute_val = attribute_val

    def copy_my_fn(self):
        return self.source_my_fn(self)

    def copy_my_fn_input(self, value):
        return self.source_my_fn_input(self, value)
