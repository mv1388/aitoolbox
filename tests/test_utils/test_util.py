import unittest

from AIToolbox.utils import util
from tests.utils import DummyOptimizer


class TestUtil(unittest.TestCase):
    def test_function_exists(self):
        self.assertTrue(util.function_exists(DummyOptimizer(), 'state_dict'))
        self.assertTrue(util.function_exists(DummyOptimizer(), 'zero_grad'))

        self.assertFalse(util.function_exists(DummyOptimizer(), 'missing_fn'))
        self.assertFalse(util.function_exists(DummyOptimizer(), 'zero_grad_ctr'))
        self.assertFalse(util.function_exists(DummyOptimizer(), 'step_ctr'))
