import unittest

from aitoolbox.experiment.result_package.abstract_result_packages import MultipleResultPackageWrapper
from tests.utils import DummyResultPackageExtendVariable


class TestMultipleResultPackageWrapper(unittest.TestCase):
    def test_basic(self):
        result_d_1 = {'metric1': 33232, 'metric2': 1000}
        pkg_1 = DummyResultPackageExtendVariable(result_d_1)
        pkg_1.prepare_result_package([10] * 100, [11] * 100, {'dddd': 222})

        result_d_2 = {'metric3': 1, 'metric4': 2}
        pkg_2 = DummyResultPackageExtendVariable(result_d_2)
        pkg_2.prepare_result_package([10] * 100, [11] * 100, {'qqq': 445})

        multi_result = MultipleResultPackageWrapper()
        multi_result.prepare_result_package([pkg_1, pkg_2])

        self.assertEqual(str(multi_result),
                         f"--> {pkg_1.pkg_name}:\nmetric1: {result_d_1['metric1']}\nmetric2: {result_d_1['metric2']}\n"
                         f"--> {pkg_2.pkg_name}:\nmetric3: {result_d_2['metric3']}\nmetric4: {result_d_2['metric4']}")

        self.assertEqual(multi_result.get_results(), {pkg_1.pkg_name: result_d_1, f'{pkg_2.pkg_name}1': result_d_2})

    def test_concatenate_package(self):
        result_d_1 = {'metric1': 33232, 'metric2': 1000}
        pkg_1 = DummyResultPackageExtendVariable(result_d_1)
        pkg_1.prepare_result_package([10] * 100, [11] * 100, {'dddd': 222})

        result_d_2 = {'metric3': 1, 'metric4': 2}
        pkg_2 = DummyResultPackageExtendVariable(result_d_2)
        pkg_2.prepare_result_package([10] * 100, [11] * 100, {'qqq': 445})

        result_d_3 = {'metric1NEW': 33232, 'metric2NEW': 1000}
        pkg_3 = DummyResultPackageExtendVariable(result_d_3)
        pkg_3.prepare_result_package([10] * 100, [11] * 100, {'dddd': 222})

        result_d_4 = {'metric1NEW': 33232, 'metric2NEW': 1000}
        pkg_4 = DummyResultPackageExtendVariable(result_d_4)
        pkg_4.pkg_name = 'NEWPackageName'
        pkg_4.prepare_result_package([10] * 100, [11] * 100, {'dddd': 222})

        multi_result = MultipleResultPackageWrapper()
        multi_result.prepare_result_package([pkg_1, pkg_2])

        concat_package = multi_result + pkg_3
        self.assertEqual(concat_package.get_results(), {pkg_1.pkg_name: result_d_1, f'{pkg_2.pkg_name}1': result_d_2,
                                                        f'{pkg_3.pkg_name}2': result_d_3})
        # Not the same object - deepcopy done under the hood
        self.assertNotEqual(multi_result, concat_package)
        self.assertNotEqual(pkg_3, concat_package)

        concat_package_2 = multi_result + pkg_4
        self.assertEqual(concat_package_2.get_results(), {pkg_1.pkg_name: result_d_1, f'{pkg_2.pkg_name}1': result_d_2,
                                                          f'{pkg_4.pkg_name}': result_d_4})

        concat_package_full = concat_package + pkg_4
        self.assertEqual(concat_package_full.get_results(), {pkg_1.pkg_name: result_d_1, f'{pkg_2.pkg_name}1': result_d_2,
                                                             f'{pkg_3.pkg_name}2': result_d_3, f'{pkg_4.pkg_name}': result_d_4})
        # Not the same object - deepcopy done under the hood
        self.assertNotEqual(concat_package_full, concat_package)
        self.assertNotEqual(concat_package_full, pkg_4)

    def test_append_package(self):
        result_d_1 = {'metric1': 33232, 'metric2': 1000}
        pkg_1 = DummyResultPackageExtendVariable(result_d_1)
        pkg_1.prepare_result_package([10] * 100, [11] * 100, {'dddd': 222})

        result_d_2 = {'metric3': 1, 'metric4': 2}
        pkg_2 = DummyResultPackageExtendVariable(result_d_2)
        pkg_2.prepare_result_package([10] * 100, [11] * 100, {'qqq': 445})

        result_d_3 = {'metric1NEW': 33232, 'metric2NEW': 1000}
        pkg_3 = DummyResultPackageExtendVariable(result_d_3)
        pkg_3.prepare_result_package([10] * 100, [11] * 100, {'dddd': 222})

        result_d_4 = {'metric1NEW': 33232, 'metric2NEW': 1000}
        pkg_4 = DummyResultPackageExtendVariable(result_d_4)
        pkg_4.pkg_name = 'NEWPackageName'
        pkg_4.prepare_result_package([10] * 100, [11] * 100, {'dddd': 222})

        multi_result = MultipleResultPackageWrapper()
        multi_result.prepare_result_package([pkg_1, pkg_2])

        multi_result += pkg_3

        self.assertEqual(multi_result.get_results(), {pkg_1.pkg_name: result_d_1, f'{pkg_2.pkg_name}1': result_d_2,
                                                      f'{pkg_3.pkg_name}2': result_d_3})

        multi_result += pkg_4
        self.assertEqual(multi_result.get_results(), {pkg_1.pkg_name: result_d_1, f'{pkg_2.pkg_name}1': result_d_2,
                                                      f'{pkg_3.pkg_name}2': result_d_3, f'{pkg_4.pkg_name}': result_d_4})

    def test_concatenate_dict(self):
        result_d_1 = {'metric1': 33232, 'metric2': 1000}
        pkg_1 = DummyResultPackageExtendVariable(result_d_1)
        pkg_1.prepare_result_package([10] * 100, [11] * 100, {'dddd': 222})

        result_d_2 = {'metric3': 1, 'metric4': 2}
        pkg_2 = DummyResultPackageExtendVariable(result_d_2)
        pkg_2.prepare_result_package([10] * 100, [11] * 100, {'qqq': 445})

        pkg_dict_3 = {'metric1NEW': 33232, 'metric2NEW': 1000}

        multi_result = MultipleResultPackageWrapper()
        multi_result.prepare_result_package([pkg_1, pkg_2])

        concat_package = multi_result + pkg_dict_3
        self.assertEqual(concat_package.get_results(), {pkg_1.pkg_name: result_d_1, f'{pkg_2.pkg_name}1': result_d_2,
                                                        'PreCalculatedResult': pkg_dict_3})
        self.assertNotEqual(concat_package, multi_result)
        self.assertNotEqual(concat_package, pkg_dict_3)

        concat_package_2 = pkg_dict_3 + multi_result
        self.assertEqual(concat_package_2.get_results(), {pkg_1.pkg_name: result_d_1, f'{pkg_2.pkg_name}1': result_d_2,
                                                          'PreCalculatedResult': pkg_dict_3})

    def test_append_dict(self):
        result_d_1 = {'metric1': 33232, 'metric2': 1000}
        pkg_1 = DummyResultPackageExtendVariable(result_d_1)
        pkg_1.prepare_result_package([10] * 100, [11] * 100, {'dddd': 222})

        result_d_2 = {'metric3': 1, 'metric4': 2}
        pkg_2 = DummyResultPackageExtendVariable(result_d_2)
        pkg_2.prepare_result_package([10] * 100, [11] * 100, {'qqq': 445})

        pkg_dict_3 = {'metric1NEW': 33232, 'metric2NEW': 1000}

        multi_result = MultipleResultPackageWrapper()
        multi_result.prepare_result_package([pkg_1, pkg_2])

        multi_result += pkg_dict_3
        self.assertEqual(multi_result.get_results(), {pkg_1.pkg_name: result_d_1, f'{pkg_2.pkg_name}1': result_d_2,
                                                      'PreCalculatedResult': pkg_dict_3})

        multi_result += pkg_dict_3
        self.assertEqual(multi_result.get_results(), {pkg_1.pkg_name: result_d_1, f'{pkg_2.pkg_name}1': result_d_2,
                                                      'PreCalculatedResult': pkg_dict_3, 'PreCalculatedResult3': pkg_dict_3})

    def test_concatenate_fail(self):
        result_d_1 = {'metric1': 33232, 'metric2': 1000}
        pkg_1 = DummyResultPackageExtendVariable(result_d_1)

        result_d_2 = {'metric3': 1, 'metric4': 2}
        pkg_2 = DummyResultPackageExtendVariable(result_d_2)
        pkg_2.prepare_result_package([10] * 100, [11] * 100, {'qqq': 445})

        with self.assertRaises(ValueError):
            _ = pkg_1 + pkg_2

        with self.assertRaises(ValueError):
            _ = pkg_1 + {'metric3': 1, 'metric4': 2}

        pkg_1.prepare_result_package([10] * 100, [11] * 100, {'dddd': 222})

        with self.assertRaises(ValueError):
            _ = pkg_1 + 2233

        with self.assertRaises(ValueError):
            _ = pkg_1 + [32313]

    def test_append_fail(self):
        result_d_1 = {'metric1': 33232, 'metric2': 1000}
        pkg_1 = DummyResultPackageExtendVariable(result_d_1)

        result_d_2 = {'metric3': 1, 'metric4': 2}
        pkg_2 = DummyResultPackageExtendVariable(result_d_2)
        pkg_2.prepare_result_package([10] * 100, [11] * 100, {'qqq': 445})

        with self.assertRaises(ValueError):
            pkg_1 += pkg_2

        with self.assertRaises(ValueError):
            pkg_2 += pkg_1

        with self.assertRaises(ValueError):
            pkg_1 += {'metric3': 1, 'metric4': 2}

        pkg_1.prepare_result_package([10] * 100, [11] * 100, {'dddd': 222})

        with self.assertRaises(ValueError):
            pkg_1 += 2233

        with self.assertRaises(ValueError):
            pkg_1 += [32313]
