import unittest

from tests.utils import *
from aitoolbox.experiment.result_package.abstract_result_packages import MultipleResultPackageWrapper, PreCalculatedResultPackage


class TestAbstractResultPackage(unittest.TestCase):
    def test_basic(self):
        result_pkg = DummyResultPackageExtend()
        result_pkg.prepare_result_package([10] * 100, [11] * 100, {})

        self.assertEqual(result_pkg.results_dict, {'dummy': 111, 'extended_dummy': 1323123.44})
        self.assertEqual(result_pkg.get_results(), {'dummy': 111, 'extended_dummy': 1323123.44})
        self.assertEqual(result_pkg.get_hyperparameters(), {})

        self.assertEqual(str(result_pkg), 'dummy: 111.0\nextended_dummy: 1323123.44')
        self.assertEqual(len(result_pkg), 2)

    def test_get_results_error_trigger(self):
        result_pkg = DummyResultPackage()
        self.assertIsNone(result_pkg.get_results())

        result_pkg = DummyResultPackage(strict_content_check=True)
        with self.assertRaises(ValueError):
            result_pkg.get_results()

    def test_qa_check_hyperparameters_dict_error_trigger(self):
        result_pkg = DummyResultPackage(strict_content_check=True)

        with self.assertRaises(ValueError):
            result_pkg.qa_check_hyperparameters_dict()

    def test_warn_about_result_data_problem(self):
        result_pkg = DummyResultPackage(strict_content_check=False)
        result_pkg.warn_about_result_data_problem('aaaaa')

        result_pkg = DummyResultPackage(strict_content_check=True)
        with self.assertRaises(ValueError):
            result_pkg.warn_about_result_data_problem('aaaaa')

    def test_get_additional_results_dump_paths(self):
        paths_1 = [['filename', 'file/path/filename']]
        result_pkg_1 = DummyResultPackageExtendV2(paths_1)
        self.assertEqual(result_pkg_1.get_additional_results_dump_paths(), paths_1)
        self.assertEqual(result_pkg_1.additional_results_dump_paths, paths_1)

        paths_2 = [['filename', 'file/path/filename'], ['fafafdfa', 'ewqewq/eqwq/rrrrrr/fafafdfa']]
        result_pkg_2 = DummyResultPackageExtendV2(paths_2)
        self.assertEqual(result_pkg_2.get_additional_results_dump_paths(), paths_2)
        self.assertEqual(result_pkg_2.additional_results_dump_paths, paths_2)

    def test_format_enforcement_get_additional_results_dump_paths(self):
        # Test wrong format catching
        with self.assertRaises(ValueError):
            result_pkg_2 = DummyResultPackageExtendV2('file/path/string/not/list')
            result_pkg_2.get_additional_results_dump_paths()

        with self.assertRaises(ValueError):
            result_pkg_2 = DummyResultPackageExtendV2(['file/path/string/not/list'])
            result_pkg_2.get_additional_results_dump_paths()

        with self.assertRaises(ValueError):
            result_pkg_2 = DummyResultPackageExtendV2(['file/path/string/not/list', 'another/string/'])
            result_pkg_2.get_additional_results_dump_paths()

        with self.assertRaises(ValueError):
            result_pkg_2 = DummyResultPackageExtendV2([['file/path/string/not/list/not2/elements/insublist']])
            result_pkg_2.get_additional_results_dump_paths()

        with self.assertRaises(ValueError):
            result_pkg_2 = DummyResultPackageExtendV2([['file/path/string/not/list/not2/'], ['still/not/2elements']])
            result_pkg_2.get_additional_results_dump_paths()

        with self.assertRaises(ValueError):
            result_pkg_2 = DummyResultPackageExtendV2([['file/path/string/not/list/not2/', 2332]])
            result_pkg_2.get_additional_results_dump_paths()

        with self.assertRaises(ValueError):
            result_pkg_2 = DummyResultPackageExtendV2([[2332, 'file/path/string/not/list/not2/']])
            result_pkg_2.get_additional_results_dump_paths()

        with self.assertRaises(ValueError):
            result_pkg_2 = DummyResultPackageExtendV2([[{'ddasd': 223}, 2332]])
            result_pkg_2.get_additional_results_dump_paths()

        with self.assertRaises(ValueError):
            result_pkg_2 = DummyResultPackageExtendV2([['file/path/not2/', 'dad'], ['weaeew']])
            result_pkg_2.get_additional_results_dump_paths()

        with self.assertRaises(ValueError):
            result_pkg_2 = DummyResultPackageExtendV2([['file/path/not2/', 'dad'], ['weaeew', 2]])
            result_pkg_2.get_additional_results_dump_paths()

        with self.assertRaises(ValueError):
            result_pkg_2 = DummyResultPackageExtendV2([[['file/path/not2/', 'dad'], ['weaeew', 'wadas']]])
            result_pkg_2.get_additional_results_dump_paths()

        with self.assertRaises(ValueError):
            result_pkg_2 = DummyResultPackageExtendV2([['file/path/not2/', 'dad'], 2323244])
            result_pkg_2.get_additional_results_dump_paths()

        with self.assertRaises(ValueError):
            result_pkg_2 = DummyResultPackageExtendV2([['file/path/not2/', 'dad'], 'dpasppsa'])
            result_pkg_2.get_additional_results_dump_paths()

    def test_combine_packages(self):
        result_d_1 = {'metric1': 33232, 'metric2': 1000}
        pkg_1 = DummyResultPackageExtendVariable(result_d_1)
        pkg_1.prepare_result_package([10] * 100, [11] * 100, {'dddd': 222})

        result_d_2 = {'metric3': 1, 'metric4': 2}
        pkg_2 = DummyResultPackageExtendVariable(result_d_2)
        pkg_2.prepare_result_package([10] * 100, [11] * 100, {'qqq': 445})

        combo_pkg_1_2 = pkg_1 + pkg_2
        self.assertEqual(type(combo_pkg_1_2), MultipleResultPackageWrapper)
        self.assertEqual(len(combo_pkg_1_2.result_packages), 2)
        self.assertNotEqual(combo_pkg_1_2.result_packages[0], pkg_1)
        self.assertEqual(combo_pkg_1_2.result_packages[0].results_dict, pkg_1.results_dict)
        self.assertEqual(combo_pkg_1_2.result_packages[0].get_results(), pkg_1.get_results())
        self.assertEqual(combo_pkg_1_2.result_packages[0].get_hyperparameters(), pkg_1.get_hyperparameters())

        self.assertNotEqual(combo_pkg_1_2.result_packages[1], pkg_2)
        self.assertEqual(combo_pkg_1_2.result_packages[1].results_dict, pkg_2.results_dict)
        self.assertEqual(combo_pkg_1_2.result_packages[1].get_results(), pkg_2.get_results())
        self.assertEqual(combo_pkg_1_2.result_packages[1].get_hyperparameters(), pkg_2.get_hyperparameters())

        self.assertEqual(combo_pkg_1_2.results_dict, {pkg_1.pkg_name: result_d_1, f'{pkg_2.pkg_name}1': result_d_2})

        self.assertEqual(combo_pkg_1_2.y_true['DummyPackage1'].tolist(), [10] * 100)
        self.assertEqual(combo_pkg_1_2.y_predicted['DummyPackage1'].tolist(), [11] * 100)
        self.assertEqual(combo_pkg_1_2.hyperparameters, {'dddd': 222})

    def test_combine_package_w_dict(self):
        result_d_1 = {'metric1': 33232, 'metric2': 1000}
        pkg_1 = DummyResultPackageExtendVariable(result_d_1)
        pkg_1.prepare_result_package([10] * 100, [11] * 100, {'dddd': 222}, )

        pkg_2_dict = {'metric3': 11111}

        combo_pkg_1_2 = pkg_1 + pkg_2_dict

        self.assertEqual(type(combo_pkg_1_2), MultipleResultPackageWrapper)
        self.assertEqual(len(combo_pkg_1_2.result_packages), 2)
        self.assertNotEqual(combo_pkg_1_2.result_packages[0], pkg_1)
        self.assertEqual(combo_pkg_1_2.result_packages[0].results_dict, pkg_1.results_dict)
        self.assertEqual(combo_pkg_1_2.result_packages[0].get_results(), pkg_1.get_results())
        self.assertEqual(combo_pkg_1_2.result_packages[0].get_hyperparameters(), pkg_1.get_hyperparameters())

        self.assertEqual(type(combo_pkg_1_2.result_packages[1]), PreCalculatedResultPackage)
        self.assertEqual(combo_pkg_1_2.result_packages[1].results_dict, pkg_2_dict)
        self.assertEqual(combo_pkg_1_2.result_packages[1].get_results(), pkg_2_dict)
        self.assertEqual(combo_pkg_1_2.result_packages[1].get_hyperparameters(), {})

        self.assertEqual(combo_pkg_1_2.results_dict, {pkg_1.pkg_name: result_d_1,
                                                      combo_pkg_1_2.result_packages[1].pkg_name: pkg_2_dict})
        self.assertEqual(combo_pkg_1_2.results_dict, {pkg_1.pkg_name: result_d_1, 'PreCalculatedResult': pkg_2_dict})

    def test_combine_package_metric_name_clash(self):
        result_d_1 = {'metricSAME': 33232, 'metric2': 1000}
        pkg_1 = DummyResultPackageExtendVariable(result_d_1)
        pkg_1.prepare_result_package([10] * 100, [11] * 100, {'dddd': 222})

        result_d_2 = {'metricSAME': 33232, 'metric3': 1000}
        pkg_2 = DummyResultPackageExtendVariable(result_d_2)
        pkg_2.prepare_result_package([10] * 100, [11] * 100, {'qqq': 445})

        combo_pkg_1_2 = pkg_1 + pkg_2

        self.assertEqual(combo_pkg_1_2.results_dict, {pkg_1.pkg_name: result_d_1, f'{pkg_2.pkg_name}1': result_d_2})

    def test_combine_metric_dict_name_clash(self):
        result_d_1 = {'metricSAME': 33232, 'metric2': 1000}
        pkg_1 = DummyResultPackageExtendVariable(result_d_1)
        pkg_1.prepare_result_package([10] * 100, [11] * 100, {'dddd': 222})
        pkg_2_dict = {'metricSAME': 33232, 'metric3': 1000}

        combo_pkg_1_2 = pkg_1 + pkg_2_dict
        self.assertEqual(combo_pkg_1_2.results_dict, {pkg_1.pkg_name: result_d_1,
                                                      combo_pkg_1_2.result_packages[1].pkg_name: pkg_2_dict})
        self.assertEqual(combo_pkg_1_2.results_dict, {pkg_1.pkg_name: result_d_1, 'PreCalculatedResult': pkg_2_dict})

    def test_fail_dict_not_defined(self):
        result_d_1 = {'metric1': 33232, 'metric2': 1000}
        pkg_1 = DummyResultPackageExtendVariable(result_d_1)

        with self.assertRaises(ValueError):
            pkg_2_dict = {'metric3': 11111}
            combo_pkg_1_2 = pkg_1 + pkg_2_dict

        with self.assertRaises(ValueError):
            pkg_2_dict = {'metric3': 11111}
            combo_pkg_1_2 = pkg_2_dict + pkg_1

    def test_fail_dict_not_defined_pkg(self):
        result_d_1 = {'metricSAME': 33232, 'metric2': 1000}
        pkg_1 = DummyResultPackageExtendVariable(result_d_1)

        result_d_2 = {'metricSAME': 33232, 'metric3': 1000}
        pkg_2 = DummyResultPackageExtendVariable(result_d_2)
        pkg_2.prepare_result_package([10] * 100, [11] * 100, {'qqq': 445})

        with self.assertRaises(ValueError):
            combo_pkg_1_2 = pkg_1 + pkg_2

    def test_append_packages(self):
        result_d_1 = {'metric1': 33232, 'metric2': 1000}
        pkg_1 = DummyResultPackageExtendVariable(result_d_1)
        pkg_1.prepare_result_package([10] * 100, [11] * 100, {'dddd': 222})

        result_d_2 = {'metric3': 1, 'metric4': 2}
        pkg_2 = DummyResultPackageExtendVariable(result_d_2)
        pkg_2.prepare_result_package([100] * 100, [110] * 100, {'qqq': 445})

        pkg_1 += pkg_2

        self.assertEqual(type(pkg_1), DummyResultPackageExtendVariable)
        self.assertEqual(pkg_1.results_dict, {**result_d_1, **result_d_2})
        self.assertEqual(pkg_1.results_dict, {'metric1': 33232, 'metric2': 1000, 'metric3': 1, 'metric4': 2})

        self.assertEqual(pkg_1.y_true.tolist(), [10] * 100)
        self.assertEqual(pkg_1.y_predicted.tolist(), [11] * 100)
        self.assertEqual(pkg_1.hyperparameters, {'dddd': 222})
        self.assertEqual(pkg_1.get_hyperparameters(), {'dddd': 222})

    def test_append_dict_packages(self):
        result_d_1 = {'metric1': 33232, 'metric2': 1000}
        pkg_1 = DummyResultPackageExtendVariable(result_d_1)
        pkg_1.prepare_result_package([10] * 100, [11] * 100, {'dddd': 222})

        pkg_dict_2 = {'metric3': 1, 'metric4': 2}

        pkg_1 += pkg_dict_2

        self.assertEqual(type(pkg_1), DummyResultPackageExtendVariable)
        self.assertEqual(pkg_1.results_dict, {**result_d_1, **pkg_dict_2})
        self.assertEqual(pkg_1.results_dict, {'metric1': 33232, 'metric2': 1000, 'metric3': 1, 'metric4': 2})

        self.assertEqual(pkg_1.y_true.tolist(), [10] * 100)
        self.assertEqual(pkg_1.y_predicted.tolist(), [11] * 100)
        self.assertEqual(pkg_1.hyperparameters, {'dddd': 222})
        self.assertEqual(pkg_1.get_hyperparameters(), {'dddd': 222})

    def test_fail_append_packages_name_clash_val_fail(self):
        result_d_1 = {'metricSAME': 33232, 'metric2': 1000}
        pkg_1 = DummyResultPackageExtendVariable(result_d_1)
        pkg_1.prepare_result_package([10] * 100, [11] * 100, {'dddd': 222})

        result_d_2 = {'metricSAME': 1, 'metric4': 2}
        pkg_2 = DummyResultPackageExtendVariable(result_d_2)
        pkg_2.prepare_result_package([100] * 100, [110] * 100, {'qqq': 445})

        with self.assertRaises(ValueError):
            pkg_1 += pkg_2

        with self.assertRaises(ValueError):
            pkg_1 += [23323]

        with self.assertRaises(ValueError):
            pkg_1 += 33121

    def test_fail_append_dict_packages_name_clash(self):
        result_d_1 = {'metricSAME': 33232, 'metric2': 1000}
        pkg_1 = DummyResultPackageExtendVariable(result_d_1)
        pkg_1.prepare_result_package([10] * 100, [11] * 100, {'dddd': 222})

        pkg_dict_2 = {'metricSAME': 1, 'metric4': 2}

        with self.assertRaises(ValueError):
            pkg_1 += pkg_dict_2

    def test_package_contains(self):
        result_d_1 = {'metric1': 33232, 'metric2': 1000}
        pkg_1 = DummyResultPackageExtendVariable(result_d_1)
        with self.assertRaises(ValueError):
            res = 'metric1' in pkg_1

        pkg_1.prepare_result_package([10] * 100, [11] * 100, {'dddd': 222})
        self.assertTrue('metric1' in pkg_1)
        self.assertTrue('metric2' in pkg_1)
        self.assertFalse('metricMissing' in pkg_1)

    def test_package_get_item(self):
        result_d_1 = {'metric1': 33232, 'metric2': 1000}
        pkg_1 = DummyResultPackageExtendVariable(result_d_1)
        with self.assertRaises(ValueError):
            res = pkg_1['metric1']

        pkg_1.prepare_result_package([10] * 100, [11] * 100, {'dddd': 222})
        self.assertEqual(pkg_1['metric1'], result_d_1['metric1'])
        self.assertEqual(pkg_1['metric2'], result_d_1['metric2'])
        with self.assertRaises(KeyError):
            res = pkg_1['metricMissing']
