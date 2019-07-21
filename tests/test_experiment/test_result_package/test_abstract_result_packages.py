import unittest

from tests.utils import *
from AIToolbox.experiment.training_history import TrainingHistory
from AIToolbox.experiment.result_package.abstract_result_packages import AbstractResultPackage, \
    MultipleResultPackageWrapper, PreCalculatedResultPackage


class TestAbstractResultPackage(unittest.TestCase):
    def test_basic(self):
        history = {'val_loss': [2.2513437271118164, 2.1482439041137695, 2.0187528133392334, 1.7953970432281494,
                                1.5492324829101562, 1.715561032295227, 1.631982684135437, 1.3721977472305298,
                                1.039527416229248, 0.9796673059463501],
                   'val_acc': [0.25999999046325684, 0.36000001430511475, 0.5, 0.5400000214576721, 0.5400000214576721,
                               0.5799999833106995, 0.46000000834465027, 0.699999988079071, 0.7599999904632568,
                               0.7200000286102295],
                   'loss': [2.3088033199310303, 2.2141530513763428, 2.113713264465332, 1.912109375, 1.666761875152588,
                            1.460097312927246, 1.6031768321990967, 1.534214973449707, 1.1710081100463867,
                            0.8969314098358154],
                   'acc': [0.07999999821186066, 0.33000001311302185, 0.3100000023841858, 0.5299999713897705,
                           0.5799999833106995, 0.6200000047683716, 0.4300000071525574, 0.5099999904632568,
                           0.6700000166893005, 0.7599999904632568]}
        train_hist = TrainingHistory().wrap_pre_prepared_history(history)
        result_pkg = DummyResultPackageExtend()
        result_pkg.prepare_result_package([10] * 100, [11] * 100, {}, train_hist)

        self.assertEqual(result_pkg.results_dict, {'dummy': 111, 'extended_dummy': 1323123.44})
        self.assertEqual(result_pkg.get_results(), {'dummy': 111, 'extended_dummy': 1323123.44})
        self.assertEqual(result_pkg.get_hyperparameters(), {})
        self.assertEqual(result_pkg.get_training_history(), train_hist.get_train_history())

        self.assertEqual(str(result_pkg), 'dummy: 111.0\nextended_dummy: 1323123.44')
        self.assertEqual(len(result_pkg), 2)
        
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

    @staticmethod
    def build_train_hist():
        history = {'val_loss': [2.2513437271118164, 2.1482439041137695, 2.0187528133392334, 1.7953970432281494,
                                1.5492324829101562, 1.715561032295227, 1.631982684135437, 1.3721977472305298,
                                1.039527416229248, 0.9796673059463501],
                   'val_acc': [0.25999999046325684, 0.36000001430511475, 0.5, 0.5400000214576721, 0.5400000214576721,
                               0.5799999833106995, 0.46000000834465027, 0.699999988079071, 0.7599999904632568,
                               0.7200000286102295],
                   'loss': [2.3088033199310303, 2.2141530513763428, 2.113713264465332, 1.912109375, 1.666761875152588,
                            1.460097312927246, 1.6031768321990967, 1.534214973449707, 1.1710081100463867,
                            0.8969314098358154],
                   'acc': [0.07999999821186066, 0.33000001311302185, 0.3100000023841858, 0.5299999713897705,
                           0.5799999833106995, 0.6200000047683716, 0.4300000071525574, 0.5099999904632568,
                           0.6700000166893005, 0.7599999904632568]}
        train_hist = TrainingHistory().wrap_pre_prepared_history(history)
        return train_hist

    def test_combine_packages(self):
        train_hist_1 = self.build_train_hist()
        result_d_1 = {'metric1': 33232, 'metric2': 1000}
        pkg_1 = DummyResultPackageExtendVariable(result_d_1)
        pkg_1.prepare_result_package([10] * 100, [11] * 100, {'dddd': 222}, train_hist_1)

        result_d_2 = {'metric3': 1, 'metric4': 2}
        pkg_2 = DummyResultPackageExtendVariable(result_d_2)
        pkg_2.prepare_result_package([10] * 100, [11] * 100, {'qqq': 445}, self.build_train_hist())

        combo_pkg_1_2 = pkg_1 + pkg_2
        self.assertEqual(type(combo_pkg_1_2), MultipleResultPackageWrapper)
        self.assertEqual(len(combo_pkg_1_2.result_packages), 2)
        self.assertNotEqual(combo_pkg_1_2.result_packages[0], pkg_1)
        self.assertEqual(combo_pkg_1_2.result_packages[0].results_dict, pkg_1.results_dict)
        self.assertEqual(combo_pkg_1_2.result_packages[0].get_results(), pkg_1.get_results())
        self.assertEqual(combo_pkg_1_2.result_packages[0].get_hyperparameters(), pkg_1.get_hyperparameters())
        self.assertEqual(combo_pkg_1_2.result_packages[0].get_training_history(), pkg_1.get_training_history())

        self.assertNotEqual(combo_pkg_1_2.result_packages[1], pkg_2)
        self.assertEqual(combo_pkg_1_2.result_packages[1].results_dict, pkg_2.results_dict)
        self.assertEqual(combo_pkg_1_2.result_packages[1].get_results(), pkg_2.get_results())
        self.assertEqual(combo_pkg_1_2.result_packages[1].get_hyperparameters(), pkg_2.get_hyperparameters())
        self.assertEqual(combo_pkg_1_2.result_packages[1].get_training_history(), pkg_2.get_training_history())

        self.assertEqual(combo_pkg_1_2.results_dict, {pkg_1.pkg_name: result_d_1, f'{pkg_2.pkg_name}1': result_d_2})

        self.assertEqual(combo_pkg_1_2.training_history.get_train_history(), train_hist_1.get_train_history())
        self.assertEqual(combo_pkg_1_2.y_true['DummyPackage1'].tolist(), [10] * 100)
        self.assertEqual(combo_pkg_1_2.y_predicted['DummyPackage1'].tolist(), [11] * 100)
        self.assertEqual(combo_pkg_1_2.hyperparameters, {'dddd': 222})

    def test_combine_package_w_dict(self):
        train_hist_1 = self.build_train_hist()
        result_d_1 = {'metric1': 33232, 'metric2': 1000}
        pkg_1 = DummyResultPackageExtendVariable(result_d_1)
        pkg_1.prepare_result_package([10] * 100, [11] * 100, {'dddd': 222}, train_hist_1)

        pkg_2_dict = {'metric3': 11111}

        combo_pkg_1_2 = pkg_1 + pkg_2_dict

        self.assertEqual(type(combo_pkg_1_2), MultipleResultPackageWrapper)
        self.assertEqual(len(combo_pkg_1_2.result_packages), 2)
        self.assertNotEqual(combo_pkg_1_2.result_packages[0], pkg_1)
        self.assertEqual(combo_pkg_1_2.result_packages[0].results_dict, pkg_1.results_dict)
        self.assertEqual(combo_pkg_1_2.result_packages[0].get_results(), pkg_1.get_results())
        self.assertEqual(combo_pkg_1_2.result_packages[0].get_hyperparameters(), pkg_1.get_hyperparameters())
        self.assertEqual(combo_pkg_1_2.result_packages[0].get_training_history(), pkg_1.get_training_history())

        self.assertEqual(type(combo_pkg_1_2.result_packages[1]), PreCalculatedResultPackage)
        self.assertEqual(combo_pkg_1_2.result_packages[1].results_dict, pkg_2_dict)
        self.assertEqual(combo_pkg_1_2.result_packages[1].get_results(), pkg_2_dict)
        self.assertEqual(combo_pkg_1_2.result_packages[1].get_hyperparameters(), {})
        self.assertEqual(combo_pkg_1_2.result_packages[1].get_training_history(), TrainingHistory().get_train_history())

        self.assertEqual(combo_pkg_1_2.results_dict, {pkg_1.pkg_name: result_d_1,
                                                      combo_pkg_1_2.result_packages[1].pkg_name: pkg_2_dict})
        self.assertEqual(combo_pkg_1_2.results_dict, {pkg_1.pkg_name: result_d_1, 'PreCalculatedResult': pkg_2_dict})

    def test_combine_package_metric_name_clash(self):
        result_d_1 = {'metricSAME': 33232, 'metric2': 1000}
        pkg_1 = DummyResultPackageExtendVariable(result_d_1)
        pkg_1.prepare_result_package([10] * 100, [11] * 100, {'dddd': 222}, self.build_train_hist())

        result_d_2 = {'metricSAME': 33232, 'metric3': 1000}
        pkg_2 = DummyResultPackageExtendVariable(result_d_2)
        pkg_2.prepare_result_package([10] * 100, [11] * 100, {'qqq': 445}, self.build_train_hist())

        combo_pkg_1_2 = pkg_1 + pkg_2

        self.assertEqual(combo_pkg_1_2.results_dict, {pkg_1.pkg_name: result_d_1, f'{pkg_2.pkg_name}1': result_d_2})

    def test_combine_metric_dict_name_clash(self):
        result_d_1 = {'metricSAME': 33232, 'metric2': 1000}
        pkg_1 = DummyResultPackageExtendVariable(result_d_1)
        pkg_1.prepare_result_package([10] * 100, [11] * 100, {'dddd': 222}, self.build_train_hist())
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
        pkg_2.prepare_result_package([10] * 100, [11] * 100, {'qqq': 445}, self.build_train_hist())

        with self.assertRaises(ValueError):
            combo_pkg_1_2 = pkg_1 + pkg_2

    def test_append_packages(self):
        train_hist_1 = self.build_train_hist()
        result_d_1 = {'metric1': 33232, 'metric2': 1000}
        pkg_1 = DummyResultPackageExtendVariable(result_d_1)
        pkg_1.prepare_result_package([10] * 100, [11] * 100, {'dddd': 222}, train_hist_1)

        result_d_2 = {'metric3': 1, 'metric4': 2}
        pkg_2 = DummyResultPackageExtendVariable(result_d_2)
        pkg_2.prepare_result_package([100] * 100, [110] * 100, {'qqq': 445}, self.build_train_hist())

        pkg_1 += pkg_2

        self.assertEqual(type(pkg_1), DummyResultPackageExtendVariable)
        self.assertEqual(pkg_1.results_dict, {**result_d_1, **result_d_2})
        self.assertEqual(pkg_1.results_dict, {'metric1': 33232, 'metric2': 1000, 'metric3': 1, 'metric4': 2})

        self.assertEqual(pkg_1.y_true.tolist(), [10] * 100)
        self.assertEqual(pkg_1.y_predicted.tolist(), [11] * 100)
        self.assertEqual(pkg_1.training_history, train_hist_1)
        self.assertEqual(pkg_1.get_training_history(), train_hist_1.get_train_history())
        self.assertEqual(pkg_1.hyperparameters, {'dddd': 222})
        self.assertEqual(pkg_1.get_hyperparameters(), {'dddd': 222})

    def test_append_dict_packages(self):
        train_hist_1 = self.build_train_hist()
        result_d_1 = {'metric1': 33232, 'metric2': 1000}
        pkg_1 = DummyResultPackageExtendVariable(result_d_1)
        pkg_1.prepare_result_package([10] * 100, [11] * 100, {'dddd': 222}, train_hist_1)

        pkg_dict_2 = {'metric3': 1, 'metric4': 2}

        pkg_1 += pkg_dict_2

        self.assertEqual(type(pkg_1), DummyResultPackageExtendVariable)
        self.assertEqual(pkg_1.results_dict, {**result_d_1, **pkg_dict_2})
        self.assertEqual(pkg_1.results_dict, {'metric1': 33232, 'metric2': 1000, 'metric3': 1, 'metric4': 2})

        self.assertEqual(pkg_1.y_true.tolist(), [10] * 100)
        self.assertEqual(pkg_1.y_predicted.tolist(), [11] * 100)
        self.assertEqual(pkg_1.training_history, train_hist_1)
        self.assertEqual(pkg_1.get_training_history(), train_hist_1.get_train_history())
        self.assertEqual(pkg_1.hyperparameters, {'dddd': 222})
        self.assertEqual(pkg_1.get_hyperparameters(), {'dddd': 222})

    def test_fail_append_packages_name_clash_val_fail(self):
        train_hist_1 = self.build_train_hist()
        result_d_1 = {'metricSAME': 33232, 'metric2': 1000}
        pkg_1 = DummyResultPackageExtendVariable(result_d_1)
        pkg_1.prepare_result_package([10] * 100, [11] * 100, {'dddd': 222}, train_hist_1)

        result_d_2 = {'metricSAME': 1, 'metric4': 2}
        pkg_2 = DummyResultPackageExtendVariable(result_d_2)
        pkg_2.prepare_result_package([100] * 100, [110] * 100, {'qqq': 445}, self.build_train_hist())

        with self.assertRaises(ValueError):
            pkg_1 += pkg_2

        with self.assertRaises(ValueError):
            pkg_1 += [23323]

        with self.assertRaises(ValueError):
            pkg_1 += 33121

    def test_fail_append_dict_packages_name_clash(self):
        train_hist_1 = self.build_train_hist()
        result_d_1 = {'metricSAME': 33232, 'metric2': 1000}
        pkg_1 = DummyResultPackageExtendVariable(result_d_1)
        pkg_1.prepare_result_package([10] * 100, [11] * 100, {'dddd': 222}, train_hist_1)

        pkg_dict_2 = {'metricSAME': 1, 'metric4': 2}

        with self.assertRaises(ValueError):
            pkg_1 += pkg_dict_2

    def test_package_contains(self):
        train_hist_1 = self.build_train_hist()
        result_d_1 = {'metric1': 33232, 'metric2': 1000}
        pkg_1 = DummyResultPackageExtendVariable(result_d_1)
        with self.assertRaises(ValueError):
            res = 'metric1' in pkg_1

        pkg_1.prepare_result_package([10] * 100, [11] * 100, {'dddd': 222}, train_hist_1)
        self.assertTrue('metric1' in pkg_1)
        self.assertTrue('metric2' in pkg_1)
        self.assertFalse('metricMissing' in pkg_1)

    def test_package_get_item(self):
        train_hist_1 = self.build_train_hist()
        result_d_1 = {'metric1': 33232, 'metric2': 1000}
        pkg_1 = DummyResultPackageExtendVariable(result_d_1)
        with self.assertRaises(ValueError):
            res = pkg_1['metric1']

        pkg_1.prepare_result_package([10] * 100, [11] * 100, {'dddd': 222}, train_hist_1)
        self.assertEqual(pkg_1['metric1'], result_d_1['metric1'])
        self.assertEqual(pkg_1['metric2'], result_d_1['metric2'])
        with self.assertRaises(KeyError):
            res = pkg_1['metricMissing']


class TestMultipleResultPackageWrapper(unittest.TestCase):
    @staticmethod
    def build_train_hist():
        history = {'val_loss': [2.2513437271118164, 2.1482439041137695, 2.0187528133392334, 1.7953970432281494,
                                1.5492324829101562, 1.715561032295227, 1.631982684135437, 1.3721977472305298,
                                1.039527416229248, 0.9796673059463501],
                   'val_acc': [0.25999999046325684, 0.36000001430511475, 0.5, 0.5400000214576721, 0.5400000214576721,
                               0.5799999833106995, 0.46000000834465027, 0.699999988079071, 0.7599999904632568,
                               0.7200000286102295],
                   'loss': [2.3088033199310303, 2.2141530513763428, 2.113713264465332, 1.912109375, 1.666761875152588,
                            1.460097312927246, 1.6031768321990967, 1.534214973449707, 1.1710081100463867,
                            0.8969314098358154],
                   'acc': [0.07999999821186066, 0.33000001311302185, 0.3100000023841858, 0.5299999713897705,
                           0.5799999833106995, 0.6200000047683716, 0.4300000071525574, 0.5099999904632568,
                           0.6700000166893005, 0.7599999904632568]}
        train_hist = TrainingHistory().wrap_pre_prepared_history(history)
        return train_hist

    def test_basic(self):
        train_hist_1 = self.build_train_hist()
        result_d_1 = {'metric1': 33232, 'metric2': 1000}
        pkg_1 = DummyResultPackageExtendVariable(result_d_1)
        pkg_1.prepare_result_package([10] * 100, [11] * 100, {'dddd': 222}, train_hist_1)

        result_d_2 = {'metric3': 1, 'metric4': 2}
        pkg_2 = DummyResultPackageExtendVariable(result_d_2)
        pkg_2.prepare_result_package([10] * 100, [11] * 100, {'qqq': 445}, self.build_train_hist())

        multi_result = MultipleResultPackageWrapper()
        multi_result.prepare_result_package([pkg_1, pkg_2])

        self.assertEqual(str(multi_result),
                         f"--> {pkg_1.pkg_name}:\nmetric1: {result_d_1['metric1']}\nmetric2: {result_d_1['metric2']}\n"
                         f"--> {pkg_2.pkg_name}:\nmetric3: {result_d_2['metric3']}\nmetric4: {result_d_2['metric4']}")

        self.assertEqual(multi_result.get_results(), {pkg_1.pkg_name: result_d_1, f'{pkg_2.pkg_name}1': result_d_2})

    def test_concatenate_package(self):
        train_hist_1 = self.build_train_hist()
        result_d_1 = {'metric1': 33232, 'metric2': 1000}
        pkg_1 = DummyResultPackageExtendVariable(result_d_1)
        pkg_1.prepare_result_package([10] * 100, [11] * 100, {'dddd': 222}, train_hist_1)

        result_d_2 = {'metric3': 1, 'metric4': 2}
        pkg_2 = DummyResultPackageExtendVariable(result_d_2)
        pkg_2.prepare_result_package([10] * 100, [11] * 100, {'qqq': 445}, self.build_train_hist())

        train_hist_3 = self.build_train_hist()
        result_d_3 = {'metric1NEW': 33232, 'metric2NEW': 1000}
        pkg_3 = DummyResultPackageExtendVariable(result_d_3)
        pkg_3.prepare_result_package([10] * 100, [11] * 100, {'dddd': 222}, train_hist_3)

        train_hist_4 = self.build_train_hist()
        result_d_4 = {'metric1NEW': 33232, 'metric2NEW': 1000}
        pkg_4 = DummyResultPackageExtendVariable(result_d_4)
        pkg_4.pkg_name = 'NEWPackageName'
        pkg_4.prepare_result_package([10] * 100, [11] * 100, {'dddd': 222}, train_hist_4)

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
        pkg_1.prepare_result_package([10] * 100, [11] * 100, {'dddd': 222}, self.build_train_hist())

        result_d_2 = {'metric3': 1, 'metric4': 2}
        pkg_2 = DummyResultPackageExtendVariable(result_d_2)
        pkg_2.prepare_result_package([10] * 100, [11] * 100, {'qqq': 445}, self.build_train_hist())

        train_hist_3 = self.build_train_hist()
        result_d_3 = {'metric1NEW': 33232, 'metric2NEW': 1000}
        pkg_3 = DummyResultPackageExtendVariable(result_d_3)
        pkg_3.prepare_result_package([10] * 100, [11] * 100, {'dddd': 222}, train_hist_3)

        train_hist_4 = self.build_train_hist()
        result_d_4 = {'metric1NEW': 33232, 'metric2NEW': 1000}
        pkg_4 = DummyResultPackageExtendVariable(result_d_4)
        pkg_4.pkg_name = 'NEWPackageName'
        pkg_4.prepare_result_package([10] * 100, [11] * 100, {'dddd': 222}, train_hist_4)

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
        pkg_1.prepare_result_package([10] * 100, [11] * 100, {'dddd': 222}, self.build_train_hist())

        result_d_2 = {'metric3': 1, 'metric4': 2}
        pkg_2 = DummyResultPackageExtendVariable(result_d_2)
        pkg_2.prepare_result_package([10] * 100, [11] * 100, {'qqq': 445}, self.build_train_hist())

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
        pkg_1.prepare_result_package([10] * 100, [11] * 100, {'dddd': 222}, self.build_train_hist())

        result_d_2 = {'metric3': 1, 'metric4': 2}
        pkg_2 = DummyResultPackageExtendVariable(result_d_2)
        pkg_2.prepare_result_package([10] * 100, [11] * 100, {'qqq': 445}, self.build_train_hist())

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
        pkg_2.prepare_result_package([10] * 100, [11] * 100, {'qqq': 445}, self.build_train_hist())

        with self.assertRaises(ValueError):
            res = pkg_1 + pkg_2

        with self.assertRaises(ValueError):
            res = pkg_1 + {'metric3': 1, 'metric4': 2}

        pkg_1.prepare_result_package([10] * 100, [11] * 100, {'dddd': 222}, self.build_train_hist())

        with self.assertRaises(ValueError):
            res = pkg_1 + 2233

        with self.assertRaises(ValueError):
            res = pkg_1 + [32313]

    def test_append_fail(self):
        result_d_1 = {'metric1': 33232, 'metric2': 1000}
        pkg_1 = DummyResultPackageExtendVariable(result_d_1)

        result_d_2 = {'metric3': 1, 'metric4': 2}
        pkg_2 = DummyResultPackageExtendVariable(result_d_2)
        pkg_2.prepare_result_package([10] * 100, [11] * 100, {'qqq': 445}, self.build_train_hist())

        with self.assertRaises(ValueError):
            pkg_1 += pkg_2

        with self.assertRaises(ValueError):
            pkg_2 += pkg_1

        with self.assertRaises(ValueError):
            pkg_1 += {'metric3': 1, 'metric4': 2}

        pkg_1.prepare_result_package([10] * 100, [11] * 100, {'dddd': 222}, self.build_train_hist())

        with self.assertRaises(ValueError):
            pkg_1 += 2233

        with self.assertRaises(ValueError):
            pkg_1 += [32313]
