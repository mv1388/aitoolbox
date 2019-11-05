import unittest

from aitoolbox.experiment.result_package.abstract_result_packages import MultipleResultPackageWrapper
from aitoolbox.experiment.training_history import TrainingHistory
from tests.utils import DummyResultPackageExtendVariable


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
            pkg_1 + pkg_2

        with self.assertRaises(ValueError):
            pkg_1 + {'metric3': 1, 'metric4': 2}

        pkg_1.prepare_result_package([10] * 100, [11] * 100, {'dddd': 222}, self.build_train_hist())

        with self.assertRaises(ValueError):
            pkg_1 + 2233

        with self.assertRaises(ValueError):
            pkg_1 + [32313]

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
