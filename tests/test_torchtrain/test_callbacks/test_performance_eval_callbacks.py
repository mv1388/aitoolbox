import unittest
import os
import csv
import shutil
from tests.utils import *

from aitoolbox.torchtrain.callbacks.performance_eval import ModelPerformanceEvaluation, \
    ModelTrainHistoryFileWriter, MetricHistoryRename
from aitoolbox.torchtrain.train_loop import TrainLoop, TrainLoopCheckpoint
from aitoolbox.experiment.training_history import TrainingHistory
from aitoolbox.experiment.result_package.torch_metrics_packages import TorchMetricsPackage


THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class TestModelPerformanceEvaluationCallback(unittest.TestCase):
    def test_reg_set_experiment_dir_path_for_additional_results(self):
        dummy_optimizer = DummyOptimizer()
        dummy_train_loader = list(range(4))
        dummy_val_loader = list(range(3))
        dummy_test_loader = list(range(2))
        model = NetUnifiedBatchFeed()

        # test: if_available_output_to_project_dir=True
        result_pkg = DummyResultPackage()
        callback = ModelPerformanceEvaluation(result_pkg, {},
                                              on_each_epoch=True, on_train_data=False, on_val_data=True,
                                              if_available_output_to_project_dir=True)
        train_loop = TrainLoopCheckpoint(model,
                                         dummy_train_loader, dummy_val_loader, dummy_test_loader, dummy_optimizer, None,
                                         "project_name", "experiment_name", "local_model_result_folder_path", {},
                                         lazy_experiment_save=True)
        train_loop.callbacks_handler.register_callbacks([callback])

        self.assertEqual(result_pkg.experiment_path,
                         f'local_model_result_folder_path/project_name_experiment_name_{train_loop.experiment_timestamp}')

        # test: if_available_output_to_project_dir=False
        result_pkg = DummyResultPackage()
        callback = ModelPerformanceEvaluation(result_pkg, {},
                                              on_each_epoch=True, on_train_data=False, on_val_data=True,
                                              if_available_output_to_project_dir=False)
        train_loop = TrainLoopCheckpoint(model, dummy_train_loader, dummy_val_loader, dummy_test_loader,
                                         dummy_optimizer, None,
                                         "project_name", "experiment_name", "local_model_result_folder_path", {},
                                         lazy_experiment_save=True)
        train_loop.callbacks_handler.register_callbacks([callback])

        self.assertIsNone(result_pkg.experiment_path)

    def test_create_train_result_package(self):
        result_pkg = DummyResultPackage()
        callback_true = ModelPerformanceEvaluation(result_pkg, {},
                                                   on_each_epoch=True, on_train_data=True, on_val_data=True)
        self.assertTrue(hasattr(callback_true, 'train_result_package'))

        callback_true = ModelPerformanceEvaluation(result_pkg, {},
                                                   on_each_epoch=True, on_train_data=False, on_val_data=True)
        self.assertFalse(hasattr(callback_true, 'train_result_package'))

    def test_result_package_prepare(self):
        dummy_optimizer = DummyOptimizer()
        dummy_train_loader = list(range(4))
        dummy_val_loader = list(range(3))
        dummy_test_loader = list(range(2))
        model = NetUnifiedBatchFeed()

        result_pkg = DummyResultPackage()
        callback = ModelPerformanceEvaluation(result_pkg, {},
                                              on_each_epoch=True, on_train_data=False, on_val_data=True)
        train_loop = TrainLoop(model, dummy_train_loader, dummy_val_loader, dummy_test_loader, dummy_optimizer, None)
        train_loop.callbacks_handler.register_callbacks([callback])
        train_loop.insert_metric_result_into_history('dummy_loss', 10.)
        train_loop.insert_metric_result_into_history('dummy_acc', 55.)

        callback.evaluate_model_performance()

        self.assertEqual(callback.result_package.results_dict, {'dummy': 111})

        y_test = result_pkg.y_true
        y_pred = result_pkg.y_predicted
        metadata = result_pkg.additional_results['additional_results']

        r = []
        for i in range(1, len(dummy_val_loader) + 1):
            r += [i] * 64
        r2 = []
        for i in range(1, len(dummy_val_loader) + 1):
            r2 += [i + 100] * 64
        self.assertEqual(y_test.tolist(), r)
        self.assertEqual(y_pred.tolist(), r2)

        d = {'bla': []}
        for i in range(1, len(dummy_val_loader) + 1):
            d['bla'] += [i + 200] * 64
        self.assertEqual(metadata, d)

    def test_basic_store_evaluated_metrics_to_history(self):
        dummy_optimizer = DummyOptimizer()
        dummy_train_loader = list(range(4))
        dummy_val_loader = list(range(3))
        dummy_test_loader = list(range(2))
        model = NetUnifiedBatchFeed()

        result_pkg = DummyResultPackage()
        callback = ModelPerformanceEvaluation(result_pkg, {},
                                              on_each_epoch=True, on_train_data=False, on_val_data=True)
        train_loop = TrainLoop(model, dummy_train_loader, dummy_val_loader, dummy_test_loader, dummy_optimizer, None)
        train_loop.callbacks_handler.register_callbacks([callback])

        callback.evaluate_model_performance()

        self.assertEqual(train_loop.train_history.train_history,
                         {'loss': [], 'accumulated_loss': [], 'val_loss': [], 'val_dummy': [111]})

    def test_store_evaluated_metrics_to_history(self):
        dummy_optimizer = DummyOptimizer()
        dummy_train_loader = list(range(4))
        dummy_val_loader = list(range(3))
        dummy_test_loader = list(range(2))
        model = NetUnifiedBatchFeed()

        result_pkg = DummyResultPackage()
        callback = ModelPerformanceEvaluation(result_pkg, {},
                                              on_each_epoch=True, on_train_data=False, on_val_data=True)
        train_loop = TrainLoop(model, dummy_train_loader, dummy_val_loader, dummy_test_loader, dummy_optimizer, None)
        train_loop.callbacks_handler.register_callbacks([callback])
        train_loop.insert_metric_result_into_history('dummy_loss', 10.)
        train_loop.insert_metric_result_into_history('dummy_acc', 55.)

        callback.evaluate_model_performance()

        self.assertEqual(train_loop.train_history.train_history,
                         {'loss': [], 'accumulated_loss': [], 'val_loss': [], 'dummy_loss': [10.0], 'dummy_acc': [55.0],
                          'val_dummy': [111]})

    def test_store_evaluated_metrics_to_history_extended(self):
        dummy_optimizer = DummyOptimizer()
        dummy_train_loader = list(range(4))
        dummy_val_loader = list(range(3))
        dummy_test_loader = list(range(2))
        model = NetUnifiedBatchFeed()

        result_pkg = DummyResultPackageExtend()
        callback = ModelPerformanceEvaluation(result_pkg, {},
                                              on_each_epoch=True, on_train_data=False, on_val_data=True)
        train_loop = TrainLoop(model, dummy_train_loader, dummy_val_loader, dummy_test_loader, dummy_optimizer, None)
        train_loop.callbacks_handler.register_callbacks([callback])

        callback.evaluate_model_performance()

        self.assertEqual(train_loop.train_history.train_history,
                         {'loss': [], 'accumulated_loss': [], 'val_loss': [], 'val_dummy': [111],
                          'val_extended_dummy': [1323123.44]})

    def test_store_evaluated_metrics_to_history_multi_epoch_simulation(self):
        dummy_optimizer = DummyOptimizer()
        dummy_train_loader = list(range(4))
        dummy_val_loader = list(range(3))
        dummy_test_loader = list(range(2))
        model = NetUnifiedBatchFeed()

        result_pkg = DummyResultPackageExtend()
        callback = ModelPerformanceEvaluation(result_pkg, {},
                                              on_each_epoch=True, on_train_data=False, on_val_data=True)
        train_loop = TrainLoop(model, dummy_train_loader, dummy_val_loader, dummy_test_loader, dummy_optimizer, None)
        train_loop.callbacks_handler.register_callbacks([callback])

        # Epoch 1
        callback.evaluate_model_performance()
        self.assertEqual(train_loop.train_history.train_history,
                         {'loss': [], 'accumulated_loss': [], 'val_loss': [], 'val_dummy': [111],
                          'val_extended_dummy': [1323123.44]})

        # Epoch 2
        callback.evaluate_model_performance()
        self.assertEqual(train_loop.train_history.train_history,
                         {'loss': [], 'accumulated_loss': [], 'val_loss': [], 'val_dummy': [111.0, 123.0],
                          'val_extended_dummy': [1323123.44, 1323135.44]})

        # Epoch 3
        callback.evaluate_model_performance()
        self.assertEqual(train_loop.train_history.train_history,
                         {'loss': [], 'accumulated_loss': [], 'val_loss': [], 'val_dummy': [111.0, 123.0, 135.0],
                          'val_extended_dummy': [1323123.44, 1323135.44, 1323147.44]})

    def test_torch_metrics_result_package_performance_eval_epoch_end_reset_val_data(self):
        metric = DummyTorchMetrics(return_float=True)
        result_package = TorchMetricsPackage(metric)

        callback = ModelPerformanceEvaluation(
            result_package, {},
            on_each_epoch=False, on_train_data=False, on_val_data=True
        )

        for i in range(100):
            callback.on_epoch_end()
            self.assertEqual(metric.reset_ctr, i + 1)
            self.assertEqual(callback.result_package.metric.reset_ctr, i + 1)

    def test_torch_metrics_result_package_performance_eval_epoch_end_reset_train_val_data(self):
        metric = DummyTorchMetrics(return_float=True)
        result_package = TorchMetricsPackage(metric)

        callback = ModelPerformanceEvaluation(
            result_package, {},
            on_each_epoch=False, on_train_data=True, on_val_data=True
        )
        self.assertEqual(callback.result_package, result_package)
        self.assertNotEqual(result_package, callback.train_result_package)

        for i in range(100):
            callback.on_epoch_end()
            self.assertEqual(metric.reset_ctr, i + 1)
            self.assertEqual(callback.result_package.metric.reset_ctr, i + 1)
            self.assertEqual(callback.train_result_package.metric.reset_ctr, i + 1)

    def test_torch_metrics_result_package_performance_eval_tl_registration_move_to_device_val_data(self):
        metric = DummyTorchMetrics(return_float=True)
        result_package = TorchMetricsPackage(metric)

        callback = ModelPerformanceEvaluation(
            result_package, {},
            on_each_epoch=False, on_train_data=False, on_val_data=True,
            if_available_output_to_project_dir=False
        )
        callback.train_loop_obj = TrainLoop(NetUnifiedBatchFeed(), None, None, None, None, None)

        callback.on_train_loop_registration()
        self.assertEqual(metric.to_result, 'cpu_1')
        self.assertEqual(callback.result_package.metric.to_result, 'cpu_1')

    def test_torch_metrics_result_package_performance_eval_tl_registration_move_to_device_train_val_data(self):
        metric = DummyTorchMetrics(return_float=True)
        result_package = TorchMetricsPackage(metric)

        callback = ModelPerformanceEvaluation(
            result_package, {},
            on_each_epoch=False, on_train_data=True, on_val_data=True,
            if_available_output_to_project_dir=False
        )
        callback.train_loop_obj = TrainLoop(NetUnifiedBatchFeed(), None, None, None, None, None)

        callback.on_train_loop_registration()
        self.assertEqual(metric.to_result, 'cpu_1')
        self.assertEqual(callback.result_package.metric.to_result, 'cpu_1')
        self.assertEqual(callback.train_result_package.metric.to_result, 'cpu_1')


class TestModelTrainHistoryFileWriter(unittest.TestCase):
    def test_execute_callback(self):
        dummy_optimizer = DummyOptimizer()
        dummy_train_loader = list(range(4))
        dummy_val_loader = list(range(3))
        dummy_test_loader = list(range(2))
        model = NetUnifiedBatchFeed()

        callback = ModelTrainHistoryFileWriter(project_name='dummyProject', experiment_name='exper',
                                               local_model_result_folder_path=THIS_DIR, cloud_save_mode='local')
        train_loop = TrainLoop(model, dummy_train_loader, dummy_val_loader, dummy_test_loader, dummy_optimizer, None)
        train_loop.callbacks_handler.register_callbacks([callback])
        train_loop.train_history = TrainingHistory().wrap_pre_prepared_history({'loss': [123.4, 1223.4, 13323.4, 13323.4, 99999],
                                                                                'accumulated_loss': [], 'val_loss': [],
                                                                                'NEW_METRIC': [13323.4, 133323.4]})

        # Epoch 1
        train_loop.callbacks_handler.execute_epoch_end()
        f_path = os.path.join(THIS_DIR, 'dummyProject', f'exper_{train_loop.experiment_timestamp}',
                              'results', 'results.txt')

        with open(f_path, 'r') as f:
            f_content = [l.strip() for l in f.readlines()]

        self.assertEqual(f_content,
                         ['============================', 'Epoch: 0', '============================',
                          'loss:\t99999', 'NEW_METRIC:\t133323.4', '', ''])

        # Epoch 2
        train_loop.epoch += 1
        train_loop.insert_metric_result_into_history('COMPLETEY_NEW_METRIC', 3333.4)
        train_loop.callbacks_handler.execute_epoch_end()

        with open(f_path, 'r') as f:
            f_content = [l.strip() for l in f.readlines()]

        self.assertEqual(f_content,
                         ['============================', 'Epoch: 0', '============================',
                          'loss:\t99999', 'NEW_METRIC:\t133323.4', '', '',
                          '============================', 'Epoch: 1', '============================',
                          'loss:\t99999', 'NEW_METRIC:\t133323.4', 'COMPLETEY_NEW_METRIC:\t3333.4', '', ''])

        project_path = os.path.join(THIS_DIR, 'dummyProject')
        if os.path.exists(project_path):
            shutil.rmtree(project_path)

    def test_execute_callback_tsv_format(self):
        dummy_optimizer = DummyOptimizer()
        dummy_train_loader = list(range(4))
        dummy_val_loader = list(range(3))
        dummy_test_loader = list(range(2))
        model = NetUnifiedBatchFeed()

        callback = ModelTrainHistoryFileWriter(file_format='tsv', project_name='dummyProject', experiment_name='exper',
                                               local_model_result_folder_path=THIS_DIR, cloud_save_mode='local')
        train_loop = TrainLoop(model, dummy_train_loader, dummy_val_loader, dummy_test_loader, dummy_optimizer, None)
        train_loop.callbacks_handler.register_callbacks([callback])
        train_loop.train_history = TrainingHistory().wrap_pre_prepared_history(
            {'loss': [123.4, 1223.4, 13323.4, 13323.4, 99999],
             'accumulated_loss': [], 'val_loss': [],
             'NEW_METRIC': [13323.4, 133323.4]})

        # Epoch 1
        train_loop.callbacks_handler.execute_epoch_end()
        f_path = os.path.join(THIS_DIR, 'dummyProject', f'exper_{train_loop.experiment_timestamp}',
                              'results', 'results.tsv')

        with open(f_path, 'r') as f:
            tsv_reader = csv.reader(f, delimiter='\t')
            output_lines = [l for l in tsv_reader]

        self.assertEqual(output_lines, [['Epoch', 'loss', 'NEW_METRIC'],
                                        ['0', '99999', '133323.4']])

        # Epoch 2
        train_loop.epoch += 1
        train_loop.insert_metric_result_into_history('loss', 3333.4)
        train_loop.insert_metric_result_into_history('NEW_METRIC', 1111.2)
        train_loop.callbacks_handler.execute_epoch_end()

        with open(f_path, 'r') as f:
            tsv_reader = csv.reader(f, delimiter='\t')
            output_lines = [l for l in tsv_reader]

        self.assertEqual(output_lines, [['Epoch', 'loss', 'NEW_METRIC'],
                                        ['0', '99999', '133323.4'],
                                        ['1', '3333.4', '1111.2']])

        # Epoch 3
        train_loop.epoch += 1
        train_loop.insert_metric_result_into_history('loss', 555.4)
        train_loop.insert_metric_result_into_history('NEW_METRIC', 333.2)
        train_loop.insert_metric_result_into_history('COMPLETELY_NEW_METRIC', 123456.7)
        train_loop.callbacks_handler.execute_epoch_end()

        with open(f_path, 'r') as f:
            tsv_reader = csv.reader(f, delimiter='\t')
            output_lines = [l for l in tsv_reader]

        self.assertEqual(output_lines, [['Epoch', 'loss', 'NEW_METRIC'],
                                        ['0', '99999', '133323.4'],
                                        ['1', '3333.4', '1111.2'],
                                        ['NEW_METRICS_DETECTED'],
                                        ['Epoch', 'loss', 'NEW_METRIC', 'COMPLETELY_NEW_METRIC'],
                                        ['2', '555.4', '333.2', '123456.7']])

        project_path = os.path.join(THIS_DIR, 'dummyProject')
        if os.path.exists(project_path):
            shutil.rmtree(project_path)


class TestMetricHistoryRename(unittest.TestCase):
    def test_rename_metric(self):
        dummy_optimizer = DummyOptimizer()
        dummy_train_loader = list(range(4))
        dummy_val_loader = list(range(3))
        dummy_test_loader = list(range(2))
        model = NetUnifiedBatchFeed()

        result_pkg = DummyResultPackageExtend()
        callback = ModelPerformanceEvaluation(result_pkg, {},
                                              on_each_epoch=True, on_train_data=False, on_val_data=True)
        rename_callback = MetricHistoryRename(input_metric_path='val_dummy', new_metric_name='val_renamed_dummy',
                                              epoch_end=True, train_end=False)

        train_loop = TrainLoop(model, dummy_train_loader, dummy_val_loader, dummy_test_loader, dummy_optimizer, None)
        train_loop.callbacks_handler.register_callbacks([callback, rename_callback])

        callback.evaluate_model_performance()
        rename_callback.on_epoch_end()

        self.assertEqual(train_loop.train_history.train_history,
                         {'loss': [], 'accumulated_loss': [], 'val_loss': [], 'val_dummy': [111.0],
                          'val_extended_dummy': [1323123.44], 'val_renamed_dummy': [111.0]})

    def test_rename_metric_multi_epoch_simulation(self):
        dummy_optimizer = DummyOptimizer()
        dummy_train_loader = list(range(4))
        dummy_val_loader = list(range(3))
        dummy_test_loader = list(range(2))
        model = NetUnifiedBatchFeed()

        result_pkg = DummyResultPackageExtend()
        callback = ModelPerformanceEvaluation(result_pkg, {},
                                              on_each_epoch=True, on_train_data=False, on_val_data=True)
        rename_callback = MetricHistoryRename(input_metric_path='val_dummy', new_metric_name='val_renamed_dummy',
                                              epoch_end=True, train_end=False)

        train_loop = TrainLoop(model, dummy_train_loader, dummy_val_loader, dummy_test_loader, dummy_optimizer, None)
        train_loop.callbacks_handler.register_callbacks([callback, rename_callback])

        # Epoch 1
        callback.evaluate_model_performance()
        rename_callback.on_epoch_end()
        self.assertEqual(train_loop.train_history.train_history,
                         {'loss': [], 'accumulated_loss': [], 'val_loss': [], 'val_dummy': [111.0],
                          'val_extended_dummy': [1323123.44], 'val_renamed_dummy': [111.0]})
        train_loop.epoch += 1

        # Epoch 2
        callback.evaluate_model_performance()
        rename_callback.on_epoch_end()
        self.assertEqual(train_loop.train_history.train_history,
                         {'loss': [], 'accumulated_loss': [], 'val_loss': [], 'val_dummy': [111.0, 123.0],
                          'val_extended_dummy': [1323123.44, 1323135.44], 'val_renamed_dummy': [111.0, 123.0]})
        train_loop.epoch += 1

        # Epoch 3
        callback.evaluate_model_performance()
        rename_callback.on_epoch_end()
        self.assertEqual(train_loop.train_history.train_history,
                         {'loss': [], 'accumulated_loss': [], 'val_loss': [], 'val_dummy': [111.0, 123.0, 135.0],
                          'val_extended_dummy': [1323123.44, 1323135.44, 1323147.44], 'val_renamed_dummy': [111.0, 123.0, 135.0]})

    def test_fail_check_if_history_updated(self):
        dummy_optimizer = DummyOptimizer()
        dummy_train_loader = list(range(4))
        dummy_val_loader = list(range(3))
        dummy_test_loader = list(range(2))
        model = NetUnifiedBatchFeed()

        result_pkg = DummyResultPackageExtend()
        callback = ModelPerformanceEvaluation(result_pkg, {},
                                              on_each_epoch=True, on_train_data=False, on_val_data=True)
        rename_callback = MetricHistoryRename(input_metric_path='val_dummy', new_metric_name='val_renamed_dummy',
                                              epoch_end=True, train_end=False, strict_metric_extract=True)

        train_loop = TrainLoop(model, dummy_train_loader, dummy_val_loader, dummy_test_loader, dummy_optimizer, None)
        train_loop.callbacks_handler.register_callbacks([callback, rename_callback])

        callback.evaluate_model_performance()
        rename_callback.on_epoch_end()

        # Don't increment epoch count in the train loop to ensure that the history updated check fails
        callback.evaluate_model_performance()

        with self.assertRaises(ValueError):
            rename_callback.on_epoch_end()
