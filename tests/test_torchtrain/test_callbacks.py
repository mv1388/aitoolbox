import unittest

from tests.utils import *

from AIToolbox.torchtrain.callbacks.callbacks import AbstractCallback, ModelCheckpoint, ModelTrainEndSave, \
    EarlyStopping
from AIToolbox.torchtrain.train_loop import TrainLoop
from AIToolbox.cloud.AWS.model_save import PyTorchS3ModelSaver
from AIToolbox.experiment_save.local_save.local_model_save import PyTorchLocalModelSaver
from AIToolbox.experiment_save.experiment_saver import FullPyTorchExperimentS3Saver
from AIToolbox.experiment_save.local_experiment_saver import FullPyTorchExperimentLocalSaver


class TestAbstractCallback(unittest.TestCase):
    def test_abstract_callback_has_hook_methods(self):
        callback = AbstractCallback('test_callback')

        self.assertTrue(function_exists(callback, 'on_train_loop_registration'))
        self.assertTrue(function_exists(callback, 'on_epoch_begin'))
        self.assertTrue(function_exists(callback, 'on_epoch_end'))
        self.assertTrue(function_exists(callback, 'on_train_begin'))
        self.assertTrue(function_exists(callback, 'on_train_end'))
        self.assertTrue(function_exists(callback, 'on_batch_begin'))
        self.assertTrue(function_exists(callback, 'on_batch_end'))

    def test_on_train_loop_registration_hook(self):
        train_loop = TrainLoop(NetUnifiedBatchFeed(), None, None, None, None, None)
        callback = CallbackTracker()
        callback.register_train_loop_object(train_loop)

        self.assertIsInstance(callback, AbstractCallback)
        self.assertEqual(callback.callback_calls, ['on_train_loop_registration'])


class TestModelCheckpointCallback(unittest.TestCase):
    def test_init(self):
        callback_true = ModelCheckpoint('project_name', 'experiment_name', 'local_model_result_folder_path', args={},
                                        cloud_save_mode='s3')
        self.assertEqual(type(callback_true.model_checkpointer), PyTorchS3ModelSaver)

        # callback_true = ModelCheckpoint('project_name', 'experiment_name', 'local_model_result_folder_path',
        #                                         cloud_save_mode='gcs')
        # self.assertEqual(type(callback_true.model_checkpointer), PyTorchGoogleStorageModelSaver)

        callback_false = ModelCheckpoint('project_name', 'experiment_name', 'local_model_result_folder_path', args={},
                                         cloud_save_mode=None)
        self.assertEqual(type(callback_false.model_checkpointer), PyTorchLocalModelSaver)

    def test_optimizer_missing_state_dict_exception(self):
        callback = ModelCheckpoint('project_name', 'experiment_name', 'local_model_result_folder_path', args={},
                                   cloud_save_mode=None)
        train_loop = TrainLoop(NetUnifiedBatchFeed(), None, None, None, MiniDummyOptimizer(), None)

        with self.assertRaises(AttributeError):
            train_loop.callbacks_handler.register_callbacks([callback])


class TestModelTrainEndSaveCallback(unittest.TestCase):
    def test_init(self):
        callback_true = ModelTrainEndSave('project_name', 'experiment_name', 'local_model_result_folder_path',
                                          {}, DummyResultPackage(), cloud_save_mode='s3')
        self.assertEqual(type(callback_true.results_saver), FullPyTorchExperimentS3Saver)

        # callback_true = ModelTrainEndSave('project_name', 'experiment_name', 'local_model_result_folder_path',
        #                                           {}, DummyResultPackage(), cloud_save_mode='gcs')
        # self.assertEqual(type(callback_true.results_saver), FullPyTorchExperimentGoogleStorageSaver)

        callback_false = ModelTrainEndSave('project_name', 'experiment_name', 'local_model_result_folder_path',
                                           {}, DummyResultPackage(), cloud_save_mode=None)
        self.assertEqual(type(callback_false.results_saver), FullPyTorchExperimentLocalSaver)

    def test_train_loop_reg_set_experiment_dir_path_for_additional_results(self):
        result_pkg = DummyResultPackage()
        self.assertIsNone(result_pkg.experiment_path)

        callback = ModelTrainEndSave('project_name', 'experiment_name', 'local_model_result_folder_path',
                                     {}, result_pkg)
        train_loop = TrainLoop(NetUnifiedBatchFeed(), None, None, None, DummyOptimizer(), None)
        train_loop.callbacks_handler.register_callbacks([callback])
        
        self.assertEqual(result_pkg.experiment_path,
                         f'local_model_result_folder_path/project_name_experiment_name_{train_loop.experiment_timestamp}')


class TestEarlyStoppingCallback(unittest.TestCase):
    def test_basic_no_loss_change(self):
        self.basic_early_stop_change_check_loss(10., 10., [False, True])
        self.basic_early_stop_change_check_loss(22232.334, 22232.334, [False, True])

    def test_basic_no_acc_change(self):
        self.basic_early_stop_change_check_acc(10., 10., [False, True])
        self.basic_early_stop_change_check_acc(22232.334, 22232.334, [False, True])

    def test_basic_loss_drops(self):
        self.basic_early_stop_change_check_loss(10., 9.9, [False, False])
        self.basic_early_stop_change_check_loss(10223., 33.9, [False, False])

    def test_basic_acc_drops(self):
        self.basic_early_stop_change_check_acc(10., 9.9, [False, True])
        self.basic_early_stop_change_check_acc(10223., 33.9, [False, True])

    def test_basic_loss_grows(self):
        self.basic_early_stop_change_check_loss(10., 11.22, [False, True])
        self.basic_early_stop_change_check_loss(1., 11323.22, [False, True])

    def test_basic_acc_grows(self):
        self.basic_early_stop_change_check_acc(10., 11.22, [False, False])
        self.basic_early_stop_change_check_acc(1., 11323.22, [False, False])

    def test_delta_loss(self):
        self.basic_early_stop_change_check_loss(10., 8.1, [False, True], min_delta=2.)
        self.basic_early_stop_change_check_loss(10., 8.0, [False, True], min_delta=2.)
        self.basic_early_stop_change_check_loss(10., 7.9, [False, False], min_delta=2.)
        self.basic_early_stop_change_check_loss(10., 11.9, [False, True], min_delta=2.)

    def test_delta_acc(self):
        self.basic_early_stop_change_check_acc(10., 12.1, [False, False], min_delta=2.)
        self.basic_early_stop_change_check_acc(10., 12.0, [False, True], min_delta=2.)
        self.basic_early_stop_change_check_acc(10., 11.9, [False, True], min_delta=2.)
        self.basic_early_stop_change_check_acc(10., 9.5, [False, True], min_delta=2.)

    def basic_early_stop_change_check_loss(self, val1, val2, expected_result, min_delta=0.):
        callback = EarlyStopping(monitor='dummy_loss', min_delta=min_delta)
        train_loop = TrainLoop(NetUnifiedBatchFeed(), None, None, None, None, None)
        train_loop.callbacks_handler.register_callbacks([callback])
        self.assertFalse(train_loop.early_stop)

        result = []

        train_loop.insert_metric_result_into_history('dummy_loss', val1)
        callback.on_epoch_end()
        result.append(train_loop.early_stop)

        train_loop.insert_metric_result_into_history('dummy_loss', val2)
        callback.on_epoch_end()
        result.append(train_loop.early_stop)

        self.assertEqual(result, expected_result)

    def basic_early_stop_change_check_acc(self, val1, val2, expected_result, min_delta=0.):
        callback = EarlyStopping(monitor='dummy_acc', min_delta=min_delta)
        train_loop = TrainLoop(NetUnifiedBatchFeed(), None, None, None, None, None)
        train_loop.callbacks_handler.register_callbacks([callback])
        self.assertFalse(train_loop.early_stop)

        result = []

        train_loop.insert_metric_result_into_history('dummy_acc', val1)
        callback.on_epoch_end()
        result.append(train_loop.early_stop)

        train_loop.insert_metric_result_into_history('dummy_acc', val2)
        callback.on_epoch_end()
        result.append(train_loop.early_stop)

        self.assertEqual(result, expected_result)

    def test_patience_loss(self):
        self.eval_patience(min_delta=0., patience=0,
                           val_list=[10., 10., 10.], expected_result=[False, True, True], monitor='dummy_loss')
        self.eval_patience(min_delta=0., patience=1,
                           val_list=[10., 10., 10.], expected_result=[False, False, True], monitor='dummy_loss')
        self.eval_patience(min_delta=0., patience=2,
                           val_list=[10., 10., 10., 10.], expected_result=[False, False, False, True], monitor='dummy_loss')

        self.eval_patience(min_delta=1., patience=1,
                           val_list=[10., 9., 8.], expected_result=[False, False, False], monitor='dummy_loss')
        self.eval_patience(min_delta=1., patience=1,
                           val_list=[10., 9.4, 8.9], expected_result=[False, False, False], monitor='dummy_loss')
        self.eval_patience(min_delta=1., patience=1,
                           val_list=[10., 9.4, 9.], expected_result=[False, False, True], monitor='dummy_loss')
        self.eval_patience(min_delta=1., patience=2,
                           val_list=[10., 9.4, 9.], expected_result=[False, False, False], monitor='dummy_loss')
        self.eval_patience(min_delta=1., patience=2,
                           val_list=[10., 9.4, 9., 9.], expected_result=[False, False, False, True], monitor='dummy_loss')
        self.eval_patience(min_delta=1., patience=2,
                           val_list=[10., 9.4, 9., 8.9], expected_result=[False, False, False, False], monitor='dummy_loss')

    def test_patience_acc(self):
        self.eval_patience(min_delta=0., patience=0,
                           val_list=[10., 10., 10.], expected_result=[False, True, True], monitor='dummy_acc')
        self.eval_patience(min_delta=0., patience=1,
                           val_list=[10., 10., 10.], expected_result=[False, False, True], monitor='dummy_acc')
        self.eval_patience(min_delta=0., patience=2,
                           val_list=[10., 10., 10., 10.], expected_result=[False, False, False, True], monitor='dummy_acc')

        self.eval_patience(min_delta=1., patience=1,
                           val_list=[8., 9., 10.], expected_result=[False, False, False], monitor='dummy_acc')
        self.eval_patience(min_delta=1., patience=1,
                           val_list=[8.9, 9.4, 10.], expected_result=[False, False, False], monitor='dummy_acc')
        self.eval_patience(min_delta=1., patience=1,
                           val_list=[9., 9.4, 10.], expected_result=[False, False, True], monitor='dummy_acc')
        self.eval_patience(min_delta=1., patience=2,
                           val_list=[9., 9.4, 10.], expected_result=[False, False, False], monitor='dummy_acc')
        self.eval_patience(min_delta=1., patience=2,
                           val_list=[9., 9., 9.4, 10.], expected_result=[False, False, False, True], monitor='dummy_acc')
        self.eval_patience(min_delta=1., patience=2,
                           val_list=[9., 9., 10.0, 10.0], expected_result=[False, False, False, True], monitor='dummy_acc')
        self.eval_patience(min_delta=1., patience=2,
                           val_list=[9., 9., 10.0, 10.1], expected_result=[False, False, False, False], monitor='dummy_acc')

    def eval_patience(self, min_delta, patience, val_list, expected_result, monitor):
        callback = EarlyStopping(monitor=monitor, min_delta=min_delta, patience=patience)
        train_loop = TrainLoop(NetUnifiedBatchFeed(), None, None, None, None, None)
        train_loop.callbacks_handler.register_callbacks([callback])
        self.assertFalse(train_loop.early_stop)

        result = []

        for val in val_list:
            train_loop.insert_metric_result_into_history(monitor, val)
            callback.on_epoch_end()
            result.append(train_loop.early_stop)

        self.assertEqual(result, expected_result)
