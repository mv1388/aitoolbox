import unittest

from tests.test_torchtrain.utils import *

from AIToolbox.torchtrain.callbacks.callbacks import AbstractCallback, ModelCheckpointCallback, ModelTrainEndSaveCallback
from AIToolbox.torchtrain.train_loop import TrainLoop
from AIToolbox.AWS.model_save import PyTorchS3ModelSaver
from AIToolbox.experiment_save.local_model_save import PyTorchLocalModelSaver
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
        train_loop = TrainLoop(Net(), None, 100, DeactivateModelFeedDefinition(), None, None)
        callback = CallbackTracker()
        callback.register_train_loop_object(train_loop)

        self.assertIsInstance(callback, AbstractCallback)
        self.assertEqual(callback.callback_calls, ['on_train_loop_registration'])


class TestModelCheckpointCallback(unittest.TestCase):
    def test_init(self):
        callback_true = ModelCheckpointCallback('project_name', 'experiment_name', 'local_model_result_folder_path',
                                                save_to_s3=True)
        self.assertEqual(type(callback_true.model_checkpointer), PyTorchS3ModelSaver)

        callback_false = ModelCheckpointCallback('project_name', 'experiment_name', 'local_model_result_folder_path',
                                                 save_to_s3=False)
        self.assertEqual(type(callback_false.model_checkpointer), PyTorchLocalModelSaver)


class TestModelTrainEndSaveCallback(unittest.TestCase):
    def test_init(self):
        callback_true = ModelTrainEndSaveCallback('project_name', 'experiment_name', 'local_model_result_folder_path',
                                                  {}, DummyResultPackage(), save_to_s3=True)
        self.assertEqual(type(callback_true.results_saver), FullPyTorchExperimentS3Saver)

        callback_false = ModelTrainEndSaveCallback('project_name', 'experiment_name', 'local_model_result_folder_path',
                                                  {}, DummyResultPackage(), save_to_s3=False)
        self.assertEqual(type(callback_false.results_saver), FullPyTorchExperimentLocalSaver)

    def test_train_loop_reg_set_experiment_dir_path_for_additional_results(self):
        result_pkg = DummyResultPackage()
        self.assertIsNone(result_pkg.experiment_path)

        callback = ModelTrainEndSaveCallback('project_name', 'experiment_name', 'local_model_result_folder_path',
                                             {}, result_pkg)
        train_loop = TrainLoop(Net(), None, 100, DeactivateModelFeedDefinition(), None, None)
        train_loop.callbacks_handler.register_callbacks([callback])
        
        self.assertEqual(result_pkg.experiment_path,
                         f'local_model_result_folder_path/project_name_experiment_name_{train_loop.experiment_timestamp}')
