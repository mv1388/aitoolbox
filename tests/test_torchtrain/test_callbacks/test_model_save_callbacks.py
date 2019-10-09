import unittest
import os
import shutil

from aitoolbox.cloud.AWS.model_save import PyTorchS3ModelSaver
from aitoolbox.experiment.experiment_saver import FullPyTorchExperimentS3Saver
from aitoolbox.experiment.local_experiment_saver import FullPyTorchExperimentLocalSaver
from aitoolbox.experiment.local_save.local_model_save import PyTorchLocalModelSaver
from aitoolbox.torchtrain.callbacks.model_save_callbacks import ModelCheckpoint, ModelTrainEndSave
from aitoolbox.torchtrain.train_loop import TrainLoop
from tests.utils import NetUnifiedBatchFeed, MiniDummyOptimizer, DummyResultPackage, DummyOptimizer


THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class TestModelCheckpointCallback(unittest.TestCase):
    def test_init(self):
        callback_true = ModelCheckpoint('project_name', 'experiment_name', 'local_model_result_folder_path', hyperparams={},
                                        cloud_save_mode='s3')
        self.assertEqual(type(callback_true.model_checkpointer), PyTorchS3ModelSaver)
        self.assertFalse(callback_true._hyperparams_already_saved)

        # callback_true = ModelCheckpoint('project_name', 'experiment_name', 'local_model_result_folder_path',
        #                                         cloud_save_mode='gcs')
        # self.assertEqual(type(callback_true.model_checkpointer), PyTorchGoogleStorageModelSaver)

        callback_false = ModelCheckpoint('project_name', 'experiment_name', 'local_model_result_folder_path', hyperparams={},
                                         cloud_save_mode=None)
        self.assertEqual(type(callback_false.model_checkpointer), PyTorchLocalModelSaver)
        self.assertFalse(callback_true._hyperparams_already_saved)

    def test_optimizer_missing_state_dict_exception(self):
        callback = ModelCheckpoint('project_name', 'experiment_name', 'local_model_result_folder_path', hyperparams={},
                                   cloud_save_mode=None)
        train_loop = TrainLoop(NetUnifiedBatchFeed(), None, None, None, MiniDummyOptimizer(), None)

        with self.assertRaises(AttributeError):
            train_loop.callbacks_handler.register_callbacks([callback])

    def test_save_args(self):
        hyperparams = {'param_1': 100, 'param_A': 234, 'LR': 0.001, 'path': 'bla/bladddd'}

        callback = ModelCheckpoint('project_name', 'experiment_name', THIS_DIR,
                                   hyperparams=hyperparams,
                                   cloud_save_mode=None)
        train_loop = TrainLoop(NetUnifiedBatchFeed(), None, None, None, DummyOptimizer(), None)
        train_loop.callbacks_handler.register_callbacks([callback])

        self.assertFalse(callback._hyperparams_already_saved)

        callback.save_hyperparams()
        self.assertTrue(callback._hyperparams_already_saved)

        saved_file_path = os.path.join(THIS_DIR, 'project_name',
                                       f'experiment_name_{train_loop.experiment_timestamp}', 'hyperparams_list.txt')

        with open(saved_file_path, 'r') as f:
            f_lines = f.readlines()

        self.assertEqual(len(f_lines), len(hyperparams))
        self.assertEqual(sorted([el.split(':\t')[0] for el in f_lines]),
                         sorted(hyperparams.keys()))

        for line in f_lines:
            k, v = line.strip().split(':\t')
            self.assertEqual(str(hyperparams[k]), v)

        project_path = os.path.join(THIS_DIR, 'project_name')
        if os.path.exists(project_path):
            shutil.rmtree(project_path)


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

    def test_save_args(self):
        hyperparams = {'paramaaaaa_1': 10330, 'param_A': 234, 'LR': 0.001, 'path': 'bla/bladddd'}

        callback = ModelTrainEndSave('project_name', 'experiment_name', THIS_DIR,
                                     hyperparams=hyperparams,
                                     val_result_package=DummyResultPackage(),
                                     cloud_save_mode=None)
        train_loop = TrainLoop(NetUnifiedBatchFeed(), None, None, None, DummyOptimizer(), None)
        train_loop.callbacks_handler.register_callbacks([callback])

        self.assertFalse(callback._hyperparams_already_saved)

        callback.save_hyperparams()
        self.assertTrue(callback._hyperparams_already_saved)

        saved_file_path = os.path.join(THIS_DIR, 'project_name',
                                       f'experiment_name_{train_loop.experiment_timestamp}', 'hyperparams_list.txt')

        with open(saved_file_path, 'r') as f:
            f_lines = f.readlines()

        self.assertEqual(len(f_lines), len(hyperparams))
        self.assertEqual(sorted([el.split(':\t')[0] for el in f_lines]),
                         sorted(hyperparams.keys()))

        for line in f_lines:
            k, v = line.strip().split(':\t')
            self.assertEqual(str(hyperparams[k]), v)

        project_path = os.path.join(THIS_DIR, 'project_name')
        if os.path.exists(project_path):
            shutil.rmtree(project_path)
