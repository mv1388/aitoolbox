import unittest
import os
import shutil
import torch

from aitoolbox.cloud.AWS.model_save import PyTorchS3ModelSaver
from aitoolbox.experiment.experiment_saver import FullPyTorchExperimentS3Saver
from aitoolbox.experiment.local_experiment_saver import FullPyTorchExperimentLocalSaver
from aitoolbox.experiment.local_save.local_model_save import PyTorchLocalModelSaver
from aitoolbox.torchtrain.callbacks.model_save import ModelCheckpoint, ModelIterationCheckpoint, ModelTrainEndSave
from aitoolbox.torchtrain.train_loop import TrainLoop
from tests.utils import NetUnifiedBatchFeed, MiniDummyOptimizer, DummyResultPackage, DummyOptimizer


THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class TestModelCheckpointCallback(unittest.TestCase):
    def test_init(self):
        callback_true = ModelCheckpoint('project_name', 'experiment_name', 'local_model_result_folder_path', hyperparams={},
                                        cloud_save_mode='s3')
        self.assertIsNone(callback_true.model_checkpointer)
        self.assertFalse(callback_true._hyperparams_already_saved)

        # callback_true = ModelCheckpoint('project_name', 'experiment_name', 'local_model_result_folder_path',
        #                                         cloud_save_mode='gcs')
        # self.assertEqual(type(callback_true.model_checkpointer), PyTorchGoogleStorageModelSaver)

        callback_false = ModelCheckpoint('project_name', 'experiment_name', 'local_model_result_folder_path', hyperparams={},
                                         cloud_save_mode=None)
        self.assertIsNone(callback_false.model_checkpointer)
        self.assertFalse(callback_true._hyperparams_already_saved)

    def test_checkpointer_type_on_train_start(self):
        callback = ModelCheckpoint('project_name', 'experiment_name', 'local_model_result_folder_path', hyperparams={},
                                   cloud_save_mode='s3')
        train_loop = TrainLoop(NetUnifiedBatchFeed(), None, None, None, DummyOptimizer(), None,
                               lazy_experiment_save=True)
        train_loop.callbacks_handler.register_callbacks([callback], cache_callbacks=True)
        self.assertIsNone(callback.model_checkpointer)
        train_loop.callbacks_handler.register_callbacks(None, cache_callbacks=False)
        self.assertEqual(type(callback.model_checkpointer), PyTorchS3ModelSaver)

        callback_2 = ModelCheckpoint('project_name', 'experiment_name', 'local_model_result_folder_path', hyperparams={},
                                     cloud_save_mode=None)
        train_loop = TrainLoop(NetUnifiedBatchFeed(), None, None, None, DummyOptimizer(), None,
                               lazy_experiment_save=True)
        train_loop.callbacks_handler.register_callbacks([callback_2], cache_callbacks=True)
        self.assertIsNone(callback_2.model_checkpointer)
        train_loop.callbacks_handler.register_callbacks(None, cache_callbacks=False)
        self.assertEqual(type(callback_2.model_checkpointer), PyTorchLocalModelSaver)

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
        train_loop.callbacks_handler.execute_train_begin()

        self.assertTrue(callback._hyperparams_already_saved)

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

    def test_save_args_false(self):
        hyperparams = {'param_1': 100, 'param_A': 234, 'LR': 0.001, 'path': 'bla/bladddd'}

        callback = ModelCheckpoint('project_name', 'experiment_name', THIS_DIR,
                                   hyperparams=hyperparams,
                                   cloud_save_mode=None)
        train_loop = TrainLoop(NetUnifiedBatchFeed(), None, None, None, DummyOptimizer(), None,
                               lazy_experiment_save=True)
        train_loop.callbacks_handler.register_callbacks([callback])
        train_loop.callbacks_handler.execute_train_begin()

        self.assertFalse(callback._hyperparams_already_saved)


class TestModelIterationCheckpoint(unittest.TestCase):
    def test_end_of_batch_model_saving_with_iteration_info(self):
        hyperparams = {'param_1': 100, 'param_A': 234, 'LR': 0.001, 'path': 'bla/bladddd'}

        callback = ModelIterationCheckpoint(
            2, 'project_name', 'experiment_name', THIS_DIR,
            hyperparams=hyperparams,
            cloud_save_mode=None)
        train_loop = TrainLoop(NetUnifiedBatchFeed(), None, None, None, DummyOptimizer(), None)
        train_loop.callbacks_handler.register_callbacks([callback])
        train_loop.callbacks_handler.execute_train_begin()

        for iteration in range(10):
            train_loop.total_iteration_idx = iteration
            train_loop.callbacks_handler.execute_batch_end()

            if iteration % 5 == 0:
                train_loop.callbacks_handler.execute_epoch_end()
                train_loop.epoch += 1

        experiment_dir_path = os.path.join(THIS_DIR, 'project_name',
                                           f'experiment_name_{train_loop.experiment_timestamp}')

        model_file_names = \
            [
                f'model_experiment_name_{train_loop.experiment_timestamp}_E{ep}.pth'
                for ep in range(2)
            ] + \
            [
                f'model_experiment_name_{train_loop.experiment_timestamp}_E1_ITER2.pth',
                f'model_experiment_name_{train_loop.experiment_timestamp}_E1_ITER4.pth',
                f'model_experiment_name_{train_loop.experiment_timestamp}_E2_ITER6.pth',
                f'model_experiment_name_{train_loop.experiment_timestamp}_E2_ITER8.pth',
            ]

        self.assertEqual(
            sorted(os.listdir(os.path.join(experiment_dir_path, 'checkpoint_model'))),
            sorted(model_file_names)
        )

        expected_ctrs = [(0, 0), (1, 5), (1, 2), (1, 4), (2, 6), (2, 8)]
        self.assertEqual(len(model_file_names), len(expected_ctrs))

        for file_name, (exp_epoch, exp_iter) in zip(sorted(model_file_names), expected_ctrs):
            file_path = os.path.join(experiment_dir_path, 'checkpoint_model', file_name)
            model_dict = torch.load(file_path)

            self.assertEqual(model_dict['epoch'], exp_epoch)
            self.assertEqual(model_dict['iteration_idx'], exp_iter)

        project_path = os.path.join(THIS_DIR, 'project_name')
        if os.path.exists(project_path):
            shutil.rmtree(project_path)


class TestModelTrainEndSaveCallback(unittest.TestCase):
    def test_init(self):
        callback_true = ModelTrainEndSave('project_name', 'experiment_name', 'local_model_result_folder_path',
                                          {}, DummyResultPackage(), cloud_save_mode='s3')
        self.assertIsNone(callback_true.results_saver)

        # callback_true = ModelTrainEndSave('project_name', 'experiment_name', 'local_model_result_folder_path',
        #                                           {}, DummyResultPackage(), cloud_save_mode='gcs')
        # self.assertEqual(type(callback_true.results_saver), FullPyTorchExperimentGoogleStorageSaver)

        callback_false = ModelTrainEndSave('project_name', 'experiment_name', 'local_model_result_folder_path',
                                           {}, DummyResultPackage(), cloud_save_mode=None)
        self.assertIsNone(callback_false.results_saver)

    def test_end_saver_type_on_train_start(self):
        callback = ModelTrainEndSave('project_name', 'experiment_name', 'local_model_result_folder_path',
                                     {}, DummyResultPackage(), cloud_save_mode='s3')
        train_loop = TrainLoop(NetUnifiedBatchFeed(), None, None, None, DummyOptimizer(), None,
                               lazy_experiment_save=True)
        train_loop.callbacks_handler.register_callbacks([callback], cache_callbacks=True)
        self.assertIsNone(callback.results_saver)
        train_loop.callbacks_handler.register_callbacks(None, cache_callbacks=False)
        self.assertEqual(type(callback.results_saver), FullPyTorchExperimentS3Saver)

        callback_2 = ModelTrainEndSave('project_name', 'experiment_name', 'local_model_result_folder_path',
                                       {}, DummyResultPackage(), cloud_save_mode=None)
        train_loop = TrainLoop(NetUnifiedBatchFeed(), None, None, None, DummyOptimizer(), None,
                               lazy_experiment_save=True)
        train_loop.callbacks_handler.register_callbacks([callback_2], cache_callbacks=True)
        self.assertIsNone(callback_2.results_saver)
        train_loop.callbacks_handler.register_callbacks(None, cache_callbacks=False)
        self.assertEqual(type(callback_2.results_saver), FullPyTorchExperimentLocalSaver)

    def test_train_loop_reg_set_experiment_dir_path_for_additional_results(self):
        result_pkg = DummyResultPackage()
        self.assertIsNone(result_pkg.experiment_path)

        callback = ModelTrainEndSave('project_name', 'experiment_name', 'local_model_result_folder_path',
                                     {}, result_pkg)
        train_loop = TrainLoop(NetUnifiedBatchFeed(), None, None, None, DummyOptimizer(), None,
                               lazy_experiment_save=True)
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
        train_loop.callbacks_handler.execute_train_begin()

        self.assertTrue(callback._hyperparams_already_saved)

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
