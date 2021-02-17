import unittest

from tests.utils import *

from aitoolbox.torchtrain.callbacks.abstract import AbstractCallback, AbstractExperimentCallback
from aitoolbox.torchtrain.train_loop import TrainLoop, TrainLoopCheckpointEndSave
from tests.utils import function_exists, NetUnifiedBatchFeed, CallbackTracker


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
        self.assertTrue(function_exists(callback, 'on_after_gradient_update'))
        self.assertTrue(function_exists(callback, 'on_after_optimizer_step'))

    def test_on_train_loop_registration_hook(self):
        train_loop = TrainLoop(NetUnifiedBatchFeed(), None, None, None, None, None)
        callback = CallbackTracker()
        callback.register_train_loop_object(train_loop)

        self.assertIsInstance(callback, AbstractCallback)
        self.assertEqual(callback.callback_calls, ['on_train_loop_registration'])


class TestAbstractExperimentCallback(unittest.TestCase):
    def test_init(self):
        callback = AbstractExperimentCallback('test_callback')
        self.assertTrue(function_exists(callback, 'try_infer_experiment_details'))
        self.assertIsNone(callback.project_name)
        self.assertIsNone(callback.experiment_name)
        self.assertIsNone(callback.local_model_result_folder_path)

    def test_try_infer_experiment_details_fail(self):
        callback = AbstractExperimentCallback('test_callback')
        model = NetUnifiedBatchFeed()

        train_loop_non_exp = TrainLoop(model, None, None, None, None, None)
        train_loop_non_exp.callbacks_handler.register_callbacks([callback])

        with self.assertRaises(AttributeError):
            callback.try_infer_experiment_details(infer_cloud_details=False)

        with self.assertRaises(AttributeError):
            callback.try_infer_experiment_details(infer_cloud_details=True)

    def test_try_infer_experiment_details(self):
        callback = AbstractExperimentCallback('test_callback')
        model = NetUnifiedBatchFeed()

        project_name = 'test_project'
        experiment_name = 'test_experiment'
        local_path = 'my_local_path'

        train_loop = TrainLoopCheckpointEndSave(model, None, [], None, DummyOptimizer(), None,
                                                project_name=project_name, experiment_name=experiment_name,
                                                local_model_result_folder_path=local_path,
                                                hyperparams={}, val_result_package=DummyResultPackageExtend(),
                                                cloud_save_mode=None,
                                                lazy_experiment_save=True)
        train_loop.callbacks_handler.register_callbacks([callback])
        callback.try_infer_experiment_details(infer_cloud_details=False)

        self.assertEqual(callback.project_name, project_name)
        self.assertEqual(callback.experiment_name, experiment_name)
        self.assertEqual(callback.local_model_result_folder_path, local_path)

    def test_try_infer_experiment_details_cloud(self):
        callback = AbstractExperimentCallback('test_callback')
        model = NetUnifiedBatchFeed()

        project_name = 'test_project'
        experiment_name = 'test_experiment'
        local_path = 'my_local_path'

        train_loop = TrainLoopCheckpointEndSave(model, None, [], None, DummyOptimizer(), None,
                                                project_name=project_name, experiment_name=experiment_name,
                                                local_model_result_folder_path=local_path,
                                                hyperparams={}, val_result_package=DummyResultPackageExtend(),
                                                lazy_experiment_save=True)
        train_loop.callbacks_handler.register_callbacks([callback])
        callback.try_infer_experiment_details(infer_cloud_details=True)

        self.assertEqual(callback.project_name, project_name)
        self.assertEqual(callback.experiment_name, experiment_name)
        self.assertEqual(callback.local_model_result_folder_path, local_path)

        self.assertEqual(callback.cloud_save_mode, train_loop.cloud_save_mode)
        self.assertEqual(callback.bucket_name, train_loop.bucket_name)
        self.assertEqual(callback.cloud_dir_prefix, train_loop.cloud_dir_prefix)

    def test_try_infer_experiment_details_cloud_spec(self):
        callback = AbstractExperimentCallback('test_callback')
        model = NetUnifiedBatchFeed()

        project_name = 'test_project'
        experiment_name = 'test_experiment'
        local_path = 'my_local_path'

        cloud_save_mode = 's3'
        bucket_name = 'my_fancy_bucket'
        cloud_dir_prefix = 'MyFolder_prefix'

        train_loop = TrainLoopCheckpointEndSave(model, None, [], None, DummyOptimizer(), None,
                                                project_name=project_name, experiment_name=experiment_name,
                                                local_model_result_folder_path=local_path,
                                                hyperparams={}, val_result_package=DummyResultPackageExtend(),
                                                cloud_save_mode=cloud_save_mode, bucket_name=bucket_name,
                                                cloud_dir_prefix=cloud_dir_prefix,
                                                lazy_experiment_save=True)
        train_loop.callbacks_handler.register_callbacks([callback])
        callback.try_infer_experiment_details(infer_cloud_details=True)

        self.assertEqual(callback.project_name, project_name)
        self.assertEqual(callback.experiment_name, experiment_name)
        self.assertEqual(callback.local_model_result_folder_path, local_path)

        self.assertEqual(callback.cloud_save_mode, train_loop.cloud_save_mode)
        self.assertEqual(callback.bucket_name, train_loop.bucket_name)
        self.assertEqual(callback.cloud_dir_prefix, train_loop.cloud_dir_prefix)

        self.assertEqual(callback.cloud_save_mode, cloud_save_mode)
        self.assertEqual(callback.bucket_name, bucket_name)
        self.assertEqual(callback.cloud_dir_prefix, cloud_dir_prefix)

    def test_override_train_loop_values_in_callback(self):
        project_name = 'test_project'
        experiment_name = 'test_experiment'
        local_path = 'my_local_path'
        cloud_save_mode = 'gcs'
        bucket_name = 'my_fancy_bucket'
        cloud_dir_prefix = 'MyFolder_prefix'

        callback = AbstractExperimentCallback('test_callback',
                                              project_name, experiment_name, local_path,
                                              cloud_save_mode, bucket_name, cloud_dir_prefix)
        model = NetUnifiedBatchFeed()

        train_loop = TrainLoopCheckpointEndSave(model, None, [], None, DummyOptimizer(), None,
                                                project_name=f'TL_{project_name}', experiment_name=f'TL_{experiment_name}',
                                                local_model_result_folder_path=f'TL_{local_path}',
                                                hyperparams={}, val_result_package=DummyResultPackageExtend(),
                                                cloud_save_mode='s3', bucket_name=f'TL_{bucket_name}',
                                                cloud_dir_prefix=f'TL_{cloud_dir_prefix}',
                                                lazy_experiment_save=True)
        train_loop.callbacks_handler.register_callbacks([callback])
        callback.try_infer_experiment_details(infer_cloud_details=True)

        self.assertEqual(callback.project_name, project_name)
        self.assertEqual(callback.experiment_name, experiment_name)
        self.assertEqual(callback.local_model_result_folder_path, local_path)

        self.assertEqual(callback.cloud_save_mode, cloud_save_mode)
        self.assertEqual(callback.bucket_name, bucket_name)
        self.assertEqual(callback.cloud_dir_prefix, cloud_dir_prefix)

        self.assertEqual(train_loop.cloud_save_mode, 's3')
        self.assertEqual(train_loop.bucket_name, f'TL_{bucket_name}')
        self.assertEqual(train_loop.cloud_dir_prefix, f'TL_{cloud_dir_prefix}')
