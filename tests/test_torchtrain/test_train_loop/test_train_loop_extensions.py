import unittest

from aitoolbox.torchtrain import TrainLoopCheckpoint, TrainLoopEndSave, TrainLoopCheckpointEndSave
from aitoolbox.torchtrain.callbacks.abstract import AbstractCallback
from aitoolbox.torchtrain.callbacks.model_save import ModelCheckpoint, ModelTrainEndSave
from aitoolbox.torchtrain.tl_components.callback_handler import CallbacksHandler
from tests.utils import NetUnifiedBatchFeed, DummyOptimizer, MiniDummyOptimizer, DummyResultPackage


class TestTrainLoopCheckpoint(unittest.TestCase):
    def test_init_values(self):
        train_loop_non_val = TrainLoopCheckpoint(NetUnifiedBatchFeed(), None, None, None, DummyOptimizer(), None,
                                                 "project_name", "experiment_name", "local_model_result_folder_path", {})
        self.assertEqual(train_loop_non_val.train_history.train_history, {'loss': [], 'accumulated_loss': []})

        train_loop = TrainLoopCheckpoint(NetUnifiedBatchFeed(), None, 100, None, DummyOptimizer(), None,
                                         "project_name", "experiment_name", "local_model_result_folder_path", {})
        self.assertEqual(train_loop.train_history.train_history, {'loss': [], 'accumulated_loss': [], 'val_loss': []})

        self.assertEqual(len(train_loop.callbacks), 0)
        self.assertEqual(len(train_loop.callbacks_handler.callbacks_cache), 1)
        self.assertEqual(type(train_loop.callbacks_handler.callbacks_cache[0]), ModelCheckpoint)

        train_loop.callbacks_handler.register_callbacks([], cache_callbacks=False)
        self.assertEqual(len(train_loop.callbacks), 1)
        self.assertEqual(type(train_loop.callbacks[0]), ModelCheckpoint)

        self.assertIsInstance(train_loop.callbacks_handler, CallbacksHandler)
        self.assertEqual(train_loop.callbacks_handler.train_loop_obj, train_loop)
        self.assertFalse(train_loop.early_stop)

    def test_optimizer_missing_state_dict_exception(self):
        raised = False
        try:
            TrainLoopCheckpoint(NetUnifiedBatchFeed(), None, None, None, MiniDummyOptimizer(), None,
                                "project_name", "experiment_name",
                                "local_model_result_folder_path", {})
        except AttributeError:
            raised = True
        self.assertFalse(raised)

        with self.assertRaises(AttributeError):
            TrainLoopCheckpoint(
                NetUnifiedBatchFeed(), None, None, None, MiniDummyOptimizer(), None,
                "project_name", "experiment_name",
                "local_model_result_folder_path", {}
            ).callbacks_handler.register_callbacks(None)


class TestTrainLoopEndSave(unittest.TestCase):
    def test_init_values(self):
        train_loop_non_val = TrainLoopEndSave(NetUnifiedBatchFeed(), None, 100, 100, DummyOptimizer(), None,
                                              "project_name", "experiment_name", "local_model_result_folder_path",
                                              hyperparams={}, val_result_package=DummyResultPackage(),
                                              test_result_package=DummyResultPackage(), cloud_save_mode='s3')
        self.assertEqual(train_loop_non_val.train_history.train_history, {'loss': [], 'accumulated_loss': [], 'val_loss': []})

        dummy_result_package = DummyResultPackage()
        train_loop = TrainLoopEndSave(NetUnifiedBatchFeed(), None, 100, None, DummyOptimizer(), None,
                                      "project_name", "experiment_name", "local_model_result_folder_path",
                                      hyperparams={}, val_result_package=dummy_result_package, cloud_save_mode='s3')
        self.assertEqual(train_loop.train_history.train_history, {'loss': [], 'accumulated_loss': [], 'val_loss': []})

        self.assertEqual(len(train_loop.callbacks), 0)
        self.assertEqual(len(train_loop.callbacks_handler.callbacks_cache), 1)
        self.assertEqual(type(train_loop.callbacks_handler.callbacks_cache[0]), ModelTrainEndSave)

        train_loop.callbacks_handler.register_callbacks([], cache_callbacks=False)
        self.assertEqual(len(train_loop.callbacks), 1)
        self.assertEqual(type(train_loop.callbacks[0]), ModelTrainEndSave)
        self.assertEqual(train_loop.callbacks[0].val_result_package, dummy_result_package)
        self.assertEqual(train_loop.callbacks[0].test_result_package, None)
        self.assertEqual(train_loop.callbacks[0].result_package, None)

        self.assertIsInstance(train_loop.callbacks_handler, CallbacksHandler)
        self.assertEqual(train_loop.callbacks_handler.train_loop_obj, train_loop)
        self.assertFalse(train_loop.early_stop)

    def test_loader_package_exceptions(self):
        with self.assertRaises(ValueError):
            TrainLoopEndSave(NetUnifiedBatchFeed(), None, None, 100, None,
                             None,
                             "project_name", "experiment_name",
                             "local_model_result_folder_path",
                             hyperparams={},
                             val_result_package=DummyResultPackage(),
                             test_result_package=DummyResultPackage(),
                             cloud_save_mode='s3')
        with self.assertRaises(ValueError):
            TrainLoopEndSave(NetUnifiedBatchFeed(), None, 100, None, None,
                             None,
                             "project_name", "experiment_name",
                             "local_model_result_folder_path",
                             hyperparams={},
                             val_result_package=DummyResultPackage(),
                             test_result_package=DummyResultPackage(),
                             cloud_save_mode='s3')
        with self.assertRaises(ValueError):
            TrainLoopEndSave(NetUnifiedBatchFeed(), None, 100, None, None,
                             None,
                             "project_name", "experiment_name",
                             "local_model_result_folder_path",
                             hyperparams={},
                             test_result_package=DummyResultPackage(),
                             cloud_save_mode='s3')
        with self.assertRaises(ValueError):
            TrainLoopEndSave(NetUnifiedBatchFeed(), None, None, None, None,
                             None,
                             "project_name", "experiment_name",
                             "local_model_result_folder_path",
                             hyperparams={},
                             val_result_package=DummyResultPackage(),
                             test_result_package=DummyResultPackage(),
                             cloud_save_mode='s3')
        with self.assertRaises(ValueError):
            TrainLoopEndSave(NetUnifiedBatchFeed(), None, 100, 100, None,
                             None,
                             "project_name", "experiment_name",
                             "local_model_result_folder_path",
                             hyperparams={},
                             val_result_package=None,
                             test_result_package=None,
                             cloud_save_mode='s3')


class TestTrainLoopCheckpointEndSave(unittest.TestCase):
    def test_init_values(self):
        train_loop_non_val = TrainLoopCheckpointEndSave(NetUnifiedBatchFeed(), None, 100, None, DummyOptimizer(), None,
                                                        "project_name", "experiment_name", "local_model_result_folder_path",
                                                        hyperparams={}, val_result_package=DummyResultPackage(), cloud_save_mode='s3')
        self.assertEqual(train_loop_non_val.train_history.train_history, {'loss': [], 'accumulated_loss': [], 'val_loss': []})

        dummy_result_package = DummyResultPackage()
        train_loop = TrainLoopCheckpointEndSave(NetUnifiedBatchFeed(), None, 100, None, DummyOptimizer(), None,
                                                "project_name", "experiment_name", "local_model_result_folder_path",
                                                hyperparams={}, val_result_package=dummy_result_package, cloud_save_mode='s3')
        self.assertEqual(train_loop.train_history.train_history, {'loss': [], 'accumulated_loss': [], 'val_loss': []})

        self.assertEqual(len(train_loop.callbacks), 0)
        self.assertEqual(len(train_loop.callbacks_handler.callbacks_cache), 2)

        self.assertEqual(type(train_loop.callbacks_handler.callbacks_cache[0]), ModelTrainEndSave)
        self.assertEqual(train_loop.callbacks_handler.callbacks_cache[0].val_result_package, dummy_result_package)
        self.assertEqual(train_loop.callbacks_handler.callbacks_cache[0].test_result_package, None)
        self.assertEqual(train_loop.callbacks_handler.callbacks_cache[0].result_package, None)
        self.assertEqual(type(train_loop.callbacks_handler.callbacks_cache[1]), ModelCheckpoint)

        train_loop.callbacks_handler.register_callbacks([], cache_callbacks=False)
        self.assertEqual(len(train_loop.callbacks), 2)
        self.assertEqual(type(train_loop.callbacks[1]), ModelTrainEndSave)
        self.assertEqual(train_loop.callbacks[1].val_result_package, dummy_result_package)
        self.assertEqual(train_loop.callbacks[1].test_result_package, None)
        self.assertEqual(train_loop.callbacks[1].result_package, None)

        self.assertEqual(type(train_loop.callbacks[0]), ModelCheckpoint)

        self.assertIsInstance(train_loop.callbacks_handler, CallbacksHandler)
        self.assertEqual(train_loop.callbacks_handler.train_loop_obj, train_loop)
        self.assertFalse(train_loop.early_stop)

    def test_init_val_test_loader_values(self):
        dummy_result_package_val = DummyResultPackage()
        dummy_result_package_test = DummyResultPackage()
        train_loop = TrainLoopCheckpointEndSave(NetUnifiedBatchFeed(), None, 100, 100, DummyOptimizer(),
                                                None,
                                                "project_name", "experiment_name",
                                                "local_model_result_folder_path",
                                                hyperparams={},
                                                val_result_package=dummy_result_package_val,
                                                test_result_package=dummy_result_package_test,
                                                cloud_save_mode='s3')
        self.assertEqual(train_loop.callbacks_handler.callbacks_cache[0].val_result_package, dummy_result_package_val)
        self.assertEqual(train_loop.callbacks_handler.callbacks_cache[0].test_result_package, dummy_result_package_test)

        train_loop.callbacks_handler.register_callbacks([], cache_callbacks=False)
        self.assertEqual(train_loop.callbacks[1].val_result_package, dummy_result_package_val)
        self.assertEqual(train_loop.callbacks[1].test_result_package, dummy_result_package_test)
        self.assertEqual(train_loop.callbacks[1].result_package, None)

    def test_callback_registration(self):
        dummy_result_package = DummyResultPackage()
        train_loop = TrainLoopCheckpointEndSave(NetUnifiedBatchFeed(), None, 100, None, DummyOptimizer(), None,
                                                "project_name", "experiment_name",
                                                "local_model_result_folder_path",
                                                hyperparams={}, val_result_package=dummy_result_package,
                                                cloud_save_mode='s3')

        self.assertEqual(len(train_loop.callbacks), 0)
        self.assertEqual(len(train_loop.callbacks_handler.callbacks_cache), 2)
        # In the cache the callbacks aren't yet sorted based on specified priority
        for reg_cb, true_cb in zip(train_loop.callbacks_handler.callbacks_cache, [ModelTrainEndSave, ModelCheckpoint]):
            self.assertEqual(type(reg_cb), true_cb)

        train_loop.callbacks_handler.register_callbacks([], cache_callbacks=False)
        for reg_cb in train_loop.callbacks:
            self.assertEqual(reg_cb.train_loop_obj, train_loop)

        train_loop.callbacks_handler.register_callbacks([AbstractCallback('callback_test2')])
        self.assertEqual(len(train_loop.callbacks), 3)
        # After callbacks are released from cache they also get sorted by priority
        for reg_cb, true_cb in zip(train_loop.callbacks, [ModelCheckpoint, AbstractCallback, ModelTrainEndSave]):
            self.assertEqual(type(reg_cb), true_cb)

        for reg_cb in train_loop.callbacks:
            self.assertEqual(reg_cb.train_loop_obj, train_loop)

        for reg_cb, cb_name in zip(train_loop.callbacks,
                                   ['Model checkpoint at end of epoch', 'callback_test2', 'Model save at the end of training']):
            self.assertEqual(reg_cb.callback_name, cb_name)

    def test_optimizer_missing_state_dict_exception(self):
        dummy_result_package = DummyResultPackage()

        with self.assertRaises(AttributeError):
            TrainLoopCheckpointEndSave(
                NetUnifiedBatchFeed(), None, 100, None, MiniDummyOptimizer(), None,
                "project_name", "experiment_name",
                "local_model_result_folder_path",
                hyperparams={}, val_result_package=dummy_result_package, cloud_save_mode='s3'
            ).callbacks_handler.register_callbacks(None)

    def test_loader_package_exceptions(self):
        with self.assertRaises(ValueError):
            TrainLoopCheckpointEndSave(NetUnifiedBatchFeed(), None, None, 100, DummyOptimizer(),
                                       None,
                                       "project_name", "experiment_name",
                                       "local_model_result_folder_path",
                                       hyperparams={},
                                       val_result_package=DummyResultPackage(),
                                       test_result_package=DummyResultPackage(),
                                       cloud_save_mode='s3')
        with self.assertRaises(ValueError):
            TrainLoopCheckpointEndSave(NetUnifiedBatchFeed(), None, 100, None, DummyOptimizer(),
                                       None,
                                       "project_name", "experiment_name",
                                       "local_model_result_folder_path",
                                       hyperparams={},
                                       val_result_package=DummyResultPackage(),
                                       test_result_package=DummyResultPackage(),
                                       cloud_save_mode='s3')
        with self.assertRaises(ValueError):
            TrainLoopCheckpointEndSave(NetUnifiedBatchFeed(), None, 100, None, DummyOptimizer(),
                                       None,
                                       "project_name", "experiment_name",
                                       "local_model_result_folder_path",
                                       hyperparams={},
                                       test_result_package=DummyResultPackage(),
                                       cloud_save_mode='s3')
        with self.assertRaises(ValueError):
            TrainLoopCheckpointEndSave(NetUnifiedBatchFeed(), None, None, None, DummyOptimizer(),
                                       None,
                                       "project_name", "experiment_name",
                                       "local_model_result_folder_path",
                                       hyperparams={},
                                       val_result_package=DummyResultPackage(),
                                       test_result_package=DummyResultPackage(),
                                       cloud_save_mode='s3')
        with self.assertRaises(ValueError):
            TrainLoopCheckpointEndSave(NetUnifiedBatchFeed(), None, 100, 100, DummyOptimizer(),
                                       None,
                                       "project_name", "experiment_name",
                                       "local_model_result_folder_path",
                                       hyperparams={},
                                       val_result_package=None,
                                       test_result_package=None,
                                       cloud_save_mode='s3')
