import unittest

from tests.utils import *

from AIToolbox.torchtrain.train_loop import TrainLoop, TrainLoopModelCheckpoint, TrainLoopModelEndSave, TrainLoopModelCheckpointEndSave
from AIToolbox.torchtrain.callbacks.callback_handler import CallbacksHandler
from AIToolbox.torchtrain.callbacks.callbacks import ModelCheckpointCallback, ModelTrainEndSaveCallback


class TestTrainLoop(unittest.TestCase):
    def test_init_values(self):
        train_loop_non_val = TrainLoop(Net(), None, None, None, DeactivateModelFeedDefinition(), None, None)
        self.assertEqual(train_loop_non_val.train_history, {'loss': [], 'accumulated_loss': []})

        train_loop = TrainLoop(Net(), None, 100, None, DeactivateModelFeedDefinition(), None, None)
        self.assertEqual(train_loop.train_history, {'loss': [], 'accumulated_loss': [], 'val_loss': []})
        
        self.assertEqual(train_loop.callbacks, [])
        self.assertIsInstance(train_loop.callbacks_handler, CallbacksHandler)
        self.assertEqual(train_loop.callbacks_handler.train_loop_obj, train_loop)
        self.assertFalse(train_loop.early_stop)

    def test_callback_registration(self):
        train_loop = TrainLoop(Net(), None, 100, None, DeactivateModelFeedDefinition(), None, None)
        train_loop.callbacks_handler.register_callbacks([AbstractCallback('callback_test1'),
                                                         CallbackTracker(), CallbackTrackerShort()])

        self.assertEqual(len(train_loop.callbacks), 3)
        for reg_cb, true_cb in zip(train_loop.callbacks, [AbstractCallback, CallbackTracker, CallbackTrackerShort]):
            self.assertEqual(type(reg_cb), true_cb)
        for reg_cb in train_loop.callbacks:
            self.assertEqual(reg_cb.train_loop_obj, train_loop)

        train_loop.callbacks_handler.register_callbacks([AbstractCallback('callback_test2')])
        self.assertEqual(len(train_loop.callbacks), 4)
        for reg_cb, true_cb in zip(train_loop.callbacks, [AbstractCallback, CallbackTracker, CallbackTrackerShort, AbstractCallback]):
            self.assertEqual(type(reg_cb), true_cb)

        for reg_cb in train_loop.callbacks:
            self.assertEqual(reg_cb.train_loop_obj, train_loop)

        for reg_cb, cb_name in zip(train_loop.callbacks,
                                   ['callback_test1', 'CallbackTracker1', 'CallbackTracker2', 'callback_test2']):
            self.assertEqual(reg_cb.callback_name, cb_name)

    def test_callback_on_execution(self):
        num_epochs = 2
        dummy_feed_def = DeactivateModelFeedDefinition()
        dummy_optimizer = DummyOptimizer()
        dummy_train_loader = list(range(4))
        dummy_val_loader = list(range(3))
        dummy_test_loader = list(range(2))

        model = Net()
        train_loop = TrainLoop(model, dummy_train_loader, dummy_val_loader, dummy_test_loader,
                               dummy_feed_def, dummy_optimizer, None)
        train_loop.callbacks_handler.register_callbacks([AbstractCallback('callback_test1'),
                                                         CallbackTracker(), CallbackTrackerShort(),
                                                         AbstractCallback('callback_test2')])

        callback_full = train_loop.callbacks[1]
        callback_short = train_loop.callbacks[2]

        model_return = train_loop.do_train(num_epoch=num_epochs)

        self.assertEqual(model, model_return)
        self.assertFalse(train_loop.early_stop)
        self.assertEqual(train_loop.epoch, num_epochs-1)

        self.assertEqual(dummy_feed_def.dummy_batch.back_ctr, num_epochs*len(dummy_train_loader))
        self.assertEqual(dummy_feed_def.dummy_batch.item_ctr,
                         num_epochs * len(dummy_train_loader) + num_epochs * len(dummy_train_loader) +
                         num_epochs * len(dummy_val_loader) +
                         len(dummy_test_loader))

        self.assertEqual(dummy_optimizer.zero_grad_ctr, num_epochs*len(dummy_train_loader))
        self.assertEqual(dummy_optimizer.step_ctr, num_epochs * len(dummy_train_loader))

        self.assertEqual(callback_full.callback_calls,
                         ['on_train_loop_registration', 'on_train_begin', 'on_epoch_begin', 'on_batch_begin',
                          'on_batch_end', 'on_batch_begin', 'on_batch_end', 'on_batch_begin', 'on_batch_end',
                          'on_batch_begin', 'on_batch_end', 'on_epoch_end', 'on_epoch_begin', 'on_batch_begin',
                          'on_batch_end', 'on_batch_begin', 'on_batch_end', 'on_batch_begin', 'on_batch_end',
                          'on_batch_begin', 'on_batch_end', 'on_epoch_end', 'on_train_end'])
        self.assertEqual(callback_full.call_ctr,
                         {'on_train_loop_registration': 1, 'on_epoch_begin': 2, 'on_epoch_end': 2, 'on_train_begin': 1,
                          'on_train_end': 1, 'on_batch_begin': 8, 'on_batch_end': 8})

        self.assertEqual(callback_short.callback_calls,
                         ['on_epoch_begin', 'on_batch_begin', 'on_batch_begin', 'on_batch_begin', 'on_batch_begin',
                          'on_epoch_end', 'on_epoch_begin', 'on_batch_begin', 'on_batch_begin', 'on_batch_begin',
                          'on_batch_begin', 'on_epoch_end', 'on_train_end'])
        self.assertEqual(callback_short.call_ctr,
                         {'on_train_loop_registration': 0, 'on_epoch_begin': 2, 'on_epoch_end': 2, 'on_train_begin': 0,
                          'on_train_end': 1, 'on_batch_begin': 8, 'on_batch_end': 0})

    def test_predict_train_data(self):
        self.eval_prediction('train')

    def test_predict_val_data(self):
        self.eval_prediction('val')

    def test_predict_test_data(self):
        self.eval_prediction('test')

    def eval_prediction(self, eval_mode):
        num_epochs = 2
        dummy_feed_def = DeactivateModelFeedDefinition()
        dummy_optimizer = DummyOptimizer()
        dummy_train_loader = list(range(4))
        dummy_val_loader = list(range(3))
        dummy_test_loader = list(range(2))

        model = Net()
        train_loop = TrainLoop(model, dummy_train_loader, dummy_val_loader, dummy_test_loader,
                               dummy_feed_def, dummy_optimizer, None)
        train_loop.callbacks_handler.register_callbacks([AbstractCallback('callback_test1'),
                                                         CallbackTracker(), CallbackTrackerShort(),
                                                         AbstractCallback('callback_test2')])
        train_loop.do_train(num_epoch=num_epochs)

        if eval_mode == 'train':
            y_test, y_pred, metadata = train_loop.predict_on_train_set()
            data_loader = dummy_train_loader
        elif eval_mode == 'val':
            y_test, y_pred, metadata = train_loop.predict_on_validation_set()
            data_loader = dummy_val_loader
        else:
            y_test, y_pred, metadata = train_loop.predict_on_test_set()
            data_loader = dummy_test_loader

        r = []
        for i in range(1, len(data_loader) + 1):
            r += [i] * 64
        r2 = []
        for i in range(1, len(data_loader) + 1):
            r2 += [i + 100] * 64
        self.assertEqual(y_test.tolist(), r)
        self.assertEqual(y_pred.tolist(), r2)

        d = {'bla': []}
        for i in range(1, len(data_loader) + 1):
            d['bla'] += [i + 200] * 64
        self.assertEqual(metadata, d)

    def test_basic_history_tracking(self):
        num_epochs = 2
        dummy_feed_def = DeactivateModelFeedDefinition()
        dummy_optimizer = DummyOptimizer()
        dummy_train_loader = list(range(4))
        dummy_val_loader = list(range(3))
        dummy_test_loader = list(range(2))

        model = Net()
        train_loop = TrainLoop(model, dummy_train_loader, dummy_val_loader, dummy_test_loader,
                               dummy_feed_def, dummy_optimizer, None)
        train_loop.callbacks_handler.register_callbacks([AbstractCallback('callback_test1'),
                                                         CallbackTracker(), CallbackTrackerShort(),
                                                         AbstractCallback('callback_test2')])
        train_loop.do_train(num_epoch=num_epochs)

        self.assertEqual(train_loop.train_history, {'loss': [1.0, 1.0], 'accumulated_loss': [1.0, 1.0], 'val_loss': [1.0, 1.0]})

        train_loop.insert_metric_result_into_history('test_metric_1', 10.11)
        train_loop.insert_metric_result_into_history('test_metric_2', 40.11)
        train_loop.insert_metric_result_into_history('test_metric_1', 100.11)
        train_loop.insert_metric_result_into_history('test_metric_2', 400.11)

        self.assertEqual(train_loop.train_history,
                         {'loss': [1.0, 1.0], 'accumulated_loss': [1.0, 1.0], 'val_loss': [1.0, 1.0],
                          'test_metric_1': [10.11, 100.11], 'test_metric_2': [40.11, 400.11]})


class TestTrainLoopModelCheckpoint(unittest.TestCase):
    def test_init_values(self):
        train_loop_non_val = TrainLoopModelCheckpoint(Net(), None, None, None, DeactivateModelFeedDefinition(), None, None,
                                                      "project_name", "experiment_name", "local_model_result_folder_path")
        self.assertEqual(train_loop_non_val.train_history, {'loss': [], 'accumulated_loss': []})

        train_loop = TrainLoopModelCheckpoint(Net(), None, 100, None, DeactivateModelFeedDefinition(), None, None,
                                              "project_name", "experiment_name", "local_model_result_folder_path")
        self.assertEqual(train_loop.train_history, {'loss': [], 'accumulated_loss': [], 'val_loss': []})

        self.assertEqual(len(train_loop.callbacks), 1)
        self.assertEqual(type(train_loop.callbacks[0]), ModelCheckpointCallback)

        self.assertIsInstance(train_loop.callbacks_handler, CallbacksHandler)
        self.assertEqual(train_loop.callbacks_handler.train_loop_obj, train_loop)
        self.assertFalse(train_loop.early_stop)


class TestTrainLoopModelEndSave(unittest.TestCase):
    def test_init_values(self):
        train_loop_non_val = TrainLoopModelEndSave(Net(), None, None, None, DeactivateModelFeedDefinition(), None, None,
                                                   "project_name", "experiment_name", "local_model_result_folder_path",
                                                   args={}, val_result_package=DummyResultPackage(),
                                                   test_result_package=DummyResultPackage(), save_to_s3=True)
        self.assertEqual(train_loop_non_val.train_history, {'loss': [], 'accumulated_loss': []})

        dummy_result_package = DummyResultPackage()
        train_loop = TrainLoopModelEndSave(Net(), None, 100, None, DeactivateModelFeedDefinition(), None, None,
                                           "project_name", "experiment_name", "local_model_result_folder_path",
                                           args={}, val_result_package=dummy_result_package, save_to_s3=True)
        self.assertEqual(train_loop.train_history, {'loss': [], 'accumulated_loss': [], 'val_loss': []})

        self.assertEqual(len(train_loop.callbacks), 1)
        self.assertEqual(type(train_loop.callbacks[0]), ModelTrainEndSaveCallback)
        self.assertEqual(train_loop.callbacks[0].val_result_package, dummy_result_package)
        self.assertEqual(train_loop.callbacks[0].test_result_package, None)
        self.assertEqual(train_loop.callbacks[0].result_package, None)

        self.assertIsInstance(train_loop.callbacks_handler, CallbacksHandler)
        self.assertEqual(train_loop.callbacks_handler.train_loop_obj, train_loop)
        self.assertFalse(train_loop.early_stop)


class TestTrainLoopModelCheckpointEndSave(unittest.TestCase):
    def test_init_values(self):
        train_loop_non_val = TrainLoopModelCheckpointEndSave(Net(), None, None, None, DeactivateModelFeedDefinition(), None, None,
                                                             "project_name", "experiment_name", "local_model_result_folder_path",
                                                             args={}, val_result_package=DummyResultPackage(), save_to_s3=True)
        self.assertEqual(train_loop_non_val.train_history, {'loss': [], 'accumulated_loss': []})

        dummy_result_package = DummyResultPackage()
        train_loop = TrainLoopModelCheckpointEndSave(Net(), None, 100, None, DeactivateModelFeedDefinition(), None, None,
                                                     "project_name", "experiment_name", "local_model_result_folder_path",
                                                     args={}, val_result_package=dummy_result_package, save_to_s3=True)
        self.assertEqual(train_loop.train_history, {'loss': [], 'accumulated_loss': [], 'val_loss': []})

        self.assertEqual(len(train_loop.callbacks), 2)
        self.assertEqual(type(train_loop.callbacks[0]), ModelTrainEndSaveCallback)
        self.assertEqual(train_loop.callbacks[0].val_result_package, dummy_result_package)
        self.assertEqual(train_loop.callbacks[0].test_result_package, None)
        self.assertEqual(train_loop.callbacks[0].result_package, None)

        self.assertEqual(type(train_loop.callbacks[1]), ModelCheckpointCallback)

        self.assertIsInstance(train_loop.callbacks_handler, CallbacksHandler)
        self.assertEqual(train_loop.callbacks_handler.train_loop_obj, train_loop)
        self.assertFalse(train_loop.early_stop)

    def test_init_val_test_loader_values(self):
        dummy_result_package_val = DummyResultPackage()
        dummy_result_package_test = DummyResultPackage()
        train_loop = TrainLoopModelCheckpointEndSave(Net(), None, 100, None, DeactivateModelFeedDefinition(), None,
                                                     None,
                                                     "project_name", "experiment_name",
                                                     "local_model_result_folder_path",
                                                     args={},
                                                     val_result_package=dummy_result_package_val,
                                                     test_result_package=dummy_result_package_test,
                                                     save_to_s3=True)
        self.assertEqual(train_loop.callbacks[0].val_result_package, dummy_result_package_val)
        self.assertEqual(train_loop.callbacks[0].test_result_package, dummy_result_package_test)
        self.assertEqual(train_loop.callbacks[0].result_package, None)

    def test_callback_registration(self):
        dummy_result_package = DummyResultPackage()
        train_loop = TrainLoopModelCheckpointEndSave(Net(), None, 100, None, DeactivateModelFeedDefinition(), None, None,
                                                     "project_name", "experiment_name",
                                                     "local_model_result_folder_path",
                                                     args={}, val_result_package=dummy_result_package, save_to_s3=True)

        self.assertEqual(len(train_loop.callbacks), 2)
        for reg_cb, true_cb in zip(train_loop.callbacks, [ModelTrainEndSaveCallback, ModelCheckpointCallback]):
            self.assertEqual(type(reg_cb), true_cb)
        for reg_cb in train_loop.callbacks:
            self.assertEqual(reg_cb.train_loop_obj, train_loop)

        train_loop.callbacks_handler.register_callbacks([AbstractCallback('callback_test2')])
        self.assertEqual(len(train_loop.callbacks), 3)
        for reg_cb, true_cb in zip(train_loop.callbacks, [ModelTrainEndSaveCallback, ModelCheckpointCallback, AbstractCallback]):
            self.assertEqual(type(reg_cb), true_cb)

        for reg_cb in train_loop.callbacks:
            self.assertEqual(reg_cb.train_loop_obj, train_loop)

        for reg_cb, cb_name in zip(train_loop.callbacks,
                                   ['Model save at the end of training',  'Model checkpoint at end of epoch', 'callback_test2']):
            self.assertEqual(reg_cb.callback_name, cb_name)
