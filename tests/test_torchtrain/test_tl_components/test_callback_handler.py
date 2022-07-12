import unittest

from tests.utils import *
from aitoolbox.torchtrain.train_loop.components.callback_handler import CallbacksHandler
from aitoolbox.torchtrain.callbacks.abstract import AbstractCallback
from aitoolbox.torchtrain.train_loop import TrainLoop


class TestCallbacksHandler(unittest.TestCase):
    def test_callback_handler_has_hook_methods(self):
        callback_handler_1 = CallbacksHandler(None)
        self.check_callback_handler_for_hooks(callback_handler_1)

        train_loop = TrainLoop(NetUnifiedBatchFeed(), None, 100, None, None, None)
        callback_handler_2 = CallbacksHandler(train_loop)
        self.check_callback_handler_for_hooks(callback_handler_2)

    def check_callback_handler_for_hooks(self, callback_handler):
        self.assertTrue(function_exists(callback_handler, 'register_callbacks'))
        self.assertTrue(function_exists(callback_handler, 'execute_epoch_begin'))
        self.assertTrue(function_exists(callback_handler, 'execute_epoch_end'))
        self.assertTrue(function_exists(callback_handler, 'execute_train_begin'))
        self.assertTrue(function_exists(callback_handler, 'execute_train_end'))
        self.assertTrue(function_exists(callback_handler, 'execute_batch_begin'))
        self.assertTrue(function_exists(callback_handler, 'execute_batch_end'))
        self.assertTrue(function_exists(callback_handler, 'execute_gradient_update'))
        self.assertTrue(function_exists(callback_handler, 'execute_optimizer_step'))
        self.assertTrue(function_exists(callback_handler, 'execute_multiprocess_start'))
        self.assertTrue(function_exists(callback_handler, 'execute_after_batch_prediction'))

    def test_register_callbacks(self):
        train_loop = TrainLoop(NetUnifiedBatchFeed(), None, 100, None, None, None)
        cb_handler = CallbacksHandler(train_loop)

        batch_begin_cb = BatchBeginCB(execution_order=50)
        batch_begin_train_begin_cb = BatchBeginTrainBeginCB(execution_order=1)
        batch_begin_train_begin_after_opti_cb = BatchBeginTrainBeginAfterOptiCB(execution_order=0)
        callbacks = [batch_begin_cb, batch_begin_train_begin_cb, batch_begin_train_begin_after_opti_cb]

        cb_handler.register_callbacks(callbacks)

        self.assertEqual(train_loop.callbacks,
                         [batch_begin_train_begin_after_opti_cb, batch_begin_train_begin_cb, batch_begin_cb])

        self.assertEqual(
            cb_handler.registered_cbs,
            [
                [], [],
                [batch_begin_train_begin_after_opti_cb, batch_begin_train_begin_cb], [],
                [batch_begin_train_begin_after_opti_cb, batch_begin_train_begin_cb, batch_begin_cb], [],
                [], [batch_begin_train_begin_after_opti_cb], [], []
            ]
        )

        self.assertEqual(cb_handler.cbs_on_train_begin,
                         [batch_begin_train_begin_after_opti_cb, batch_begin_train_begin_cb])

        self.assertEqual(cb_handler.cbs_on_batch_begin,
                         [batch_begin_train_begin_after_opti_cb, batch_begin_train_begin_cb, batch_begin_cb])

        self.assertEqual(cb_handler.cbs_on_after_optimizer_step,
                         [batch_begin_train_begin_after_opti_cb])

        self.assertEqual(cb_handler.cbs_on_epoch_begin, [])
        self.assertEqual(cb_handler.cbs_on_epoch_end, [])

    def test_enforce_callbacks_quality(self):
        train_loop = TrainLoop(NetUnifiedBatchFeed(), None, None, None, None, None)
        cb_handler = train_loop.callbacks_handler
        # Workaround in order to test behaviour without the GPU
        train_loop.device = torch.device(f"cuda:0")

        callback_0 = AbstractCallback('dummy cb', device_idx_execution=0)
        with self.assertRaises(ValueError):
            cb_handler.enforce_callbacks_quality([callback_0])
        with self.assertRaises(ValueError):
            train_loop.callbacks_handler.register_callbacks([callback_0])

        callback_2 = AbstractCallback('dummy cb', device_idx_execution=2)
        with self.assertRaises(ValueError):
            cb_handler.enforce_callbacks_quality([callback_2])
        with self.assertRaises(ValueError):
            train_loop.callbacks_handler.register_callbacks([callback_2])

    def test_split_on_execution_position(self):
        train_loop = TrainLoop(NetUnifiedBatchFeed(), None, 100, None, None, None)
        cb_handler = CallbacksHandler(train_loop)

        batch_begin_cb = BatchBeginCB()
        batch_begin_train_begin_cb = BatchBeginTrainBeginCB()
        batch_begin_train_begin_after_opti_cb = BatchBeginTrainBeginAfterOptiCB()
        callbacks = [batch_begin_cb, batch_begin_train_begin_cb, batch_begin_train_begin_after_opti_cb]

        cb_handler.split_on_execution_position(callbacks)

        self.assertEqual(
            cb_handler.registered_cbs,
            [
                [], [],
                [batch_begin_train_begin_cb, batch_begin_train_begin_after_opti_cb], [],
                [batch_begin_cb, batch_begin_train_begin_cb, batch_begin_train_begin_after_opti_cb], [],
                [], [batch_begin_train_begin_after_opti_cb], [], []
            ]
        )

        self.assertEqual(cb_handler.cbs_on_train_begin,
                         [batch_begin_train_begin_cb, batch_begin_train_begin_after_opti_cb])

        self.assertEqual(cb_handler.cbs_on_batch_begin,
                         [batch_begin_cb, batch_begin_train_begin_cb, batch_begin_train_begin_after_opti_cb])

        self.assertEqual(cb_handler.cbs_on_after_optimizer_step,
                         [batch_begin_train_begin_after_opti_cb])

        self.assertEqual(cb_handler.cbs_on_epoch_begin, [])
        self.assertEqual(cb_handler.cbs_on_epoch_end, [])

    def test_split_on_execution_position_ordering(self):
        train_loop = TrainLoop(NetUnifiedBatchFeed(), None, 100, None, None, None)
        cb_handler = CallbacksHandler(train_loop)

        batch_begin_cb = BatchBeginCB(execution_order=50)
        batch_begin_train_begin_cb = BatchBeginTrainBeginCB(execution_order=1)
        batch_begin_train_begin_after_opti_cb = BatchBeginTrainBeginAfterOptiCB(execution_order=0)
        callbacks = [batch_begin_cb, batch_begin_train_begin_cb, batch_begin_train_begin_after_opti_cb]

        cb_handler.split_on_execution_position(callbacks)

        self.assertEqual(
            cb_handler.registered_cbs,
            [
                [], [],
                [batch_begin_train_begin_after_opti_cb, batch_begin_train_begin_cb], [],
                [batch_begin_train_begin_after_opti_cb, batch_begin_train_begin_cb, batch_begin_cb], [],
                [], [batch_begin_train_begin_after_opti_cb], [], []
            ]
        )

        self.assertEqual(cb_handler.cbs_on_train_begin,
                         [batch_begin_train_begin_after_opti_cb, batch_begin_train_begin_cb])

        self.assertEqual(cb_handler.cbs_on_batch_begin,
                         [batch_begin_train_begin_after_opti_cb, batch_begin_train_begin_cb, batch_begin_cb])

        self.assertEqual(cb_handler.cbs_on_after_optimizer_step,
                         [batch_begin_train_begin_after_opti_cb])

        self.assertEqual(cb_handler.cbs_on_epoch_begin, [])
        self.assertEqual(cb_handler.cbs_on_epoch_end, [])

    def test_split_on_execution_register_train_loop(self):
        train_loop = TrainLoop(NetUnifiedBatchFeed(), None, 100, None, None, None)
        cb_handler = CallbacksHandler(train_loop)

        batch_begin_cb = BatchBeginCB(execution_order=50)
        batch_begin_train_begin_cb = BatchBeginTrainBeginCB(execution_order=1)
        batch_begin_train_begin_after_opti_cb = BatchBeginTrainBeginAfterOptiCB(execution_order=0)
        callbacks = [batch_begin_cb, batch_begin_train_begin_cb, batch_begin_train_begin_after_opti_cb]

        cb_handler.split_on_execution_position(callbacks, register_train_loop=True)

        self.assertTrue(batch_begin_cb.registered_tl)

    def test_split_on_execution_execute_split_callbacks(self):
        train_loop = TrainLoop(NetUnifiedBatchFeed(), None, 100, None, None, None)
        cb_handler = CallbacksHandler(train_loop)

        batch_begin_cb = BatchBeginCB(execution_order=50)
        batch_begin_train_begin_cb = BatchBeginTrainBeginCB(execution_order=1)
        batch_begin_train_begin_after_opti_cb = BatchBeginTrainBeginAfterOptiCB(execution_order=0)
        callbacks = [batch_begin_cb, batch_begin_train_begin_cb, batch_begin_train_begin_after_opti_cb]

        cb_handler.split_on_execution_position(callbacks, register_train_loop=True)

        self.assertFalse(batch_begin_train_begin_after_opti_cb.exe_on_batch_begin)
        self.assertFalse(cb_handler.cbs_on_batch_begin[0].exe_on_batch_begin)
        self.assertFalse(batch_begin_train_begin_after_opti_cb.exe_on_train_begin)
        self.assertFalse(cb_handler.cbs_on_train_begin[0].exe_on_train_begin)
        self.assertFalse(batch_begin_train_begin_after_opti_cb.exe_on_after_optimizer_step)
        self.assertFalse(cb_handler.cbs_on_after_optimizer_step[0].exe_on_after_optimizer_step)

        cb_handler.execute_batch_begin()
        self.assertTrue(batch_begin_train_begin_after_opti_cb.exe_on_batch_begin)
        self.assertTrue(cb_handler.cbs_on_batch_begin[0].exe_on_batch_begin)
        self.assertFalse(batch_begin_train_begin_after_opti_cb.exe_on_train_begin)
        self.assertFalse(cb_handler.cbs_on_train_begin[0].exe_on_train_begin)
        self.assertFalse(batch_begin_train_begin_after_opti_cb.exe_on_after_optimizer_step)
        self.assertFalse(cb_handler.cbs_on_after_optimizer_step[0].exe_on_after_optimizer_step)

        cb_handler.execute_train_begin()
        self.assertTrue(batch_begin_train_begin_after_opti_cb.exe_on_batch_begin)
        self.assertTrue(cb_handler.cbs_on_batch_begin[0].exe_on_batch_begin)
        self.assertTrue(batch_begin_train_begin_after_opti_cb.exe_on_train_begin)
        self.assertTrue(cb_handler.cbs_on_train_begin[0].exe_on_train_begin)
        self.assertFalse(batch_begin_train_begin_after_opti_cb.exe_on_after_optimizer_step)
        self.assertFalse(cb_handler.cbs_on_after_optimizer_step[0].exe_on_after_optimizer_step)

        cb_handler.execute_optimizer_step()
        self.assertTrue(batch_begin_train_begin_after_opti_cb.exe_on_batch_begin)
        self.assertTrue(cb_handler.cbs_on_batch_begin[0].exe_on_batch_begin)
        self.assertTrue(batch_begin_train_begin_after_opti_cb.exe_on_train_begin)
        self.assertTrue(cb_handler.cbs_on_train_begin[0].exe_on_train_begin)
        self.assertTrue(batch_begin_train_begin_after_opti_cb.exe_on_after_optimizer_step)
        self.assertTrue(cb_handler.cbs_on_after_optimizer_step[0].exe_on_after_optimizer_step)

    def test_handler_cache_empty_callbacks(self):
        train_loop = TrainLoop(NetUnifiedBatchFeed(), None, 100, None, None, None)
        cb_handler = CallbacksHandler(train_loop)
        self.assertEqual(cb_handler.callbacks_cache, [])

        cb_handler.register_callbacks([], cache_callbacks=True)
        self.assertEqual(train_loop.callbacks, [])
        self.assertEqual(cb_handler.callbacks_cache, [])

        cb_handler.register_callbacks(None, cache_callbacks=True)
        self.assertEqual(train_loop.callbacks, [])
        self.assertEqual(cb_handler.callbacks_cache, [])

    def test_handler_cache_callbacks(self):
        train_loop = TrainLoop(NetUnifiedBatchFeed(), None, 100, None, None, None)
        cb_handler = CallbacksHandler(train_loop)
        self.assertEqual(cb_handler.callbacks_cache, [])

        batch_begin_cb = BatchBeginCB(execution_order=50)
        batch_begin_train_begin_cb = BatchBeginTrainBeginCB(execution_order=1)
        batch_begin_train_begin_after_opti_cb = BatchBeginTrainBeginAfterOptiCB(execution_order=0)
        callbacks = [batch_begin_cb, batch_begin_train_begin_cb, batch_begin_train_begin_after_opti_cb]

        cb_handler.register_callbacks(callbacks, cache_callbacks=True)
        self.assertEqual(train_loop.callbacks, [])
        self.assertEqual(cb_handler.registered_cbs, [[], [], [], [], [], [], [], [], [], []])
        self.assertEqual(cb_handler.callbacks_cache, callbacks)
        for cb in callbacks:
            self.assertIsNone(cb.train_loop_obj)
        for cb in cb_handler.callbacks_cache:
            self.assertIsNone(cb.train_loop_obj)

        cb_handler.register_callbacks([], cache_callbacks=False)
        self.assertEqual(train_loop.callbacks,
                         [batch_begin_train_begin_after_opti_cb, batch_begin_train_begin_cb, batch_begin_cb])
        self.assertEqual(
            cb_handler.registered_cbs,
            [[], [], [batch_begin_train_begin_after_opti_cb, batch_begin_train_begin_cb], [],
             [batch_begin_train_begin_after_opti_cb, batch_begin_train_begin_cb, batch_begin_cb], [], [],
             [batch_begin_train_begin_after_opti_cb], [], []]
        )
        self.assertEqual(cb_handler.callbacks_cache, [])

    def test_handler_cache_callbacks_further_add(self):
        train_loop = TrainLoop(NetUnifiedBatchFeed(), None, 100, None, None, None)
        cb_handler = CallbacksHandler(train_loop)
        self.assertEqual(cb_handler.callbacks_cache, [])

        batch_begin_cb = BatchBeginCB(execution_order=50)
        batch_begin_train_begin_cb = BatchBeginTrainBeginCB(execution_order=1)
        batch_begin_train_begin_after_opti_cb = BatchBeginTrainBeginAfterOptiCB(execution_order=0)
        callbacks = [batch_begin_cb, batch_begin_train_begin_cb, batch_begin_train_begin_after_opti_cb]

        cb_handler.register_callbacks(callbacks, cache_callbacks=True)
        self.assertEqual(train_loop.callbacks, [])
        self.assertEqual(cb_handler.registered_cbs, [[], [], [], [], [], [], [], [], [], []])
        self.assertEqual(cb_handler.callbacks_cache, callbacks)
        for cb in callbacks:
            self.assertIsNone(cb.train_loop_obj)
        for cb in cb_handler.callbacks_cache:
            self.assertIsNone(cb.train_loop_obj)

        batch_begin_cb_add = BatchBeginCB(execution_order=12)
        cb_handler.register_callbacks([batch_begin_cb_add], cache_callbacks=False)
        self.assertEqual(train_loop.callbacks,
                         [batch_begin_train_begin_after_opti_cb, batch_begin_train_begin_cb,
                          batch_begin_cb_add, batch_begin_cb])
        self.assertEqual(
            cb_handler.registered_cbs,
            [[], [], [batch_begin_train_begin_after_opti_cb, batch_begin_train_begin_cb], [],
             [batch_begin_train_begin_after_opti_cb, batch_begin_train_begin_cb, batch_begin_cb_add, batch_begin_cb],
             [], [],
             [batch_begin_train_begin_after_opti_cb], [], []]
        )
        self.assertEqual(cb_handler.callbacks_cache, [])

        batch_begin_train_begin_cb_add = BatchBeginTrainBeginCB(execution_order=1000)
        cb_handler.register_callbacks([batch_begin_train_begin_cb_add], cache_callbacks=False)
        self.assertEqual(train_loop.callbacks,
                         [batch_begin_train_begin_after_opti_cb, batch_begin_train_begin_cb,
                          batch_begin_cb_add, batch_begin_cb, batch_begin_train_begin_cb_add])
        self.assertEqual(
            cb_handler.registered_cbs,
            [[], [],
             [batch_begin_train_begin_after_opti_cb, batch_begin_train_begin_cb, batch_begin_train_begin_cb_add], [],
             [batch_begin_train_begin_after_opti_cb, batch_begin_train_begin_cb, batch_begin_cb_add, batch_begin_cb,
              batch_begin_train_begin_cb_add], [], [],
             [batch_begin_train_begin_after_opti_cb], [], []]
        )
        self.assertEqual(cb_handler.callbacks_cache, [])

    def test_handler_execution_after_batch_prediction(self):
        self.execute_training_with_on_batch_prediction_cb(
            num_epochs=5, train_loader=list(range(4)), val_loader=list(range(3)), test_loader=None
        )
        self.execute_training_with_on_batch_prediction_cb(
            num_epochs=30, train_loader=list(range(4)), val_loader=list(range(3)), test_loader=None
        )
        self.execute_training_with_on_batch_prediction_cb(
            num_epochs=10, train_loader=list(range(4)), val_loader=list(range(3)), test_loader=list(range(2))
        )
        self.execute_training_with_on_batch_prediction_cb(
            num_epochs=30, train_loader=list(range(20)), val_loader=list(range(8)), test_loader=list(range(5))
        )

    def execute_training_with_on_batch_prediction_cb(self, num_epochs, train_loader, val_loader, test_loader):
        train_loader_size = len(train_loader) if train_loader is not None else 0
        val_loader_size = len(val_loader) if val_loader is not None else 0
        test_loader_size = len(test_loader) if test_loader is not None else 0

        dummy_optimizer = DummyOptimizer()
        dummy_loss = DummyLoss()
        train_loop = TrainLoop(
            NetUnifiedBatchFeed(),
            train_loader, val_loader, test_loader,
            dummy_optimizer, dummy_loss
        )

        callback = AfterBatchPredictionCB(execute_callbacks=True)
        train_loop.fit(num_epochs=num_epochs, callbacks=[callback])

        self.assertEqual(callback.cb_execution_ctr,
                         num_epochs * (train_loader_size + val_loader_size + test_loader_size))
        self.assertEqual(
            callback.cb_execution_ctr_dict,
            {'train': num_epochs * train_loader_size,
             'validation': num_epochs * val_loader_size,
             'test': num_epochs * test_loader_size}
        )

    def test_handler_disable_execution_after_batch_prediction(self):
        dummy_optimizer = DummyOptimizer()
        dummy_loss = DummyLoss()
        dummy_train_loader = list(range(4))
        dummy_val_loader = list(range(3))
        train_loop = TrainLoop(
            NetUnifiedBatchFeed(),
            dummy_train_loader, dummy_val_loader, None,
            dummy_optimizer, dummy_loss
        )

        callback = AfterBatchPredictionCB(execute_callbacks=False)
        train_loop.fit(num_epochs=5, callbacks=[callback])

        self.assertEqual(callback.cb_execution_ctr, 0)
        self.assertEqual(callback.cb_execution_ctr_dict, {'train': 0, 'validation': 0, 'test': 0})


class BatchBeginCB(AbstractCallback):
    def __init__(self, execution_order=0):
        super().__init__('', execution_order)
        self.registered_tl = False

    def on_train_loop_registration(self):
        self.registered_tl = True

    def on_batch_begin(self):
        print("executed")


class BatchBeginTrainBeginCB(AbstractCallback):
    def __init__(self, execution_order=0):
        super().__init__('', execution_order)

    def on_batch_begin(self):
        print("executed")

    def on_train_begin(self):
        print("executed")


class BatchBeginTrainBeginAfterOptiCB(AbstractCallback):
    def __init__(self, execution_order=0):
        super().__init__('', execution_order)
        self.exe_on_batch_begin = False
        self.exe_on_train_begin = False
        self.exe_on_after_optimizer_step = False

    def on_batch_begin(self):
        self.exe_on_batch_begin = True

    def on_train_begin(self):
        self.exe_on_train_begin = True

    def on_after_optimizer_step(self):
        self.exe_on_after_optimizer_step = True


class AfterBatchPredictionCB(AbstractCallback):
    def __init__(self, execute_callbacks, execution_order=0):
        super().__init__('', execution_order)
        self.execute_callbacks = execute_callbacks
        self.cb_execution_ctr = 0
        self.cb_execution_ctr_dict = {'train': 0, 'validation': 0, 'test': 0}

    def on_epoch_end(self):
        self.train_loop_obj.predict_on_train_set(execute_callbacks=self.execute_callbacks)
        self.train_loop_obj.predict_on_validation_set(execute_callbacks=self.execute_callbacks)

        if self.train_loop_obj.test_loader is not None:
            self.train_loop_obj.predict_on_test_set(execute_callbacks=self.execute_callbacks)

    def on_after_batch_prediction(self, y_pred_batch, y_test_batch, metadata_batch, dataset_info):
        self.cb_execution_ctr += 1

        dataset_type = dataset_info['type']
        self.cb_execution_ctr_dict[dataset_type] += 1
