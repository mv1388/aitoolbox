import unittest

from tests.utils import *
from aitoolbox.torchtrain.train_loop.components.callback_handler import CallbacksHandler, BasicCallbacksHandler
from aitoolbox.torchtrain.callbacks.abstract import AbstractCallback
from aitoolbox.torchtrain.train_loop import TrainLoop


class TestCallbacksHandler(unittest.TestCase):
    def test_basic_callback_handler_has_hook_methods(self):
        callback_handler_1 = BasicCallbacksHandler(None)
        self.check_callback_handler_for_hooks(callback_handler_1)

        train_loop = TrainLoop(NetUnifiedBatchFeed(), None, 100, None, None, None)
        callback_handler_2 = BasicCallbacksHandler(train_loop)
        self.check_callback_handler_for_hooks(callback_handler_2)

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
                [], [batch_begin_train_begin_after_opti_cb], []
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
                [], [batch_begin_train_begin_after_opti_cb], []
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
                [], [batch_begin_train_begin_after_opti_cb], []
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

    def test_basic_handler_register_cb_ordering(self):
        train_loop = TrainLoop(NetUnifiedBatchFeed(), None, 100, None, None, None)
        cb_handler = BasicCallbacksHandler(train_loop)

        batch_begin_cb = BatchBeginCB(execution_order=50)
        batch_begin_train_begin_cb = BatchBeginTrainBeginCB(execution_order=1)
        batch_begin_train_begin_after_opti_cb = BatchBeginTrainBeginAfterOptiCB(execution_order=0)
        callbacks = [batch_begin_cb, batch_begin_train_begin_cb, batch_begin_train_begin_after_opti_cb]

        self.assertFalse(batch_begin_cb.registered_tl)

        cb_handler.register_callbacks(callbacks)

        self.assertEqual(
            train_loop.callbacks,
            [batch_begin_train_begin_after_opti_cb, batch_begin_train_begin_cb, batch_begin_cb]
        )
        self.assertEqual(
            cb_handler.train_loop_obj.callbacks,
            [batch_begin_train_begin_after_opti_cb, batch_begin_train_begin_cb, batch_begin_cb]
        )
        self.assertTrue(batch_begin_cb.registered_tl)

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

    def test_basic_handler_cache_empty_callbacks(self):
        train_loop = TrainLoop(NetUnifiedBatchFeed(), None, 100, None, None, None)
        cb_handler = BasicCallbacksHandler(train_loop)
        self.assertEqual(cb_handler.callbacks_cache, [])

        cb_handler.register_callbacks([], cache_callbacks=True)
        self.assertEqual(train_loop.callbacks, [])
        self.assertEqual(cb_handler.callbacks_cache, [])

        cb_handler.register_callbacks(None, cache_callbacks=True)
        self.assertEqual(train_loop.callbacks, [])
        self.assertEqual(cb_handler.callbacks_cache, [])

    def test_basic_handler_cache_callbacks(self):
        train_loop = TrainLoop(NetUnifiedBatchFeed(), None, 100, None, None, None)
        cb_handler = BasicCallbacksHandler(train_loop)
        self.assertEqual(cb_handler.callbacks_cache, [])

        batch_begin_cb = BatchBeginCB(execution_order=50)
        batch_begin_train_begin_cb = BatchBeginTrainBeginCB(execution_order=1)
        batch_begin_train_begin_after_opti_cb = BatchBeginTrainBeginAfterOptiCB(execution_order=0)
        callbacks = [batch_begin_cb, batch_begin_train_begin_cb, batch_begin_train_begin_after_opti_cb]

        cb_handler.register_callbacks(callbacks, cache_callbacks=True)
        self.assertEqual(train_loop.callbacks, [])
        self.assertEqual(cb_handler.callbacks_cache, callbacks)
        for cb in callbacks:
            self.assertIsNone(cb.train_loop_obj)
        for cb in cb_handler.callbacks_cache:
            self.assertIsNone(cb.train_loop_obj)

        cb_handler.register_callbacks([], cache_callbacks=False)
        self.assertEqual(train_loop.callbacks,
                         [batch_begin_train_begin_after_opti_cb, batch_begin_train_begin_cb, batch_begin_cb])

    def test_basic_handler_cache_callbacks_further_add(self):
        train_loop = TrainLoop(NetUnifiedBatchFeed(), None, 100, None, None, None)
        cb_handler = BasicCallbacksHandler(train_loop)
        self.assertEqual(cb_handler.callbacks_cache, [])

        batch_begin_cb = BatchBeginCB(execution_order=50)
        batch_begin_train_begin_cb = BatchBeginTrainBeginCB(execution_order=1)
        batch_begin_train_begin_after_opti_cb = BatchBeginTrainBeginAfterOptiCB(execution_order=0)
        callbacks = [batch_begin_cb, batch_begin_train_begin_cb, batch_begin_train_begin_after_opti_cb]

        cb_handler.register_callbacks(callbacks, cache_callbacks=True)
        self.assertEqual(train_loop.callbacks, [])
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
        self.assertEqual(cb_handler.callbacks_cache, [])

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
        self.assertEqual(cb_handler.registered_cbs, [[], [], [], [], [], [], [], [], []])
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
             [batch_begin_train_begin_after_opti_cb], []]
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
        self.assertEqual(cb_handler.registered_cbs, [[], [], [], [], [], [], [], [], []])
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
             [batch_begin_train_begin_after_opti_cb], []]
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
             [batch_begin_train_begin_after_opti_cb], []]
        )
        self.assertEqual(cb_handler.callbacks_cache, [])


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
