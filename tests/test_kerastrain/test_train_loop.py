import unittest
import keras

from tests.utils import *

from AIToolbox.kerastrain.train_loop import TrainLoop, TrainLoopModelCheckpoint, TrainLoopModelEndSave, TrainLoopModelCheckpointEndSave
from AIToolbox.kerastrain.callbacks.callback_handler import CallbacksHandler
from AIToolbox.kerastrain.callbacks.callbacks import AbstractKerasCallback, ModelCheckpointCallback, ModelTrainEndSaveCallback


class KerasCallbackTracker(AbstractKerasCallback):
    def __init__(self):
        AbstractKerasCallback.__init__(self, 'CallbackTracker1')
        self.callback_calls = []
        self.call_ctr = {'on_train_loop_registration': 0, 'on_epoch_begin': 0, 'on_epoch_end': 0, 'on_train_begin': 0,
                         'on_train_end': 0, 'on_batch_begin': 0, 'on_batch_end': 0}

    def on_train_loop_registration(self):
        self.callback_calls.append('on_train_loop_registration')
        self.call_ctr['on_train_loop_registration'] += 1

    def on_epoch_begin(self, epoch, logs=None):
        self.callback_calls.append('on_epoch_begin')
        self.call_ctr['on_epoch_begin'] += 1

    def on_epoch_end(self, epoch, logs=None):
        self.callback_calls.append('on_epoch_end')
        self.call_ctr['on_epoch_end'] += 1

    def on_train_begin(self, logs=None):
        self.callback_calls.append('on_train_begin')
        self.call_ctr['on_train_begin'] += 1

    def on_train_end(self, logs=None):
        self.callback_calls.append('on_train_end')
        self.call_ctr['on_train_end'] += 1

    def on_batch_begin(self, epoch, logs=None):
        self.callback_calls.append('on_batch_begin')
        self.call_ctr['on_batch_begin'] += 1

    def on_batch_end(self, epoch, logs=None):
        self.callback_calls.append('on_batch_end')
        self.call_ctr['on_batch_end'] += 1


class KerasCallbackTrackerShort(AbstractKerasCallback):
    def __init__(self):
        AbstractKerasCallback.__init__(self, 'CallbackTracker2')
        self.callback_calls = []
        self.call_ctr = {'on_train_loop_registration': 0, 'on_epoch_begin': 0, 'on_epoch_end': 0, 'on_train_begin': 0,
                         'on_train_end': 0, 'on_batch_begin': 0, 'on_batch_end': 0}

    def on_epoch_begin(self, epoch, logs=None):
        self.callback_calls.append('on_epoch_begin')
        self.call_ctr['on_epoch_begin'] += 1

    def on_epoch_end(self, epoch, logs=None):
        self.callback_calls.append('on_epoch_end')
        self.call_ctr['on_epoch_end'] += 1

    def on_train_end(self, logs=None):
        self.callback_calls.append('on_train_end')
        self.call_ctr['on_train_end'] += 1

    def on_batch_begin(self, epoch, logs=None):
        self.callback_calls.append('on_batch_begin')
        self.call_ctr['on_batch_begin'] += 1


class TestTrainLoop(unittest.TestCase):
    def test_init_values(self):
        train_loop = TrainLoop(keras_dummy_model(), None, None, None,
                               optimizer=keras.optimizers.rmsprop(lr=0.0001, decay=1e-6), 
                               criterion='categorical_crossentropy', metrics=['accuracy'])

        self.assertEqual(train_loop.callbacks, [])
        self.assertIsInstance(train_loop.callbacks_handler, CallbacksHandler)
        self.assertEqual(train_loop.callbacks_handler.train_loop_obj, train_loop)

    def test_callback_registration(self):
        train_loop = TrainLoop(keras_dummy_model(), None, None, None,
                               optimizer=keras.optimizers.rmsprop(lr=0.0001, decay=1e-6),
                               criterion='categorical_crossentropy', metrics=['accuracy'])
        train_loop.callbacks_handler.register_callbacks([AbstractKerasCallback('callback_test1'),
                                                         KerasCallbackTracker(), KerasCallbackTrackerShort()])

        self.assertEqual(len(train_loop.callbacks), 3)
        for reg_cb, true_cb in zip(train_loop.callbacks, [AbstractKerasCallback, KerasCallbackTracker, KerasCallbackTrackerShort]):
            self.assertEqual(type(reg_cb), true_cb)
        for reg_cb in train_loop.callbacks:
            self.assertEqual(reg_cb.train_loop_obj, train_loop)

        train_loop.callbacks_handler.register_callbacks([AbstractKerasCallback('callback_test2')])
        self.assertEqual(len(train_loop.callbacks), 4)
        for reg_cb, true_cb in zip(train_loop.callbacks, [AbstractKerasCallback, KerasCallbackTracker, KerasCallbackTrackerShort, AbstractKerasCallback]):
            self.assertEqual(type(reg_cb), true_cb)

        for reg_cb in train_loop.callbacks:
            self.assertEqual(reg_cb.train_loop_obj, train_loop)

        for reg_cb, cb_name in zip(train_loop.callbacks,
                                   ['callback_test1', 'CallbackTracker1', 'CallbackTracker2', 'callback_test2']):
            self.assertEqual(reg_cb.callback_name, cb_name)
