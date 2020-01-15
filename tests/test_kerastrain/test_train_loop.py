import unittest
import keras
import numpy
import os

from tests.utils import *

# from aitoolbox.kerastrain.train_loop import TrainLoop, TrainLoopModelCheckpoint, TrainLoopModelEndSave, TrainLoopModelCheckpointEndSave
# from aitoolbox.kerastrain.callbacks.callback_handler import CallbacksHandler
# from aitoolbox.kerastrain.callbacks.callbacks import AbstractKerasCallback, ModelCheckpoint, ModelTrainEndSave
#
#
# THIS_DIR = os.path.dirname(os.path.abspath(__file__))
#
#
# class KerasCallbackTracker(AbstractKerasCallback):
#     def __init__(self):
#         AbstractKerasCallback.__init__(self, 'CallbackTracker1')
#         self.callback_calls = []
#         self.call_ctr = {'on_train_loop_registration': 0, 'on_epoch_begin': 0, 'on_epoch_end': 0, 'on_train_begin': 0,
#                          'on_train_end': 0, 'on_batch_begin': 0, 'on_batch_end': 0}
#
#     def on_train_loop_registration(self):
#         self.callback_calls.append('on_train_loop_registration')
#         self.call_ctr['on_train_loop_registration'] += 1
#
#     def on_epoch_begin(self, epoch, logs=None):
#         self.callback_calls.append('on_epoch_begin')
#         self.call_ctr['on_epoch_begin'] += 1
#
#     def on_epoch_end(self, epoch, logs=None):
#         self.callback_calls.append('on_epoch_end')
#         self.call_ctr['on_epoch_end'] += 1
#
#     def on_train_begin(self, logs=None):
#         self.callback_calls.append('on_train_begin')
#         self.call_ctr['on_train_begin'] += 1
#
#     def on_train_end(self, logs=None):
#         self.callback_calls.append('on_train_end')
#         self.call_ctr['on_train_end'] += 1
#
#     def on_batch_begin(self, epoch, logs=None):
#         self.callback_calls.append('on_batch_begin')
#         self.call_ctr['on_batch_begin'] += 1
#
#     def on_batch_end(self, epoch, logs=None):
#         self.callback_calls.append('on_batch_end')
#         self.call_ctr['on_batch_end'] += 1
#
#
# class KerasCallbackTrackerShort(AbstractKerasCallback):
#     def __init__(self):
#         AbstractKerasCallback.__init__(self, 'CallbackTracker2')
#         self.callback_calls = []
#         self.call_ctr = {'on_train_loop_registration': 0, 'on_epoch_begin': 0, 'on_epoch_end': 0, 'on_train_begin': 0,
#                          'on_train_end': 0, 'on_batch_begin': 0, 'on_batch_end': 0}
#
#     def on_epoch_begin(self, epoch, logs=None):
#         self.callback_calls.append('on_epoch_begin')
#         self.call_ctr['on_epoch_begin'] += 1
#
#     def on_epoch_end(self, epoch, logs=None):
#         self.callback_calls.append('on_epoch_end')
#         self.call_ctr['on_epoch_end'] += 1
#
#     def on_train_end(self, logs=None):
#         self.callback_calls.append('on_train_end')
#         self.call_ctr['on_train_end'] += 1
#
#     def on_batch_begin(self, epoch, logs=None):
#         self.callback_calls.append('on_batch_begin')
#         self.call_ctr['on_batch_begin'] += 1
#
#
# class TestTrainLoop(unittest.TestCase):
#     def test_init_values(self):
#         train_loop = TrainLoop(keras_dummy_model(), None, None, None,
#                                optimizer=keras.optimizers.rmsprop(lr=0.0001, decay=1e-6),
#                                criterion='categorical_crossentropy', metrics=['accuracy'])
#
#         self.assertEqual(train_loop.callbacks, [])
#         self.assertIsInstance(train_loop.callbacks_handler, CallbacksHandler)
#         self.assertEqual(train_loop.callbacks_handler.train_loop_obj, train_loop)
#
#     def test_callback_registration(self):
#         train_loop = TrainLoop(keras_dummy_model(), None, None, None,
#                                optimizer=keras.optimizers.rmsprop(lr=0.0001, decay=1e-6),
#                                criterion='categorical_crossentropy', metrics=['accuracy'])
#         train_loop.callbacks_handler.register_callbacks([AbstractKerasCallback('callback_test1'),
#                                                          KerasCallbackTracker(), KerasCallbackTrackerShort()])
#
#         self.assertEqual(len(train_loop.callbacks), 3)
#         for reg_cb, true_cb in zip(train_loop.callbacks, [AbstractKerasCallback, KerasCallbackTracker, KerasCallbackTrackerShort]):
#             self.assertEqual(type(reg_cb), true_cb)
#         for reg_cb in train_loop.callbacks:
#             self.assertEqual(reg_cb.train_loop_obj, train_loop)
#
#         train_loop.callbacks_handler.register_callbacks([AbstractKerasCallback('callback_test2')])
#         self.assertEqual(len(train_loop.callbacks), 4)
#         for reg_cb, true_cb in zip(train_loop.callbacks, [AbstractKerasCallback, KerasCallbackTracker, KerasCallbackTrackerShort, AbstractKerasCallback]):
#             self.assertEqual(type(reg_cb), true_cb)
#
#         for reg_cb in train_loop.callbacks:
#             self.assertEqual(reg_cb.train_loop_obj, train_loop)
#
#         for reg_cb, cb_name in zip(train_loop.callbacks,
#                                    ['callback_test1', 'CallbackTracker1', 'CallbackTracker2', 'callback_test2']):
#             self.assertEqual(reg_cb.callback_name, cb_name)
#
#     def test_callback_on_execution(self):
#         num_epochs = 2
#         model = keras_dummy_model()
#
#         # Based on the example from: https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
#         dataset = numpy.loadtxt(os.path.join(THIS_DIR, "data.csv"), delimiter=",")
#         # split into input (X) and output (Y) variables
#         X = dataset[:, 0:8]
#         Y = dataset[:, 8]
#
#         train_loop = TrainLoop(model, [X, Y], None, None,
#                                optimizer='adam',
#                                criterion='binary_crossentropy', metrics=['accuracy'])
#         train_loop.callbacks_handler.register_callbacks([AbstractKerasCallback('callback_test1'),
#                                                          KerasCallbackTracker(), KerasCallbackTrackerShort(),
#                                                          AbstractKerasCallback('callback_test2')])
#
#         callback_full = train_loop.callbacks[1]
#         callback_short = train_loop.callbacks[2]
#
#         model_return = train_loop.fit(num_epochs=num_epochs, batch_size=300)
#
#         self.assertEqual(model, model_return)
#
#         self.assertEqual(callback_full.callback_calls,
#                          ['on_train_loop_registration', 'on_train_begin', 'on_epoch_begin', 'on_batch_begin',
#                           'on_batch_end', 'on_batch_begin', 'on_batch_end', 'on_batch_begin', 'on_batch_end',
#                           'on_epoch_end', 'on_epoch_begin', 'on_batch_begin', 'on_batch_end', 'on_batch_begin',
#                           'on_batch_end', 'on_batch_begin', 'on_batch_end', 'on_epoch_end', 'on_train_end'])
#         self.assertEqual(callback_full.call_ctr,
#                          {'on_train_loop_registration': 1, 'on_epoch_begin': 2, 'on_epoch_end': 2, 'on_train_begin': 1,
#                           'on_train_end': 1, 'on_batch_begin': 6, 'on_batch_end': 6})
#
#         self.assertEqual(callback_short.callback_calls,
#                          ['on_epoch_begin', 'on_batch_begin', 'on_batch_begin', 'on_batch_begin', 'on_epoch_end',
#                           'on_epoch_begin', 'on_batch_begin', 'on_batch_begin', 'on_batch_begin', 'on_epoch_end',
#                           'on_train_end'])
#         self.assertEqual(callback_short.call_ctr,
#                          {'on_train_loop_registration': 0, 'on_epoch_begin': 2, 'on_epoch_end': 2, 'on_train_begin': 0,
#                           'on_train_end': 1, 'on_batch_begin': 6, 'on_batch_end': 0})
#
#     def test_basic_history_tracking(self):
#         num_epochs = 2
#         model = keras_dummy_model()
#
#         # Based on the example from: https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
#         dataset = numpy.loadtxt(os.path.join(THIS_DIR, "data.csv"), delimiter=",")
#         # split into input (X) and output (Y) variables
#         X = dataset[:, 0:8]
#         Y = dataset[:, 8]
#
#         train_loop = TrainLoop(model, [X, Y], None, None,
#                                optimizer='adam',
#                                criterion='binary_crossentropy', metrics=['accuracy'])
#         train_loop.callbacks_handler.register_callbacks([AbstractKerasCallback('callback_test1'),
#                                                          KerasCallbackTracker(), KerasCallbackTrackerShort(),
#                                                          AbstractKerasCallback('callback_test2')])
#
#         train_loop.fit(num_epochs=num_epochs, batch_size=300)
#         train_history = train_loop.train_history
#
#         self.assertEqual(train_history.epoch, [0, 1])
#         self.assertEqual(len(train_history.history), 2)
#         self.assertEqual(sorted(train_history.history.keys()), sorted(['loss', 'accuracy']))
#         self.assertEqual(len(train_history.history['loss']), 2)
#         self.assertEqual(len(train_history.history['accuracy']), 2)
#         self.assertEqual(train_history.params,
#                          {'batch_size': 300, 'epochs': 2, 'steps': None, 'samples': 768, 'verbose': 1,
#                           'do_validation': False, 'metrics': ['loss', 'accuracy']})
