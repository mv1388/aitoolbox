import unittest
import torch
from torch.utils.data.dataset import TensorDataset
from torch.utils.data.dataloader import DataLoader

from tests.utils import *

from aitoolbox.torchtrain.callbacks.basic import EarlyStopping, DataSubsetTestRun, FunctionOnTrainLoop
from aitoolbox.torchtrain.train_loop import TrainLoop


class TestEarlyStoppingCallback(unittest.TestCase):
    def test_basic_no_loss_change(self):
        self.basic_early_stop_change_check_loss(10., 10., [False, True])
        self.basic_early_stop_change_check_loss(22232.334, 22232.334, [False, True])

    def test_basic_no_acc_change(self):
        self.basic_early_stop_change_check_acc(10., 10., [False, True])
        self.basic_early_stop_change_check_acc(22232.334, 22232.334, [False, True])

    def test_basic_loss_drops(self):
        self.basic_early_stop_change_check_loss(10., 9.9, [False, False])
        self.basic_early_stop_change_check_loss(10223., 33.9, [False, False])

    def test_basic_acc_drops(self):
        self.basic_early_stop_change_check_acc(10., 9.9, [False, True])
        self.basic_early_stop_change_check_acc(10223., 33.9, [False, True])

    def test_basic_loss_grows(self):
        self.basic_early_stop_change_check_loss(10., 11.22, [False, True])
        self.basic_early_stop_change_check_loss(1., 11323.22, [False, True])

    def test_basic_acc_grows(self):
        self.basic_early_stop_change_check_acc(10., 11.22, [False, False])
        self.basic_early_stop_change_check_acc(1., 11323.22, [False, False])

    def test_delta_loss(self):
        self.basic_early_stop_change_check_loss(10., 8.1, [False, True], min_delta=2.)
        self.basic_early_stop_change_check_loss(10., 8.0, [False, True], min_delta=2.)
        self.basic_early_stop_change_check_loss(10., 7.9, [False, False], min_delta=2.)
        self.basic_early_stop_change_check_loss(10., 11.9, [False, True], min_delta=2.)

    def test_delta_acc(self):
        self.basic_early_stop_change_check_acc(10., 12.1, [False, False], min_delta=2.)
        self.basic_early_stop_change_check_acc(10., 12.0, [False, True], min_delta=2.)
        self.basic_early_stop_change_check_acc(10., 11.9, [False, True], min_delta=2.)
        self.basic_early_stop_change_check_acc(10., 9.5, [False, True], min_delta=2.)

    def basic_early_stop_change_check_loss(self, val1, val2, expected_result, min_delta=0.):
        callback = EarlyStopping(monitor='dummy_loss', min_delta=min_delta)
        train_loop = TrainLoop(NetUnifiedBatchFeed(), None, None, None, None, None)
        train_loop.callbacks_handler.register_callbacks([callback])
        self.assertFalse(train_loop.early_stop)

        result = []

        train_loop.insert_metric_result_into_history('dummy_loss', val1)
        callback.on_epoch_end()
        result.append(train_loop.early_stop)

        train_loop.insert_metric_result_into_history('dummy_loss', val2)
        callback.on_epoch_end()
        result.append(train_loop.early_stop)

        self.assertEqual(result, expected_result)

    def basic_early_stop_change_check_acc(self, val1, val2, expected_result, min_delta=0.):
        callback = EarlyStopping(monitor='dummy_acc', min_delta=min_delta)
        train_loop = TrainLoop(NetUnifiedBatchFeed(), None, None, None, None, None)
        train_loop.callbacks_handler.register_callbacks([callback])
        self.assertFalse(train_loop.early_stop)

        result = []

        train_loop.insert_metric_result_into_history('dummy_acc', val1)
        callback.on_epoch_end()
        result.append(train_loop.early_stop)

        train_loop.insert_metric_result_into_history('dummy_acc', val2)
        callback.on_epoch_end()
        result.append(train_loop.early_stop)

        self.assertEqual(result, expected_result)

    def test_patience_loss(self):
        self.eval_patience(min_delta=0., patience=0,
                           val_list=[10., 10., 10.], expected_result=[False, True, True], monitor='dummy_loss')
        self.eval_patience(min_delta=0., patience=1,
                           val_list=[10., 10., 10.], expected_result=[False, False, True], monitor='dummy_loss')
        self.eval_patience(min_delta=0., patience=2,
                           val_list=[10., 10., 10., 10.], expected_result=[False, False, False, True], monitor='dummy_loss')

        self.eval_patience(min_delta=1., patience=1,
                           val_list=[10., 9., 8.], expected_result=[False, False, False], monitor='dummy_loss')
        self.eval_patience(min_delta=1., patience=1,
                           val_list=[10., 9.4, 8.9], expected_result=[False, False, False], monitor='dummy_loss')
        self.eval_patience(min_delta=1., patience=1,
                           val_list=[10., 9.4, 9.], expected_result=[False, False, True], monitor='dummy_loss')
        self.eval_patience(min_delta=1., patience=2,
                           val_list=[10., 9.4, 9.], expected_result=[False, False, False], monitor='dummy_loss')
        self.eval_patience(min_delta=1., patience=2,
                           val_list=[10., 9.4, 9., 9.], expected_result=[False, False, False, True], monitor='dummy_loss')
        self.eval_patience(min_delta=1., patience=2,
                           val_list=[10., 9.4, 9., 8.9], expected_result=[False, False, False, False], monitor='dummy_loss')

    def test_patience_acc(self):
        self.eval_patience(min_delta=0., patience=0,
                           val_list=[10., 10., 10.], expected_result=[False, True, True], monitor='dummy_acc')
        self.eval_patience(min_delta=0., patience=1,
                           val_list=[10., 10., 10.], expected_result=[False, False, True], monitor='dummy_acc')
        self.eval_patience(min_delta=0., patience=2,
                           val_list=[10., 10., 10., 10.], expected_result=[False, False, False, True], monitor='dummy_acc')

        self.eval_patience(min_delta=1., patience=1,
                           val_list=[8., 9., 10.], expected_result=[False, False, False], monitor='dummy_acc')
        self.eval_patience(min_delta=1., patience=1,
                           val_list=[8.9, 9.4, 10.], expected_result=[False, False, False], monitor='dummy_acc')
        self.eval_patience(min_delta=1., patience=1,
                           val_list=[9., 9.4, 10.], expected_result=[False, False, True], monitor='dummy_acc')
        self.eval_patience(min_delta=1., patience=2,
                           val_list=[9., 9.4, 10.], expected_result=[False, False, False], monitor='dummy_acc')
        self.eval_patience(min_delta=1., patience=2,
                           val_list=[9., 9., 9.4, 10.], expected_result=[False, False, False, True], monitor='dummy_acc')
        self.eval_patience(min_delta=1., patience=2,
                           val_list=[9., 9., 10.0, 10.0], expected_result=[False, False, False, True], monitor='dummy_acc')
        self.eval_patience(min_delta=1., patience=2,
                           val_list=[9., 9., 10.0, 10.1], expected_result=[False, False, False, False], monitor='dummy_acc')

    def eval_patience(self, min_delta, patience, val_list, expected_result, monitor):
        callback = EarlyStopping(monitor=monitor, min_delta=min_delta, patience=patience)
        train_loop = TrainLoop(NetUnifiedBatchFeed(), None, None, None, None, None)
        train_loop.callbacks_handler.register_callbacks([callback])
        self.assertFalse(train_loop.early_stop)

        result = []

        for val in val_list:
            train_loop.insert_metric_result_into_history(monitor, val)
            callback.on_epoch_end()
            result.append(train_loop.early_stop)

        self.assertEqual(result, expected_result)


class TestDataSubsetTestRun(unittest.TestCase):
    def test_train_loader_execute_callback(self):
        train_loader = DataLoader(TensorDataset(torch.Tensor(1000, 10)), batch_size=100)

        callback = DataSubsetTestRun(num_train_batches=3)
        train_loop = TrainLoop(NetUnifiedBatchFeed(), train_loader, None, None, None, None)
        train_loop.callbacks_handler.register_callbacks([callback])
        train_loop.callbacks_handler.execute_train_begin()

        self.assertEqual(len(train_loop.train_loader), 3)
        self.assertEqual(type(train_loop.train_loader), list)
        for batch in train_loop.train_loader:
            self.assertEqual(batch[0].shape, (100, 10))

    def test_all_loaders_execute_callback(self):
        train_loader = DataLoader(TensorDataset(torch.Tensor(1000, 10)), batch_size=100)
        val_loader = DataLoader(TensorDataset(torch.Tensor(500, 10)), batch_size=50)
        test_loader = DataLoader(TensorDataset(torch.Tensor(200, 10)), batch_size=30)

        callback = DataSubsetTestRun(num_train_batches=3, num_val_batches=2, num_test_batches=2)
        train_loop = TrainLoop(NetUnifiedBatchFeed(), train_loader, val_loader, test_loader, None, None)
        train_loop.callbacks_handler.register_callbacks([callback])
        train_loop.callbacks_handler.execute_train_begin()

        self.assertEqual(len(train_loop.train_loader), 3)
        self.assertEqual(type(train_loop.train_loader), list)
        for batch in train_loop.train_loader:
            self.assertEqual(batch[0].shape, (100, 10))

        self.assertEqual(len(train_loop.validation_loader), 2)
        self.assertEqual(type(train_loop.validation_loader), list)
        for batch in train_loop.validation_loader:
            self.assertEqual(batch[0].shape, (50, 10))

        self.assertEqual(len(train_loop.test_loader), 2)
        self.assertEqual(type(train_loop.test_loader), list)
        for batch in train_loop.test_loader:
            self.assertEqual(batch[0].shape, (30, 10))

    def test_exception_throw(self):
        train_loader = DataLoader(TensorDataset(torch.Tensor(1000, 10)), batch_size=100)

        callback = DataSubsetTestRun(num_train_batches=3, num_val_batches=2)
        train_loop = TrainLoop(NetUnifiedBatchFeed(), train_loader, None, None, None, None)

        with self.assertRaises(ValueError):
            train_loop.callbacks_handler.register_callbacks([callback])

        callback = DataSubsetTestRun(num_train_batches=3, num_test_batches=2)
        train_loop = TrainLoop(NetUnifiedBatchFeed(), train_loader, None, None, None, None)

        with self.assertRaises(ValueError):
            train_loop.callbacks_handler.register_callbacks([callback])

        callback = DataSubsetTestRun(num_train_batches=3, num_val_batches=2, num_test_batches=2)
        train_loop = TrainLoop(NetUnifiedBatchFeed(), train_loader, None, None, None, None)

        with self.assertRaises(ValueError):
            train_loop.callbacks_handler.register_callbacks([callback])


def cb_fn(tl):
    tl.epoch = 100


class TestLambdaOnTrainLoop(unittest.TestCase):
    def test_execute_callback(self):
        callback = FunctionOnTrainLoop(cb_fn)
        train_loop = TrainLoop(NetUnifiedBatchFeed(), None, None, None, None, None)
        train_loop.callbacks_handler.register_callbacks([callback])
        self.assertEqual(train_loop.epoch, 0)
        callback.execute_callback()
        self.assertEqual(train_loop.epoch, 100)

    def test_on_train_loop_registration(self):
        callback = FunctionOnTrainLoop(cb_fn, tl_registration=True)
        train_loop = TrainLoop(NetUnifiedBatchFeed(), None, None, None, None, None)
        train_loop.callbacks_handler.register_callbacks([callback])
        self.assertEqual(train_loop.epoch, 100)

    def test_on_epoch_begin(self):
        callback = FunctionOnTrainLoop(cb_fn, epoch_begin=True)
        train_loop = TrainLoop(NetUnifiedBatchFeed(), None, None, None, None, None)
        train_loop.callbacks_handler.register_callbacks([callback])
        self.assertEqual(train_loop.epoch, 0)
        train_loop.callbacks_handler.execute_epoch_begin()
        self.assertEqual(train_loop.epoch, 100)

    def test_on_train_end(self):
        callback = FunctionOnTrainLoop(cb_fn, train_end=True)
        train_loop = TrainLoop(NetUnifiedBatchFeed(), None, None, None, None, None)
        train_loop.callbacks_handler.register_callbacks([callback])
        self.assertEqual(train_loop.epoch, 0)
        train_loop.callbacks_handler.execute_train_end()
        self.assertEqual(train_loop.epoch, 100)

    def test_on_after_gradient_update(self):
        callback = FunctionOnTrainLoop(cb_fn, after_gradient_update=True)
        train_loop = TrainLoop(NetUnifiedBatchFeed(), None, None, None, None, None)
        train_loop.callbacks_handler.register_callbacks([callback])
        self.assertEqual(train_loop.epoch, 0)
        self.assertTrue(train_loop.grad_cb_used)
        train_loop.callbacks_handler.execute_gradient_update()
        self.assertEqual(train_loop.epoch, 100)

    def test_on_after_optimizer_step(self):
        callback = FunctionOnTrainLoop(cb_fn, after_optimizer_step=True)
        train_loop = TrainLoop(NetUnifiedBatchFeed(), None, None, None, None, None)
        train_loop.callbacks_handler.register_callbacks([callback])
        self.assertEqual(train_loop.epoch, 0)
        self.assertTrue(train_loop.grad_cb_used)
        train_loop.callbacks_handler.execute_optimizer_step()
        self.assertEqual(train_loop.epoch, 100)

    def test_on_after_gradient_update_register_combo(self):
        def cb_fn_add(tl):
            tl.epoch += 100
        callback = FunctionOnTrainLoop(cb_fn_add, tl_registration=True, after_gradient_update=True)
        train_loop = TrainLoop(NetUnifiedBatchFeed(), None, None, None, None, None)
        train_loop.callbacks_handler.register_callbacks([callback])
        self.assertEqual(train_loop.epoch, 100)
        self.assertTrue(train_loop.grad_cb_used)
        train_loop.callbacks_handler.execute_gradient_update()
        self.assertEqual(train_loop.epoch, 200)
