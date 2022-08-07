import os
import unittest
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist

from tests.test_torchtrain.test_model import MyModel

from aitoolbox import TTModel, TTDataParallel
from aitoolbox.torchtrain.parallel import TTParallelBase, TTDistributedDataParallel


class TestTTParallelBase(unittest.TestCase):
    def test_init_default(self):
        model = MyModel()
        model_parallel = TTParallelBase(model)

        self.assertTrue(hasattr(model_parallel, 'get_loss'))
        self.assertTrue(hasattr(model_parallel, 'get_loss_eval'))
        self.assertTrue(hasattr(model_parallel, 'get_predictions'))

        self.assertTrue(hasattr(model_parallel, 'my_new_fn'))
        self.assertTrue(hasattr(model_parallel, 'get_model_level_str'))

    def test_init_attr_transfer(self):
        model = MyModel()
        model_parallel = TTParallelBase(model)

        self.assertTrue(hasattr(model_parallel, 'model_level_str'))

        self.assertEqual(model_parallel.get_model_level_str(), 'test string')
        self.assertEqual(model_parallel.get_model_level_str(), model.model_level_str)

    def test_core_model_transferred_fns(self):
        model = MyModel()
        model_parallel = TTParallelBase(model)
        self.assertEqual(model_parallel.get_loss(None, None, None), 'loss_return')
        self.assertEqual(model_parallel.get_loss_eval(None, None, None), 'loss_eval_return')
        self.assertEqual(model_parallel.get_predictions(None, None), 'predictions_return')
        self.assertEqual(model_parallel.my_new_fn(), 'my_new_fn return value')


class TestTTDataParallel(unittest.TestCase):
    def test_init_default(self):
        model = MyModel()
        model_parallel = TTDataParallel(model)

        self.assertTrue(hasattr(model_parallel, 'get_loss'))
        self.assertTrue(hasattr(model_parallel, 'get_loss_eval'))
        self.assertTrue(hasattr(model_parallel, 'get_predictions'))

        self.assertTrue(hasattr(model_parallel, 'my_new_fn'))
        self.assertTrue(hasattr(model_parallel, 'get_model_level_str'))

    def test_inheritance(self):
        model = MyModel()
        model_parallel = TTDataParallel(model)

        self.assertTrue(isinstance(model_parallel, nn.DataParallel))
        self.assertTrue(isinstance(model_parallel, TTParallelBase))

    def test_init_attr_transfer(self):
        model = MyModel()
        model_parallel = TTDataParallel(model)

        self.assertTrue(hasattr(model_parallel, 'model_level_str'))

        self.assertEqual(model_parallel.get_model_level_str(), 'test string')
        self.assertEqual(model_parallel.get_model_level_str(), model.model_level_str)

    def test_core_model_transferred_fns(self):
        model = MyModel()
        model_parallel = TTDataParallel(model)
        self.assertEqual(model_parallel.get_loss(None, None, None), 'loss_return')
        self.assertEqual(model_parallel.get_loss_eval(None, None, None), 'loss_eval_return')
        self.assertEqual(model_parallel.get_predictions(None, None), 'predictions_return')
        self.assertEqual(model_parallel.my_new_fn(), 'my_new_fn return value')

    def test_dp_model_wrap_forward_attribute_access(self):
        model = DPModel()
        dp_model = TTDataParallel(model)

        for i in range(1, 101):
            self.assertEqual(dp_model(100), i)

    def test_dp_model_wrap_get_loss_attribute_access(self):
        model = DPModel()
        dp_model = TTDataParallel(model)

        for i in range(1, 101):
            self.assertEqual(dp_model.get_loss(100, None, None),
                             (i, i, 'my_new_fn return value', 'test string'))

    def test_dp_model_wrap_get_predictions_attribute_access(self):
        model = DPModel()
        dp_model = TTDataParallel(model)

        for i in range(1, 101):
            self.assertEqual(dp_model.get_predictions(100, None),
                             (i, i, 'my_new_fn return value', 'test string'))

    def test_dp_model_wrap_all_methods_mix_attribute_access(self):
        model = DPModel()
        dp_model = TTDataParallel(model)

        for i in range(1, 101):
            self.assertEqual(dp_model(100), i)

        for i in range(1, 101):
            self.assertEqual(dp_model.get_loss(100, None, None),
                             (i + 100, i, 'my_new_fn return value', 'test string'))

    def test_dp_model_wrap_unreachable_attribute_access(self):
        model = DPModel()
        dp_model = TTDataParallel(model)

        self.assertEqual(dp_model.get_loss(100, None, None), (1, 1, 'my_new_fn return value', 'test string'))

        with self.assertRaises(AttributeError):
            dp_model.get_loss(100, None, 'unreachable')


class TestTTDistributedDataParallel(unittest.TestCase):
    def test_init_default(self):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '8888'
        mp.spawn(self._init_default, nprocs=2)

    @staticmethod
    def _init_default(gpu):
        dist.init_process_group(backend='gloo', init_method='env://', world_size=2, rank=gpu)
        model = DPModel()
        ddp_model = TTDistributedDataParallel(model)

        assert hasattr(ddp_model, 'get_loss')
        assert hasattr(ddp_model, 'get_loss_eval')
        assert hasattr(ddp_model, 'get_predictions')

        assert hasattr(ddp_model, 'my_new_fn')
        assert hasattr(ddp_model, 'get_model_level_str')

    def test_inheritance(self):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '8888'
        mp.spawn(self._inheritance, nprocs=2)

    @staticmethod
    def _inheritance(gpu):
        dist.init_process_group(backend='gloo', init_method='env://', world_size=2, rank=gpu)
        model = DPModel()
        ddp_model = TTDistributedDataParallel(model)

        assert isinstance(ddp_model, nn.parallel.DistributedDataParallel)
        assert isinstance(ddp_model, TTDistributedDataParallel)

    def test_init_attr_transfer(self):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '8888'
        mp.spawn(self._init_attr_transfer, nprocs=2)

    @staticmethod
    def _init_attr_transfer(gpu):
        dist.init_process_group(backend='gloo', init_method='env://', world_size=2, rank=gpu)
        model = DPModel()
        ddp_model = TTDistributedDataParallel(model)

        assert hasattr(ddp_model, 'model_level_str')

        assert ddp_model.get_model_level_str() == 'test string'
        assert ddp_model.get_model_level_str() == model.model_level_str

    def test_core_model_transferred_fns(self):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '8888'
        mp.spawn(self._core_model_transferred_fns, nprocs=2)

    @staticmethod
    def _core_model_transferred_fns(gpu):
        dist.init_process_group(backend='gloo', init_method='env://', world_size=2, rank=gpu)
        model = MyModel()
        ddp_model = TTDistributedDataParallel(model)

        assert ddp_model.get_loss(None, None, None) == 'loss_return'
        assert ddp_model.get_loss_eval(None, None, None) == 'loss_eval_return'
        assert ddp_model.get_predictions(None, None) == 'predictions_return'
        assert ddp_model.my_new_fn() == 'my_new_fn return value'

    def test_ddp_model_wrap_forward_attribute_access(self):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '8888'
        mp.spawn(self._ddp_model_wrap_forward_attribute_access, nprocs=2)

    @staticmethod
    def _ddp_model_wrap_forward_attribute_access(gpu):
        dist.init_process_group(backend='gloo', init_method='env://', world_size=2, rank=gpu)
        model = DPModel()
        ddp_model = TTDistributedDataParallel(model)

        for i in range(1, 101):
            assert ddp_model(100) == i

    def test_ddp_model_wrap_get_loss_attribute_access(self):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '8888'
        mp.spawn(self._ddp_model_wrap_get_loss_attribute_access, nprocs=2)

    @staticmethod
    def _ddp_model_wrap_get_loss_attribute_access(gpu):
        dist.init_process_group(backend='gloo', init_method='env://', world_size=2, rank=gpu)
        model = DPModel()
        ddp_model = TTDistributedDataParallel(model)

        for i in range(1, 101):
            assert ddp_model.get_loss(100, None, None) == (i, i, 'my_new_fn return value', 'test string')

    def test_ddp_model_wrap_get_predictions_attribute_access(self):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '8888'
        mp.spawn(self._ddp_model_wrap_get_predictions_attribute_access, nprocs=2)

    @staticmethod
    def _ddp_model_wrap_get_predictions_attribute_access(gpu):
        dist.init_process_group(backend='gloo', init_method='env://', world_size=2, rank=gpu)
        model = DPModel()
        ddp_model = TTDistributedDataParallel(model)

        for i in range(1, 101):
            assert ddp_model.get_predictions(100, None) == (i, i, 'my_new_fn return value', 'test string')

    def test_ddp_model_wrap_all_methods_mix_attribute_access(self):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '8888'
        mp.spawn(self._ddp_model_wrap_all_methods_mix_attribute_access, nprocs=2)

    @staticmethod
    def _ddp_model_wrap_all_methods_mix_attribute_access(gpu):
        dist.init_process_group(backend='gloo', init_method='env://', world_size=2, rank=gpu)
        model = DPModel()
        ddp_model = TTDistributedDataParallel(model)

        for i in range(1, 101):
            assert ddp_model(100) == i

        for i in range(1, 101):
            assert ddp_model.get_loss(100, None, None) == (i + 100, i, 'my_new_fn return value', 'test string')

    def test_ddp_model_wrap_unreachable_attribute_access(self):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '8888'
        mp.spawn(self._ddp_model_wrap_unreachable_attribute_access, nprocs=2)

    @staticmethod
    def _ddp_model_wrap_unreachable_attribute_access(gpu):
        dist.init_process_group(backend='gloo', init_method='env://', world_size=2, rank=gpu)
        model = DPModel()
        ddp_model = TTDistributedDataParallel(model)

        assert ddp_model.get_loss(100, None, None) == (1, 1, 'my_new_fn return value', 'test string')

        with unittest.TestCase().assertRaises(AttributeError):
            ddp_model.get_loss(100, None, 'unreachable')


class DPModel(TTModel):
    def __init__(self):
        super().__init__()
        self.model_level_str = 'test string'

        self.linear = nn.Linear(10, 10)

        self.forward_ctr = 0
        self.get_loss_ctr = 0
        self.get_predictions_ctr = 0

        self.unreachable_attr = "Can't get me"

        self.transfer_model_attributes = ['model_level_str', 'get_loss_ctr', 'get_predictions_ctr']

    def forward(self, batch):
        self.forward_ctr += 1
        return self.forward_ctr

    def get_loss(self, batch_data, criterion, device):
        forward_ctr = self(batch_data)
        self.get_loss_ctr += 1

        my_fn_return = self.my_new_fn()
        model_level_str = self.get_model_level_str()

        if device == 'unreachable':
            my_fn_return = self.unreachable_attr

        return forward_ctr, self.get_loss_ctr, my_fn_return, model_level_str

    def get_predictions(self, batch_data, device):
        forward_ctr = self(batch_data)
        self.get_predictions_ctr += 1

        my_fn_return = self.my_new_fn()
        model_level_str = self.get_model_level_str()

        return forward_ctr, self.get_predictions_ctr, my_fn_return, model_level_str

    def my_new_fn(self):
        return 'my_new_fn return value'

    def get_model_level_str(self):
        return self.model_level_str
