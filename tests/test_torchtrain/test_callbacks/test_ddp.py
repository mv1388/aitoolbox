import unittest

from torch.utils.data.dataloader import DataLoader
from tests.utils import SmallFFNet

from aitoolbox import TrainLoop, BasicDataset
from aitoolbox.torchtrain.callbacks.ddp import DistributedSamplerSetEpoch
from aitoolbox.torchtrain.train_loop.components.ddp_handler import DDPHandler


class TestDistributedSamplerSetEpoch(unittest.TestCase):
    def test_callback_execution(self):
        train_loader = DataLoader(BasicDataset([(1, 2) for _ in range(100)]))
        val_loader = DataLoader(BasicDataset([(1, 2) for _ in range(100)]))
        test_loader = DataLoader(BasicDataset([(1, 2) for _ in range(100)]), shuffle=True)

        _, ddp_train_sampler = DDPHandler.build_loader_sampler(train_loader, shuffle=True, world_size=4, rank=1)
        _, ddp_val_sampler = DDPHandler.build_loader_sampler(val_loader, shuffle=False, world_size=4, rank=1)
        _, ddp_test_sampler = DDPHandler.build_loader_sampler(test_loader, shuffle=False, world_size=4, rank=1)

        callback = DistributedSamplerSetEpoch(ddp_train_sampler, ddp_val_sampler, ddp_test_sampler)

        train_loop = TrainLoop(SmallFFNet(), None, None, None, None, None)
        train_loop.callbacks_handler.register_callbacks([callback])

        train_loop.callbacks_handler.execute_epoch_begin()
        self.assertEqual(ddp_train_sampler.epoch, 0)
        self.assertEqual(ddp_val_sampler.epoch, 0)
        self.assertEqual(ddp_test_sampler.epoch, 0)

        train_loop.epoch = 5

        train_loop.callbacks_handler.execute_epoch_begin()
        self.assertEqual(ddp_train_sampler.epoch, 5)
        self.assertEqual(ddp_val_sampler.epoch, 5)
        self.assertEqual(ddp_test_sampler.epoch, 5)

        train_loop.epoch = 7

        train_loop.callbacks_handler.execute_epoch_begin()
        self.assertEqual(ddp_train_sampler.epoch, 7)
        self.assertEqual(ddp_val_sampler.epoch, 7)
        self.assertEqual(ddp_test_sampler.epoch, 7)
