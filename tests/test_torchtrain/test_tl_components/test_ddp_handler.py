import unittest

from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tests.utils import SmallFFNet

from aitoolbox import TrainLoop
from aitoolbox.torchtrain.tl_components.ddp_handler import DDPHandler
from aitoolbox import BasicDataset
from aitoolbox.torchtrain.callbacks.ddp import DistributedSamplerSetEpoch


class TestDDPHandler(unittest.TestCase):
    def test_init(self):
        train_loader = DataLoader(BasicDataset([(1, 2) for _ in range(100)]))
        val_loader = DataLoader(BasicDataset([(1, 2) for _ in range(100)]))
        train_loop_none_test_loader = TrainLoop(SmallFFNet(), train_loader, val_loader, None, None, None)
        DDPHandler(train_loop_none_test_loader)

        train_loader = DataLoader(BasicDataset([(1, 2) for _ in range(100)]), shuffle=True)
        val_loader = DataLoader(BasicDataset([(1, 2) for _ in range(100)]))
        test_loader = DataLoader(BasicDataset([(1, 2) for _ in range(100)]))
        train_loop = TrainLoop(SmallFFNet(), train_loader, val_loader, test_loader, None, None)
        with self.assertRaises(ValueError):
            DDPHandler(train_loop)

        train_loader = DataLoader(BasicDataset([(1, 2) for _ in range(100)]))
        val_loader = DataLoader(BasicDataset([(1, 2) for _ in range(100)]), shuffle=True)
        test_loader = DataLoader(BasicDataset([(1, 2) for _ in range(100)]))
        train_loop = TrainLoop(SmallFFNet(), train_loader, val_loader, test_loader, None, None)
        with self.assertRaises(ValueError):
            DDPHandler(train_loop)

        train_loader = DataLoader(BasicDataset([(1, 2) for _ in range(100)]))
        val_loader = DataLoader(BasicDataset([(1, 2) for _ in range(100)]))
        test_loader = DataLoader(BasicDataset([(1, 2) for _ in range(100)]), shuffle=True)
        train_loop = TrainLoop(SmallFFNet(), train_loader, val_loader, test_loader, None, None)
        with self.assertRaises(ValueError):
            DDPHandler(train_loop)

    def test_add_distributed_samplers(self):
        train_loader = DataLoader(BasicDataset([(1, 2) for _ in range(100)]))
        val_loader = DataLoader(BasicDataset([(1, 2) for _ in range(100)]))
        test_loader = DataLoader(BasicDataset([(1, 2) for _ in range(100)]))
        train_loop = TrainLoop(SmallFFNet(), train_loader, val_loader, test_loader, None, None)

        ddp_handler = DDPHandler(train_loop)
        ddp_handler.add_distributed_samplers(world_size=4, rank=1, train_data_shuffle=True)

        self.assertTrue(isinstance(train_loop.callbacks[0], DistributedSamplerSetEpoch))

        self.assertTrue(isinstance(train_loop.train_loader.sampler, DistributedSampler))
        self.assertTrue(isinstance(train_loop.validation_loader.sampler, DistributedSampler))
        self.assertTrue(isinstance(train_loop.test_loader.sampler, DistributedSampler))

        self.assertEqual(train_loop.callbacks[0].train_sampler, train_loop.train_loader.sampler)
        self.assertEqual(train_loop.callbacks[0].validation_sampler, train_loop.validation_loader.sampler)
        self.assertEqual(train_loop.callbacks[0].test_sampler, train_loop.test_loader.sampler)

        self.assertEqual(train_loop.train_loader.sampler.dataset, train_loop.train_loader.dataset)
        self.assertEqual(train_loop.train_loader.sampler.num_replicas, 4)
        self.assertEqual(train_loop.train_loader.sampler.rank, 1)
        self.assertTrue(train_loop.train_loader.sampler.shuffle)

        self.assertEqual(train_loop.validation_loader.sampler.dataset, train_loop.validation_loader.dataset)
        self.assertEqual(train_loop.validation_loader.sampler.num_replicas, 4)
        self.assertEqual(train_loop.validation_loader.sampler.rank, 1)
        self.assertFalse(train_loop.validation_loader.sampler.shuffle)
