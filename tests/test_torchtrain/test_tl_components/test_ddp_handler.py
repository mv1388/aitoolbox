import unittest

from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tests.utils import SmallFFNet

from aitoolbox import TrainLoop, BasicDataset
from aitoolbox.torchtrain.train_loop.components.ddp_handler import DDPHandler
from aitoolbox.torchtrain.callbacks.ddp import DistributedSamplerSetEpoch


class TestDDPHandler(unittest.TestCase):
    def test_add_distributed_samplers(self):
        train_loader = DataLoader(BasicDataset([(1, 2) for _ in range(100)]), shuffle=True)
        val_loader = DataLoader(BasicDataset([(1, 2) for _ in range(100)]))
        test_loader = DataLoader(BasicDataset([(1, 2) for _ in range(100)]))
        train_loop = TrainLoop(SmallFFNet(), train_loader, val_loader, test_loader, None, None)

        self.assertEqual(len(train_loop.callbacks), 0)

        ddp_handler = DDPHandler(train_loop)
        ddp_handler.add_distributed_samplers(world_size=4, rank=1)

        self.assertTrue(isinstance(train_loop.callbacks[0], DistributedSamplerSetEpoch))
        self.assertEqual(len(train_loop.callbacks_handler.cbs_on_epoch_begin), 1)
        self.assertTrue(isinstance(train_loop.callbacks_handler.cbs_on_epoch_begin[0], DistributedSamplerSetEpoch))

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

        self.assertEqual(train_loop.test_loader.sampler.dataset, train_loop.test_loader.dataset)
        self.assertEqual(train_loop.test_loader.sampler.num_replicas, 4)
        self.assertEqual(train_loop.test_loader.sampler.rank, 1)
        self.assertFalse(train_loop.test_loader.sampler.shuffle)

    def test_add_distributed_samplers_no_train_shuffle(self):
        train_loader = DataLoader(BasicDataset([(1, 2) for _ in range(100)]))
        val_loader = DataLoader(BasicDataset([(1, 2) for _ in range(100)]))
        test_loader = DataLoader(BasicDataset([(1, 2) for _ in range(100)]))
        train_loop = TrainLoop(SmallFFNet(), train_loader, val_loader, test_loader, None, None)

        ddp_handler = DDPHandler(train_loop)
        ddp_handler.add_distributed_samplers(world_size=4, rank=1)

        self.assertFalse(train_loop.train_loader.sampler.shuffle)
        self.assertFalse(train_loop.validation_loader.sampler.shuffle)
        self.assertFalse(train_loop.test_loader.sampler.shuffle)

    def test_add_distributed_samplers_all_data_loader_shuffle(self):
        train_loader = DataLoader(BasicDataset([(1, 2) for _ in range(100)]), shuffle=True)
        val_loader = DataLoader(BasicDataset([(1, 2) for _ in range(100)]), shuffle=True)
        test_loader = DataLoader(BasicDataset([(1, 2) for _ in range(100)]), shuffle=True)
        train_loop = TrainLoop(SmallFFNet(), train_loader, val_loader, test_loader, None, None)

        ddp_handler = DDPHandler(train_loop)
        ddp_handler.add_distributed_samplers(world_size=4, rank=1)

        self.assertTrue(train_loop.train_loader.sampler.shuffle)
        self.assertTrue(train_loop.validation_loader.sampler.shuffle)
        self.assertTrue(train_loop.test_loader.sampler.shuffle)

    def test_add_distributed_samplers_train_loader_only(self):
        train_loader = DataLoader(BasicDataset([(1, 2) for _ in range(100)]), shuffle=True)
        train_loop = TrainLoop(SmallFFNet(), train_loader, None, None, None, None)

        ddp_handler = DDPHandler(train_loop)
        ddp_handler.add_distributed_samplers(world_size=4, rank=1)

        self.assertTrue(isinstance(train_loop.train_loader.sampler, DistributedSampler))
        self.assertEqual(train_loop.callbacks[0].train_sampler, train_loop.train_loader.sampler)
        self.assertIsNone(train_loop.callbacks[0].validation_sampler)
        self.assertIsNone(train_loop.callbacks[0].test_sampler)

    def test_build_loader_sampler(self):
        data_loader = DataLoader(BasicDataset([(1, 2) for _ in range(100)]))

        ddp_data_loader, ddp_data_sampler = DDPHandler.build_loader_sampler(data_loader,
                                                                            shuffle=True, world_size=4, rank=1)

        self.assertTrue(isinstance(ddp_data_loader, DataLoader))
        self.assertTrue(isinstance(ddp_data_sampler, DistributedSampler))

        self.assertEqual(ddp_data_loader.sampler, ddp_data_sampler)

        self.assertEqual(ddp_data_loader.dataset, data_loader.dataset)
        self.assertEqual(ddp_data_loader.batch_size, data_loader.batch_size)
        self.assertEqual(ddp_data_loader.num_workers, data_loader.num_workers)
        self.assertEqual(ddp_data_loader.collate_fn, data_loader.collate_fn)
        self.assertEqual(ddp_data_loader.pin_memory, data_loader.pin_memory)
        self.assertEqual(ddp_data_loader.drop_last, data_loader.drop_last)
        self.assertEqual(ddp_data_loader.timeout, data_loader.timeout)
        self.assertEqual(ddp_data_loader.worker_init_fn, data_loader.worker_init_fn)

        self.assertEqual(ddp_data_sampler.dataset, ddp_data_loader.dataset)
        self.assertEqual(ddp_data_sampler.num_replicas, 4)
        self.assertEqual(ddp_data_sampler.rank, 1)
        self.assertTrue(ddp_data_sampler.shuffle)
