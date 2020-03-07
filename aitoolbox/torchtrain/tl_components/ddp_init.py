from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from aitoolbox.torchtrain.callbacks.ddp import DistributedSamplerSetEpoch


class DDPInitializer:
    def __init__(self, train_loop_obj):
        """

        Args:
            train_loop_obj (aitoolbox.torchtrain.train_loop.TrainLoop):
        """
        self.train_loop_obj = train_loop_obj

        from torch.utils.data import RandomSampler
        if (self.train_loop_obj.train_loader is not None and isinstance(self.train_loop_obj.train_loader.sampler, RandomSampler)) or \
                (self.train_loop_obj.validation_loader is not None and isinstance(self.train_loop_obj.validation_loader.sampler, RandomSampler)) or \
                (self.train_loop_obj.test_loader is not None and isinstance(self.train_loop_obj.test_loader.sampler, RandomSampler)):
            raise ValueError('sampler option is mutually exclusive with shuffle')
        
    def add_distributed_samplers(self, world_size, rank, train_data_shuffle):
        """

        Args:
            world_size (int): world size of for the distributed training
            rank (int): rank of the current process
            train_data_shuffle (bool): should train loader return shuffled data
        """
        train_sampler = val_sampler = test_sampler = None

        if self.train_loop_obj.train_loader is not None:
            self.train_loop_obj.train_loader, train_sampler = \
                self.build_loader_sampler(self.train_loop_obj.train_loader, shuffle=train_data_shuffle,
                                          world_size=world_size, rank=rank)

        if self.train_loop_obj.validation_loader is not None:
            self.train_loop_obj.validation_loader, val_sampler = \
                self.build_loader_sampler(self.train_loop_obj.validation_loader, shuffle=train_data_shuffle,
                                          world_size=world_size, rank=rank)

        if self.train_loop_obj.test_loader is not None:
            self.train_loop_obj.test_loader, test_sampler = \
                self.build_loader_sampler(self.train_loop_obj.test_loader, shuffle=train_data_shuffle,
                                          world_size=world_size, rank=rank)

        self.train_loop_obj.callbacks_handler.register_callbacks([
            DistributedSamplerSetEpoch(train_sampler, val_sampler, test_sampler)
        ])

    @staticmethod
    def build_loader_sampler(data_loader, shuffle, world_size, rank):
        """Replicate given data loader with added distributed sampler

        Args:
            data_loader (DataLoader): original single process data loader without the distributed sampler
            shuffle (bool): should the added sampler be returning examples in the shuffled order
            world_size (int): world size of for the distributed training
            rank (int): rank of the current process

        Returns:
            DataLoader, DistributedSampler: new data loader with the sampler, reference to the distributed sampler
                included in the new data loader
        """
        data_loader_args = {
            'dataset': data_loader.dataset,
            'batch_size': data_loader.batch_size,
            'shuffle': False,
            'num_workers': data_loader.num_workers,
            'collate_fn': data_loader.collate_fn,
            'pin_memory': data_loader.pin_memory,
            'drop_last': data_loader.drop_last,
            'timeout': data_loader.timeout,
            'worker_init_fn': data_loader.worker_init_fn
        }

        ddp_sampler = DistributedSampler(dataset=data_loader.dataset, shuffle=shuffle,
                                         num_replicas=world_size, rank=rank)
        data_loader_args['sampler'] = ddp_sampler
        data_loader_sampler = DataLoader(**data_loader_args)
        return data_loader_sampler, ddp_sampler
