import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from aitoolbox.torchtrain.callbacks.ddp import DistributedSamplerSetEpoch


class DDPHandler:
    def __init__(self, train_loop_obj):
        """Distributed Data Parallel process handler for the TrainLoop

        Args:
            train_loop_obj (aitoolbox.torchtrain.train_loop.TrainLoop): reference to the encapsulating TrainLoop
        """
        self.train_loop_obj = train_loop_obj

        from torch.utils.data import SequentialSampler, RandomSampler
        if (self.train_loop_obj.train_loader is not None and
            not isinstance(self.train_loop_obj.train_loader.sampler, (SequentialSampler, RandomSampler))) or \
                (self.train_loop_obj.validation_loader is not None and
                 not isinstance(self.train_loop_obj.validation_loader.sampler, (SequentialSampler, RandomSampler))) or \
                (self.train_loop_obj.test_loader is not None and
                 not isinstance(self.train_loop_obj.test_loader.sampler, (SequentialSampler, RandomSampler))):
            print('Provided DataLoaders have a non-standard data sampler (SequentialSampler or RandomSampler). '
                  'DDP required DistributedSampler only supports sequential data reading or randomly shuffled '
                  'data reading.')

    def add_distributed_samplers(self, world_size, rank):
        """Add Distributed Samplers needed for DDP to the normal single process DataLoader provided to the TrainLoop

        Args:
            world_size (int): world size of for the distributed training
            rank (int): rank of the current process
        """
        from torch.utils.data import RandomSampler
        train_sampler = val_sampler = test_sampler = None

        if self.train_loop_obj.train_loader is not None:
            self.train_loop_obj.train_loader, train_sampler = \
                self.build_loader_sampler(
                    self.train_loop_obj.train_loader,
                    shuffle=isinstance(self.train_loop_obj.train_loader.sampler, RandomSampler),
                    world_size=world_size, rank=rank
                )

        if self.train_loop_obj.validation_loader is not None:
            self.train_loop_obj.validation_loader, val_sampler = \
                self.build_loader_sampler(
                    self.train_loop_obj.validation_loader,
                    shuffle=isinstance(self.train_loop_obj.validation_loader.sampler, RandomSampler),
                    world_size=world_size, rank=rank
                )

        if self.train_loop_obj.test_loader is not None:
            self.train_loop_obj.test_loader, test_sampler = \
                self.build_loader_sampler(
                    self.train_loop_obj.test_loader,
                    shuffle=isinstance(self.train_loop_obj.test_loader.sampler, RandomSampler),
                    world_size=world_size, rank=rank
                )

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

    def mp_sync(self, data, concat_mp_data=True, double_precision=False):
        """Multiprocess data sync

        Share input data between all the active processes so that every process has all the values from
        all the processes. This way we can achieve the same state of the data across all the parallel processes.

        Args:
            data (torch.Tensor, list, float, int): data to be synchronized between processes.
                In case this is torch.Tensor, resulting output the device location will be preserved.
            concat_mp_data (bool): should the returned list of collected tensors be concatenated into a single list
                of values
            double_precision (bool): in case the ``data`` parameter is not already a Tensor, the function wraps given
                data into a Tensor. By default, it uses PyTorch default 32 bit precision float tensor. If this parameter
                is set to ``True`` however, the double precision 64 bit tensor will be created. This is useful
                for example if input data is in 64 bit, and we want to prevent precision reduction when syncing the data
                across the workers.

        Returns:
            torch.Tensor: list of `data` variable values synced across all the active processes
        """
        input_data_device = 'cpu'
        if not hasattr(data, '__len__'):
            data = [data]

        if isinstance(data, torch.Tensor):
            input_data_device = data.device.type
        else:
            data = torch.tensor(data, dtype=torch.float32 if not double_precision else torch.float64)

        data_tensor_wrap = data.to(self.train_loop_obj.device)
        mp_data = [torch.zeros_like(data_tensor_wrap) for _ in range(dist.get_world_size())]
        dist.all_gather(mp_data, data_tensor_wrap)
        
        if concat_mp_data:
            mp_data = torch.cat(mp_data)
        # at this point all the data in mp_data still on the GPUs, optionally move back to CPU
        if input_data_device == 'cpu':
            mp_data = mp_data.cpu()

        return mp_data

    def mp_sync_dict_of_lists(self, dict_list_data):
        """Multiprocess dict of lists sync

        Convenience wrapper around the `mp_sync()` for the specific case of dict of lists syncing.

        Args:
            dict_list_data (dict): dict of lists to be synchronized across the processes

        Returns:
            dict: synchronized dict of lists with combined values gathered from all the active processes
        """
        return {k: self.mp_sync(values_list).tolist() for k, values_list in dict_list_data.items()}
