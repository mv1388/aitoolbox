from aitoolbox.torchtrain.callbacks.abstract import AbstractCallback


class DistributedSamplerSetEpoch(AbstractCallback):
    def __init__(self, train_sampler, validation_sampler, test_sampler):
        """Callback setting epoch index in the DistributedSamplers at the beginning of every epoch

        Args:
            train_sampler (torch.utils.data.distributed.DistributedSampler or None): 
                Distributed sampler for train loader
            validation_sampler (torch.utils.data.distributed.DistributedSampler or None):
                Distributed sampler for validation loader
            test_sampler (torch.utils.data.distributed.DistributedSampler or None):
                Distributed sampler for test loader
        """
        super().__init__("set_epoch for DistributedSamplers at the start of each epoch", execution_order=-100)
        self.train_sampler = train_sampler
        self.validation_sampler = validation_sampler
        self.test_sampler = test_sampler

    def on_epoch_begin(self):
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(self.train_loop_obj.epoch)
        if self.validation_sampler is not None:
            self.validation_sampler.set_epoch(self.train_loop_obj.epoch)
        if self.test_sampler is not None:
            self.test_sampler.set_epoch(self.train_loop_obj.epoch)
